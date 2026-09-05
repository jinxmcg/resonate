"""ResonatE: entities as unit-norm complex vectors, relations as linear
operators, a hop = apply the operator and renormalise, readout = real
inner product against the entity table scaled by exp(log_tau).

M = k^2 coefficients per entity. The "k x k modes" vocabulary is a
coordinate label kept for continuity with the project's history, not a
claim: no Fourier transform is taken anywhere, and the score is
invariant under any block-diagonal unitary change of basis (E -> UE,
H -> U H U^dagger), so nothing depends on the coordinates being
spectral (real=True tests the one measurable residue of "complex" at
equal parameter count).

Operator variants (constructor flags):
  block=True (every real-graph result, incl. ogbl-biokg): M/b
      independent b x b complex blocks per relation, initialised
      unitary via QR and free afterwards. Blocks do not commute, so
      mixed-relation chains are representable. ogbl-biokg: k=12, b=4.
  default (diagonal): one complex multiplier per coordinate;
      commuting, so a chain is one relation applied n times.
      Synthetic studies only.
  dense / unitary: ablations.
  real=True: the same model over the reals — E in R^{N x 2M},
      real b x b blocks on R^{2M}; identical real parameter count to
      the complex model at the same b.
"""

import math

import torch
import torch.nn as nn


def cnorm(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize complex vectors along the last dim."""
    n = z.abs().pow(2).sum(-1, keepdim=True).sqrt()
    return z / (n + eps)


class ResonatE(nn.Module):
    def __init__(self, n_entities: int = 1000, n_relations: int = 8,
                 k: int = 8, dense: bool = False, block: bool = False,
                 unitary: bool = False, block_size: int = 2,
                 tied_reverse: bool = False, ent_bias: bool = False,
                 rel_gain: bool = False, real: bool = False,
                 low_rank: int = 0, low_rank_local: bool = False):
        super().__init__()
        assert not (dense and block)
        assert not (real and (unitary or dense)), "real: block/diag only"
        if low_rank < 0 or (low_rank and (not block or tied_reverse)):
            raise ValueError("low_rank requires untied block operators and a nonnegative rank")
        if low_rank_local and not low_rank:
            raise ValueError("low_rank_local requires a positive low_rank")
        # unitary: phase-only relation params (H = e^{i.theta} always,
        # instead of free complex init'd on the unit circle). Diagonal
        # variant only — mechanism test for the depth-ceiling
        # angular-error story.
        assert not (unitary and (dense or block))
        self.unitary = unitary
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.k = k
        self.m = k * k
        self.dense = dense
        self.block = block
        self.real = real
        self.low_rank = low_rank
        self.low_rank_local = low_rank_local
        # width of the state vector: M complex, or 2M real (H22)
        width = 2 * self.m if real else self.m
        cdt = torch.float if real else torch.cfloat

        e = torch.randn(n_entities, width, dtype=cdt)
        self.E = nn.Parameter(cnorm(e))

        theta = (torch.rand(n_relations, self.m) * 2 - 1) * math.pi
        diag = torch.polar(torch.ones(n_relations, self.m), theta)
        if dense:
            # ablation: full M x M complex matrix per relation,
            # initialized as the diagonal model plus small noise
            h = torch.diag_embed(diag)
            h = h + 0.02 * torch.randn(n_relations, self.m, self.m,
                                       dtype=torch.cfloat)
            self.H = nn.Parameter(h)
        elif block:
            # non-commuting variant: M/b independent bxb unitary blocks
            # per relation (random unitaries via QR); b=2 is minimal.
            # tied_reverse: ids >= n_ops apply the adjoint (conj-T) of
            # the forward blocks — Re<Qh,t> = Re<h,Q^H t>, so both
            # directions score consistently at half the relation params
            assert width % block_size == 0
            self.block_size = block_size
            self.tied_reverse = tied_reverse
            n_ops = n_relations // 2 if tied_reverse else n_relations
            a = torch.randn(n_ops, width // block_size,
                            block_size, block_size, dtype=cdt)
            q, _ = torch.linalg.qr(a)
            self.H = nn.Parameter(q)
        elif real:
            # real diagonal: free scalars, init +-1 (no phases exist)
            self.H = nn.Parameter(torch.sign(torch.randn(n_relations, width)))
        elif unitary:
            self.theta = nn.Parameter(theta)
        else:
            self.H = nn.Parameter(diag)

        self.log_tau = nn.Parameter(torch.tensor(math.log(10.0)))

        # H10: additive per-entity score bias (popularity channel).
        # Target-side E norms already give a multiplicative channel;
        # this is the sign-independent additive one.
        self.b = nn.Parameter(torch.zeros(n_entities)) if ent_bias \
            else None

        # H16: per-(relation, direction) diagonal gain on the hopped
        # state at READOUT only (TripleRE's relation-specific
        # head/tail scaling). Hops stay unitary, so multi-hop
        # composition is untouched; the gain lets a relation collapse
        # or stress modes when comparing against targets (1-to-N,
        # hierarchy). Log-parametrised, init 0 = identity.
        self.gain = nn.Parameter(torch.zeros(n_relations, width)) \
            if rel_gain else None

        # H29: block operator plus U V*. The local control uses the
        # same factor shapes/parameter count, confined to each block.
        # Only V starts at zero: the correction is exactly zero but V
        # can learn on the first step. Preserve the RNG stream so the
        # sparse subclass draws identical entity rows in every arm.
        self.lr_u = self.lr_v = None
        if low_rank:
            with torch.random.fork_rng(devices=[]):
                fan_in = block_size if low_rank_local else width
                u = torch.randn(n_relations, width, low_rank, dtype=cdt)
                u = u / math.sqrt(fan_in)
            self.lr_u = nn.Parameter(u)
            self.lr_v = nn.Parameter(torch.zeros_like(u))

    def rows(self, idx: torch.Tensor) -> torch.Tensor:
        return self.E[idx]

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        return cnorm(self.E[idx])

    def out(self, z: torch.Tensor, r) -> torch.Tensor:
        """State as seen by the readout: the hopped state scaled by
        relation r's readout gain (identity without --rel-gain).
        r: (B,) relation ids or a single int."""
        if self.gain is None:
            return z
        return z * self.gain[r].exp()

    def hop(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """One hop: apply relation r's transfer function, renormalize."""
        source = z
        if self.unitary:
            th = self.theta[r]
            return cnorm(torch.polar(torch.ones_like(th), th) * z)
        if self.block and getattr(self, "tied_reverse", False):
            base = self.H.shape[0]
            h = self.H[torch.where(r >= base, r - base, r)]
            rev = r >= base
            if rev.any():
                h = torch.where(rev[:, None, None, None],
                                h.conj().transpose(-1, -2), h)
        else:
            h = self.H[r]
        if self.dense:
            z = torch.einsum('bnm,bm->bn', h, z)
        elif self.block:
            zb = z.reshape(z.shape[0], -1, self.block_size)
            z = torch.einsum('bkij,bkj->bki', h, zb).reshape(z.shape[0], -1)
        else:
            z = h * z
        if self.low_rank:
            u, v = self.lr_u[r], self.lr_v[r]
            if self.low_rank_local:
                shape = (len(source), -1, self.block_size, self.low_rank)
                u, v = u.reshape(shape), v.reshape(shape)
                x = source.reshape(len(source), -1, self.block_size)
                hidden = torch.einsum('bkir,bki->bkr', v.conj(), x)
                correction = torch.einsum('bkir,bkr->bki', u, hidden)
                correction = correction.reshape_as(source)
            else:
                hidden = torch.einsum('bmr,bm->br', v.conj(), source)
                correction = torch.einsum('bmr,br->bm', u, hidden)
            z = z + correction
        return cnorm(z)

    def readout(self, z: torch.Tensor, r=None) -> torch.Tensor:
        if r is not None:
            z = self.out(z, r)
        s = torch.real(z @ self.E.conj().t()) * self.log_tau.exp()
        return s if self.b is None else s + self.b

    def forward(self, head: torch.Tensor, r: torch.Tensor, n_hops: int):
        """Run an n-hop chain of relation r from head entities.

        head: (B,) entity ids; r: (B,) relation ids.
        Returns (logits over entities, list of intermediate states).
        """
        z = self.embed(head)
        states = []
        for _ in range(n_hops):
            z = self.hop(z, r)
            states.append(z)
        return self.readout(z, r), states

    def n_params(self) -> int:
        """Real-valued parameter count (complex counts as 2)."""
        return sum(2 * p.numel() if p.is_complex() else p.numel()
                   for p in self.parameters())
