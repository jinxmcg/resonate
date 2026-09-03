# ResonatE on ogbl-biokg

A 27M-parameter knowledge-graph embedding — unit-norm complex spectra
for entities, block-unitary operators for relations — plus four
label-free retrieval features read from the training graph, evaluated
on [ogbl-biokg](https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg)
with the official OGB Evaluator. Three configurations, ten
independent runs each, one test read per run.

| | test MRR | valid MRR | hits@1 | hits@3 | hits@10 | params |
|---|---|---|---|---|---|---|
| A. single model | 0.8118 ± 0.0013 | 0.8124 ± 0.0013 | 0.7408 | 0.8621 | 0.9391 | 27,124,129 |
| B. model + retrieval features | 0.8425 ± 0.0007 | 0.8434 ± 0.0007 | 0.7826 | 0.8843 | 0.9507 | 27,124,129 |
| C. distilled model + retrieval features | **0.8505 ± 0.0004** | 0.8515 ± 0.0004 | 0.7941 | 0.8904 | 0.9524 | 27,124,129 |

Mean ± unbiased std over seeds 0–9 (`python scripts/summarize.py`).
Validation for B and C is the in-sample value of the blend weights,
which are selected on the full validation split; the held-out
estimate of the same procedure (weights fit on a random half of
validation, measured on the other half) is 0.8429 ± 0.0007 and
0.8512 ± 0.0005, within 0.001 of test for every run. The distilled
model alone scores 0.8287 ± 0.0006 on validation.

For scale, the public leaderboard as of 2026-09-02: ComplEx 0.8095
(188M params), PairRE 0.8164 (188M), AutoSF 0.8309 (94M), TripleRE
0.8348 (470M), ComplEx-RP 0.8492 (188M), AutoBLM 0.8536 (192M),
UniBi 0.8550 (182M), ComplEx² 0.8583 (188M), RelEns 0.9618 (849M,
ensemble). The parameter count above counts every complex coefficient
as two reals (the convention used for ComplEx's 188M); the literal
`sum(p.numel())` over the module's parameters is 13,562,065.

The paper: [`paper/resonate.pdf`](paper/resonate.pdf).

## The model (`resonate.py`)

- **Entities**: `E ∈ ℂ^{N×144}` (k=12, M=k²). As a query source an
  entity is read through `cnorm(E[h])` (unit norm); as a target it is
  the raw row `E[t]`, so target norms act as a learned per-entity
  popularity channel.
- **Relations**: for each of the 102 (relation, direction) pairs, 36
  independent 4×4 complex blocks, initialised unitary by QR and then
  free. A hop is `cnorm(blockdiag(H_r) · z)`.
- **Score**: `Re⟨hop(cnorm(E[h]), r), E[t]⟩ · exp(log_tau)`.
- **Training** (`train_ogb.py`): each batch is 2,048 triples of one
  relation (relations sampled by frequency), scored in a random
  direction against one shared set of 4,096 negatives drawn uniformly
  from the target type's entity range — the candidate distribution
  the Evaluator uses; sampled softmax cross-entropy plus
  0.1·‖z − E[t]‖² on the hopped state; gradient clip 1.0; Adam 5e-3
  with cosine decay; 50,000 steps (≈21 epochs).
- **Distillation** (row C, `--distill`): the ten frozen row-A models
  score the batch's [positive | negatives]; their mean logits, at
  temperature 1, are the target of a KL term (weight 1) added to the
  student's loss. Teachers only ever see training triples; the
  student runs alone at inference.

## The retrieval features (`analogy_member.py`, `jaccard_member.py`)

For a query (h, r, ?) and each of the 501 candidates c (positive +
500 OGB negatives), look up the training-graph entities that already
hold the edge (·, r, c) and score c by how similar they are to h:

- **analogy**: max over holders of cos(e_h, e_holder)³ under the
  model's own embeddings; **analogy_t3**: mean of the top three.
- **jaccard**: the same with edge-set Jaccard similarity between h
  and the holder in the training graph, model-free; **jaccard_t3**
  likewise.

They read only training edges — validation and test edges are never
looked up — so the features are as label-free as the embedding. Alone
they reach 0.64–0.76 validation MRR; blended with the model they add
+0.03.

**Blend** (`ensemble_weights.py`, `freeze_test.py`): per row, each
member's 501 scores are z-normalised; for every (relation, direction)
group the weight vector is picked from a small candidate set
{uniform, top-1/3/5/9 by member MRR, softmax(η·MRR) for
η ∈ {20, 50, 100}}, plus the parent level's choice, by validation MRR,
hierarchically (global → relation family → relation; a group only
gets its own weights above 4,000 validation rows). It is a selection
among at most nine candidates per group, not a gradient fit; no
parameter is trained on validation.

## Setup

```
git clone https://github.com/jinxmcg/resonate && cd resonate
uv sync                        # Python ≥ 3.12; torch ≥ 2.6, numpy, scipy, ogb
```

`ogbl-biokg` (≈ 2.9 GB) is downloaded by `ogb` into `data_ogb/` on
first use. Reported runs used torch 2.6.0+cu124 (GTX 1080 Ti) and
torch 2.11.0+cu128 (RTX 5090); `verify.py` confirms checkpoints
trained on one reproduce on the other to < 3·10⁻⁴.

## Verify without retraining (≈ 10 min + downloads)

```
scripts/fetch_checkpoints.sh           # 20 checkpoints, ~2.2 GB, from the GitHub release
uv run python verify.py --device cuda  # split disjointness + re-evaluates the 10 row-A models
python scripts/summarize.py            # the table above from results/
```

`verify.py` asserts the train/valid/test triple sets are pairwise
disjoint as loaded, re-scores every `checkpoints/model_final_seed*.pt`
on test with the official Evaluator and checks each against the MRR in
its `results/final_seed*.log` (tolerance 3·10⁻⁴). `results/` also
holds the frozen blend weights (`frozen_*.npz`) and the committed test
receipts (`committed_*.json/.log`) of rows B and C.

To re-apply a frozen blend, the score caches must exist (see next
section; ≈ 1 h per seed for the analogy features) — after which
`freeze_test.py` with the same members is deterministic and reproduces
the receipt exactly.

## Reproduce from scratch

```
scripts/run_campaign.sh     # row A: 10 models        ~4 min/seed on a 5090, ~18 min on a 1080 Ti
scripts/run_retrieval.sh    # row B: features + blend  ~1 h/seed, mostly CPU (analogy features)
scripts/run_distill.sh      # row C: 10 students + features + blend   ~9 min/seed GPU + ~1 h/seed CPU
```

Every script honours `SEEDS`, `DEVICE`, `DATA`, `CKPT`, `OUT`
(defaults: all ten, `cuda`, `data_ogb`, `checkpoints`, `runs`). Results
land in `runs/` with the same file names as `results/`;
`python scripts/summarize.py runs` prints the statistics.

What each script runs, per seed `s`:

```
# A
python train_ogb.py --device cuda --steps 50000 --block-size 4 --seed $s \
    --save checkpoints/model_final_seed$s.pt --eval both

# B (jaccard once; the rest per seed)
python jaccard_member.py --split valid ; python jaccard_member.py --split test
python cache_scores.py   --split $sp --models checkpoints/model_final_seed$s.pt
python analogy_member.py --split $sp --models checkpoints/model_final_seed$s.pt \
    --out ens_cache/analogy_s$s.$sp.npz --out-top3 ens_cache/analogy_s${s}_t3.$sp.npz
python ensemble_weights.py --members model_final_seed$s analogy_s$s analogy_s${s}_t3 jaccard jaccard_t3 \
    --norm z --seed 0 --min-rows 2000                      # held-out estimate, valid only
python freeze_test.py      --members model_final_seed$s analogy_s$s analogy_s${s}_t3 jaccard jaccard_t3 \
    --norm z --min-rows 4000 --out frozen_single_s$s.npz --result committed_single_s$s.json   # the test read

# C
python train_ogb.py --device cuda --steps 50000 --block-size 4 --seed $s --eval valid \
    --distill checkpoints/model_final_seed{0..9}.pt --distill-w 1.0 --distill-T 1.0 \
    --save checkpoints/dist27_s$s.pt
# then B's steps with members dist27_s$s analogy_dist27_s$s analogy_dist27_s${s}_t3 jaccard jaccard_t3
```

Training is seeded (`--seed` seeds torch and numpy); CUDA kernels are
not bit-deterministic, so a retrained checkpoint reproduces the MRR to
seed-level noise (≈ 0.001), not bit-for-bit. The blend stage is exact.

## Protocol

- Every configuration's expected test range was written down before
  its first test read, and each row's ten runs are the first and only
  test reads of those ten runs. All comparisons between recipes,
  features, blend rules and distillation settings were made on the
  validation split.
- The test split is read once per run: row A by `train_ogb.py --eval
  both` at the end of training, rows B and C by `freeze_test.py`. Test
  score caches are written without computing any per-member test
  metric (`cache_scores.py`).
- Test reads made during development, outside the three rows, each a
  single shot of a configuration chosen on validation and none
  re-run: 25k-step single model 0.8024; 50k with 2×2 blocks, seeds 0
  and 1, 0.8084 / 0.8080; the final recipe's seeds 0 and 1 trained on
  the 1080 Ti 0.8134 / 0.8123 (their 5090 retrains are row A seeds 0
  and 1, 0.8135 / 0.8130); a
  uniform ten-model ensemble 0.8428; seed 0 + three analogy features
  0.8368; a 48M single (`--k 16 --neg 16384 --ent-bias`) + its
  features 0.8472; the same 48M recipe distilled from twelve
  checkpoints + features 0.8535. Nine reads in total.
- The blend weights of rows B and C are selected on the full validation
  split. The held-out estimate of the selection procedure is reported
  alongside and is within 0.001 of test on every run.
- Row C's ten students share the same ten teachers, so its std
  reflects student-seed variance only.
- No external data. Both feature families use only `split["train"]`;
  the Jaccard feature does not use the model at all.

Hardware: rows A–C were trained on rented RTX 5090 (32 GB) instances;
seeds 7–9 of row B's features and all verification ran on a GTX 1080
Ti (11 GB). Analogy features need ≈ 6 GB RAM and 25–40 min per split
per seed on one CPU core.

## Layout

```
resonate.py           model
train_ogb.py          training + official evaluation (rows A, C)
cache_scores.py       model scores on the 501 candidates -> ens_cache/<tag>.<split>.npz
analogy_member.py     analogy / analogy_t3 features
jaccard_member.py     jaccard / jaccard_t3 features (shared by all seeds)
ensemble_weights.py   held-out estimate of the per-relation blend (valid only)
freeze_test.py        freeze weights on valid, one test read
verify.py             split disjointness + checkpoint re-evaluation
scripts/              run_campaign.sh  run_retrieval.sh  run_distill.sh  summarize.py  fetch_checkpoints.sh
results/              the logs and receipts behind the table (seed-0 receipts carry the
                      development-time tags analogy = analogy_s0, pool7_k12_distill = dist27_s0,
                      analogy_dist27 = analogy_dist27_s0)
paper/                resonate.tex, refs.bib, resonate.pdf
```

Release `v1.0-biokg` carries the twenty checkpoints
(`model_final_seed{0..9}.pt`, `dist27_s{0..9}.pt`, 108 MB each; each
stores `{"model", "offset", "n_rel", "args"}`) and their `SHA256SUMS`,
which `scripts/fetch_checkpoints.sh` checks after downloading.
