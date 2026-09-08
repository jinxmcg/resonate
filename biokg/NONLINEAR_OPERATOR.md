# NL1: change the function, retain the 4x4 storage

2026-09-07. User approved testing a nonlinear function, explicitly not larger
matrices. One fresh single-model matched pair on the authorized Vast RTX 5090
instance 50109054 (79.19.114.194, direct SSH 50034), using uv. No B, ensemble,
distillation, external weights, official VALID/TEST, or submission change.
The completed MN1 internal holdout has already been observed: this is now an
internal development split, not an untouched confirmation set.

## Fixed hypothesis and comparison

For normalized source x, form the existing free complex block product z=H_r x.
Control: q=normalize(z). Nonlinear: apply f(z_i)=z_i/(1+|z_i|) to EACH complex
coordinate, then q=normalize(f(z)). This is before global hop normalization,
not after it. Complex phase is retained; magnitudes saturate. Alpha is fixed
at 1, not learned or selected. Alpha=0 is an implementation parity test only.
Apply in every relation/direction at training AND inference. Candidates use
the original raw entity vectors and real inner product, same for all columns.

Same k12, 36 free 4x4 complex blocks per directed relation, 144 complex entity
coordinates, 102 directed relations, 27,124,129 real parameters. No additional
trainable parameters or width. Operators start unitary via QR but remain free,
so they may learn to reduce saturation by changing their scale. Record fit-only
preactivation magnitude, attenuation and query-angle diagnostics; do not tune
alpha in response. This tests one nonlinear function, not all nonlinear KGE.

Train both FROM SCRATCH for 50,000 updates each. This avoids conflating a
function change with perturbing a completed linear model. Same random seed 0
and exactly copied initial state with separate tensor storage; fresh identical
optimizer states. Same actual batch tuple in both arms on each step. Original
MN1 base stream seed 0, batch 2048, 4096 shared typed uniform negatives, natural
relation-frequency sampling and uniform head/tail prediction direction.
All entity embeddings and operators co-adapt. Original unfiltered CE + 0.1
query-to-positive trajectory loss; no graph context. Adam H/tau LR .005,
RowAdagrad table LR .3, common cosine to zero over 50k, global clip 1.
fp32/complex64, no AMP/TF32, 4 CPU threads. Equal updates, not equal runtime.

Only TH1 derived fit_train.npz, holdout_train.npz and manifest are deployed.
Fit hash aa13b00c4f4e5f93f18a2af02e895f85375fedc873d09a25778090e97913796f;
holdout hash 783a55949f188b1ac09194df3bf03ed5a371284e228a17b7ca8282003b4e7e2f.
Training opens fit and metadata only, with the MN1 audited file guard denying
held-out/foreign model files. No previous model checkpoint is a training input.
Legacy and MN1 sources/protocols remain unchanged; reuse their data, loss,
optimizer and frozen-scoring helpers explicitly, pin every imported source.

## Progress, endpoints and fail-fast decision

Log steps, both actual optimizer LRs, losses and elapsed time at least every
35 seconds. Every 5000 steps and endpoint, report MRR on the same 1024-row
in-sample fit probe as MN1 (seed 36852), with private RNG and no gradient.
Probe scores are not generalization estimates and select no checkpoint.
Save only the fixed 50k endpoints plus the common initial state receipt.

Only after BOTH endpoints are frozen may a separate process evaluate TH1's
236285 held-out rows in both directions. Reuse MN1's exact fixed candidate
policy and evaluator: 500 typed uniform negatives with replacement, excluding
source, designated positive and known fit answers, no filtering via other
holdout labels. Seed 36853 per relation/direction, chunk 256, average ties.
All 472570 queries, cold endpoints and original duplicates remain included.
Rank arrays and candidate stream hash are saved. The controls can be compared
with MN1's frozen base descriptively; neither its checkpoint nor its scores
enters this training process or changes this recipe.

Primary delta: nonlinear minus linear MRR over all queries. Pair-cluster 95%
bootstrap, 2000 replicates seed 36854, grouping all rows/directions/relations
on each unordered entity pair. Report Hits@1/10, direction, relation, drug-drug
and cold-endpoint slices. No subgroup promotion. No training-seed uncertainty
is measured by this interval; no official leaderboard projection.

Advance only if delta >= .001 AND the lower 95% bound > 0. A pass supports
proposing seed confirmation, not automatic promotion. A fail stops this exact
version; no alpha, loss or learning-rate sweep. Stop earlier on synthetic
failure, nonfinite state, forbidden data access, changed receipts, unsupported
hardware or smoke throughput worse than 1 second per paired step. There is no
interim held-out evaluation and no unsupported chance-of-success extrapolation.

## Verification and execution

Synthetic tests: alpha-zero original output/loss/gradient parity; explicit
real-arithmetic complex multiplication; phase retention and finite zero/large
inputs; complex128 gradient check; demonstrable nonlinearity/gradient change;
no extra parameters; independent initial storage; finite sparse updates;
checkpoint alpha/config preservation and rejection of mismatched metadata;
candidate permutation/duplicate symmetry; detached, RNG-isolated evaluation;
inherited split, candidate, guard and loss tests. Full-size synthetic GPU smoke
before real updates. Pin source/protocol/data receipts before training.

Finite supervisor job biokg_nl1, autorestart disabled, isolated directory
/workspace/biokg_nl1. uv Python 3.12, torch 2.11.0+cu128, numpy 2.5.2,
scipy 1.18.1; reuse the private MN1 uv download cache without editing its files
manually or changing its experiment. No other GPU job or system service touched.
After train -> evaluate -> audit, the job exits; do not stop/destroy the rental.
Back up both checkpoints and receipts locally, verify hashes, release GPU.

Endpoint auditor replays all candidate streams/scores/ranks and bootstrap;
verifies model/source/artifact hashes, parameter counts, checkpoint activation,
optimizer counters/accumulators, LR schedule, training file-access receipt and
split separation. It is not a replay of all optimizer steps. Preserve all
outcomes and existing experiments.
