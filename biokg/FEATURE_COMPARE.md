# FC1: original versus improved A with matched retrieval features

Registered 2026-09-06 before new real-data feature generation or selection.
User approved comparison. Local 1080 Ti, no model training or new checkpoints.

## Fixed inputs and data boundary

- Original own `biokg/checkpoints/dist_T2_s0.pt`, SHA256
  `b463a8bcc431ada38f4834e235677464f9cd3a86346f88aecd2fd681a01260ef`.
- Improved own `biokg/results/h35f/campaign_s0/single.pt`, SHA256
  `df7c783e7b8a6110f126bca1c60d89209735accbaa5ffc0b362f2cab52d9d2f1`.
- Both k=12, 144 complex coordinates, free 4x4 blocks, 27,124,129 real
  parameters. No CFKD1 endpoints or external trained artifacts. No teachers.
- Directly open TRAIN, VALID and raw node counts only. Reuse recorded
  relation-family metadata; do not invoke an all-split dataset constructor
  or read legacy score caches, TEST, processed graphs or validation edges
  as graph input. File-open guard rejects other files under the dataset root.
- Fixed full VALID, 162,886 triples, both directions, all 500 official
  negatives. Fit/report mask: `default_rng(0).random(N) < .5`, as in the
  historical held-half feature script. Both directions stay together.
  Reused development validation is not an untouched holdout.

## Matched pipeline

Each endpoint has exactly five inputs: its direct score, its embedding
analogy max and top-three mean (signed cube), shared TRAIN Jaccard max and
top-three mean. Exclude query source from holders; unique TRAIN edges;
missing holder sentinel -1, including Jaccard for sources with no TRAIN
neighbours. Similarities use normalized complex entity embeddings. No
relation-aware analogy variant, new graph feature or negative filtering.

FP32 generation, per-row z-score (`std + 1e-6`) and mixture accumulation;
complex64 model, no AMP or TF32. All candidate columns receive the same
scorer. Use official average-tie ranks for selection and reporting.

Same finite historical-style selection for each endpoint, independently:

- Global: uniform all five; uniform top three members by fit MRR; softmax
  of per-member fit MRR with eta in {20,50,100}.
- Family/direction and relation/direction: only if at least 2,000 fit queries.
  Candidates: uniform, global, applicable family fallback, top one or three,
  and the same three eta values. Otherwise use family, then global fallback.
- Top five/nine/thirteen of five members are the same uniform recipe; remove
  those positive-rescaling duplicates. Fix feature-order tie breaking and
  retain first recipe on equal fit score. All weights nonnegative, sum one.
- Selection sees fit queries only. Freeze recipes before report evaluation.
  No gradients, iterative continuous weight fitting, report-driven retuning
  or full-VALID refit. This matches the feature definitions and selection
  hierarchy, not historical fp16 caches/pessimistic-tie metric byte for byte.
  Deduplicating the one known repeated TRAIN edge follows the audited H34
  set semantics. These declared numerical corrections apply to both arms.

## Comparisons and interpretation

Primary: improved five-feature pipeline minus original five-feature pipeline
on the report half, each with its own fit-only chosen recipe. Whole-triple
paired bootstrap 1,000 replicates, seed 3611, two-sided 95% interval.
Predeclared useful-gain flag: delta >= +0.001 and lower bound > 0. No automatic
promotion, TEST evaluation, seed campaign or submission change.

Descriptive: model-only, individual features, fixed-uniform mixture, and
each pipeline's uplift over its own model. Show fit and report separately;
do not compare a report-half number to historical TEST or tuned full VALID.

Attribution diagnostics freeze original fit-selected weights and substitute
(a) improved model scores only, (b) improved analogy only, (c) both. Jaccard
is unchanged. The crossed (a)/(b) combinations depend on two snapshots and
are diagnostic only, not proposed single-model submissions. No extra tuning
of crossed combinations. Include family/direction breakdowns and top-one
recoveries/losses, without using them to retrofit rules on report labels.

## Verification and logging

Before real execution: synthetic brute-force feature parity, candidate
permutation and duplicates, source exclusion/empty graph, frozen weights,
standalone original-score parity, official ties, finite normalized mixtures,
fit/report isolation and sparse-group fallback. Test that duplicate TRAIN
edges have no effect. Check full model-only ranks against archived VALID;
abort if absolute MRR discrepancy exceeds 1e-6, record any rank differences.

Save protocol/source/input hashes before execution, raw and normalized fp32
features, fit mask, recipes, ranks, summary, progress and endpoint audit.
Verify both model hashes unchanged and no gradients; score and feature
candidate symmetry on real batches; exact metadata alignment. Progress every
35 seconds with completed queries and partial model-only MRR explicitly
labelled, LR 0 / not applicable (no optimizer). No early decisions from
relation-ordered partial scores. Independent saved-record audit reproduces
report metrics/intervals and verifies receipts without loading TEST.

The data boundary follows [OGB's protocol](https://ogb.stanford.edu/docs/leader_rules/):
TRAIN for graph/model input, VALID for standard hyperparameter selection,
TEST reserved for final evaluation. Historical test exposure remains
disclosed; this comparison cannot itself certify submission eligibility.
