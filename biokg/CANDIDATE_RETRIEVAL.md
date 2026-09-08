# CS1: candidate-side analogy added to frozen model + retrieval

Registered 2026-09-06 before real feature generation or new result inspection.
User approved only the candidate-side analogy proposal, not path retrieval
or a wider search over the existing five-feature blend.

## Fixed inputs and data boundary

One own H35F single A, k=12, 27,124,129 real parameters, checkpoint
`biokg/results/h35f/campaign_s0/single.pt`, SHA256
`df7c783e7b8a6110f126bca1c60d89209735accbaa5ffc0b362f2cab52d9d2f1`.
All parameters frozen, gradients absent, LR=0, no optimizer or teacher.
No external trained artifact. Keep the other GPU workload untouched;
this run uses CPU with four PyTorch threads.

Directly open only TRAIN, VALID and raw node counts through a file-open
guard. Never use the all-split dataset constructor, processed graph, TEST
or legacy prediction caches. Verify the audited FC1/BC1 cache provenance,
source/input hashes and unchanged model tensors. VALID edges are queries
and evaluation labels, never retrieval graph input.

## Exact new feature

For directed query (s,r,?), let N_r(s) be the unique TRAIN targets of s.
For every candidate c, compare its frozen normalized complex entity
embedding with embeddings of `N_r(s) excluding c`. Excluding c applies to
every candidate, so a direct TRAIN self-match cannot supply a trivial score.
Neither positive-column position nor candidate order affects the feature.

Let similarities be the real Hermitian inner products (the same embedding
cosine definition as FC1). Feature 1 is the signed cube of their maximum;
feature 2 is the signed cube of the mean of their largest min(3,count)
values. Empty support has sentinel -1 in both features. Deduplicate TRAIN
edges; do not pad an existing one/two-neighbor mean with zeros. Candidate
duplicates have identical features. For head queries swap TRAIN source and
target roles. Do not infer or inject extra inverse edges into TRAIN.

This is the opposite-end analogue of current holder retrieval, evaluated
for each candidate in the same query. It is not just averaging the head
and tail evaluation metrics. Source-side analogy and Jaccard stay unchanged.

Normalize each new feature across the supplied candidates with FC1's fp32
z-score (std + 1e-6). Fix C = (Z_max + Z_top3)/2, with no renormalization.
If a whole query has no TRAIN support, both normalized features are zero;
mixing them into the baseline only positively rescales its scores.

Compute exact pairwise embedding cosines within one target entity type at
a time, in CPU matrix chunks, to reuse them across relations and repeated
query sources. The largest temporary fp32 matrix is ~8.13 GB for function
entities. Release it before the next type. It is an in-memory acceleration
cache, not an extra model or new graph. Chunk candidate/neighbor pooling
to bound intermediate memory. FP32/complex64 throughout; no AMP, TF32 or ANN.

## Fixed comparison and finite selection

Use exactly BC1's three partition seeds (0,1,2), both fold directions each,
all 162,886 VALID triples, both prediction directions and all 500 negatives.
For each fold, B is the complete score of BC1's saved half-strength recipe
selected on that fold's fit half. Reuse that exact recipe; do not reselect
or rearrange its five-feature arithmetic. Report ranks must match BC1 exactly.
Thus the matched aggregate baseline is 0.8555176434837662, not the old
0.854576 baseline and not an individual report-half result.

Treatment score: `(1-beta)*B + beta*C`, with beta selected only from the
fixed list {0,0.025,0.05,0.1,0.2}. Zero reproduces B. Select global beta,
then family/direction and relation/direction only with >=2,000 fit queries;
otherwise inherit family/global. Ties prefer the smallest beta. Do not add
other strengths, change max/top3 weighting or tune old feature weights.
Only fit-half labels select beta; save the recipe before report scoring.
No gradients or continuous optimization on VALID.

One primary contrast: treatment minus B. Each triple is reported once per
partition with its two directions together. Average reciprocal-rank deltas
over directions and partitions within each triple, then 2,000-replicate
paired bootstrap, seed 3641, descriptive 95% interval. Never count repeated
partitions as independent samples or average their scores into an ensemble.
Useful-gain flag: mean delta >= +0.001, lower bound >0, all three partition
deltas >0. Also report consistent-positive separately. No automatic promotion.

Descriptive only: six fold metrics, three partition metrics, direction
metrics, top-one recoveries/losses, selected beta counts, and full-VALID
standalone ranks of each new feature and C. No report-driven switches,
interventions, early stops or new arms. Reused VALID is not an untouched
holdout, model-seed robustness check or leaderboard/TEST measurement.

## Verification and stopping

Before execution: synthetic brute-force pooling and opposite-end holder
equivalence, genuinely different source/candidate-side example, missing and
self-only support, duplicate edges/candidates, candidate permutation and
pool-extension invariance, chunk invariance, signed cube/top3 semantics,
frozen parameters, list metadata and head/tail offsets, beta-zero exact
parity, fit/report poisoning isolation, sparse fallback, paired coverage,
official ties and repeated-triple statistics.

For every directed relation, audit representative queries against direct
embedding-dot brute force (atol=rtol=1e-5 for matrix-kernel rounding) and
require exact candidate permutation/chunk invariance on cached cosines.
Record source-neighbor counts and examples. Verify all baseline report
ranks exactly. Persist raw/new normalized features, pair scores, recipes,
report ranks, hashes, progress and summary. Independent endpoint audit
rebuilds representative TRAIN neighbor lists and direct features, reproduces
all normalization, six fit-only beta recipes, report ranks, metrics and
intervals, and checks immutable model/input/source hashes.

Progress at least every 35 seconds: stage/query count, elapsed time, LR=0,
fixed baseline MRR; candidate pipeline MRR remains pending until each fixed
report stage. Finish all six folds and the audit. No path features, new
model training, other GPU use, full-VALID refit, TEST or submission change.
