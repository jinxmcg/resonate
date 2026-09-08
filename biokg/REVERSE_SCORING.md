# RS1: single-checkpoint reverse scoring, drug–drug only

Registered 2026-09-07 before new scoring or outcome inspection. User approved
bidirectional candidate scoring and requested fail-fast behavior. Scope this
first test to drug–drug; every other family stays exactly unchanged. No
retraining, new graph feature, broader mixture search or automatic promotion.

## Frozen inputs and rules

Use only our H35F single A, k=12, 27,124,129 real parameters, checkpoint
`biokg/results/h35f/campaign_s0/single.pt`, SHA256
`df7c783e7b8a6110f126bca1c60d89209735accbaa5ffc0b362f2cab52d9d2f1`.
Its entity table, 102 existing directed relation operators and temperature
remain frozen; LR=0, gradients absent, no optimizer or extra model parameters.
Use CPU/four Torch threads, leaving GPU workloads untouched.

Reuse audited FC1 features, BC1 half-strength recipes and CS1 candidate-side
features/recipes. The exact current complete pipeline P has aggregate MRR
0.8582166512116752. Its seed0/fold0 report baseline is 0.8578497365986548;
do not compare a report-half score directly to the aggregate baseline.
Keep all old recipes and fp32 arithmetic fixed, including CS1's old beta.

Directly deserialize only our checkpoint and official VALID; raw node
counts establish offsets. TRAIN bytes may be opened solely to reverify
inherited provenance hashes, never deserialized or used to build new edges.
Reject all other dataset files (especially TEST, processed/full raw graphs)
and unapproved binary checkpoints/feature caches via a file-open guard.
No external trained artifacts, validation graph edges, validation gradients
or continuous learned combiner fitting. Ordinary finite validation selection
is disclosed. Do not infer/add reverse validation edges to any graph.

## New score

For directed query (s,r,?) and every candidate c, compute the existing model's
score of (c, inverse(r), s). With n_base=51, inverse swaps r and r+51.
Specifically, q(c,inverse(r)) = cnorm(H_inverse(r) cnorm(E[c])) and
R(s,r,c) = exp(log_tau) * Re(sum(q(c,inverse(r))*conj(E[s]))).
Use the trained reverse operator, not the forward operator's transpose or
adjoint. Keep the raw E[s] target norm exactly as in the existing scorer.
No squaring, cubing, softmax over reverse candidates or forced symmetry.

Cache q(c,inverse(r)) for drug entities per directed relation, using the
same frozen model query routine. This is an exact readout acceleration,
not another model. Chunk candidate dots in complex64/fp32; verify against
direct calls to the model with candidates as sources and s as the target.
Treat all 501 columns identically, including duplicates, true positives
and negatives. Do not restrict to the known-wrong or known-top-ten rows.

Normalize R across the supplied 501 candidates with the existing fp32
z-score (std+1e-6). For drug–drug only, new score = (1-alpha)*P + alpha*Z_R.
Else score=P exactly. Fixed alpha choices {0, .025, .05, .1, .2}; zero
recovers the control. Reuse the tested finite selector: global over fit
drug–drug rows, then family/direction and relation/direction if >=2,000 fit
queries, otherwise inherit. Ties prefer alpha=0/the smallest alpha.
The shared selector's saved `beta` field denotes RS1 alpha, not CS1 beta.
No old-weight refit, max/top3 change, new strength or other new scoring arm.

## Fail-fast screen and conditional confirmation

Use CS1's fixed triple-paired splits: partition seeds0/1/2 and both fold
directions. Both head/tail rows of a triple stay together; all 500 official
typed negatives retained. Save each new alpha recipe before report scoring.
Baseline report ranks must exactly reproduce the saved CS1 rank vector.

First evaluate only seed0/fold0 (81,199 report triples). Primary screen
contrast is new complete pipeline minus P on that entire report half, with
unchanged non-drug–drug rows included. 2,000-replicate paired bootstrap over
original report triples (average directions within triple), seed 3661.
Proceed to the remaining five folds ONLY if overall delta >=+0.0005 and
the descriptive 95% lower bound >0. Otherwise stop and audit the one-fold
screen; do not compute remaining fold recipes or report metrics. Rejection
is a resource-prioritization decision, not a proof of zero possible benefit.

If the screen passes, finish all five remaining folds with the same method.
Report the three two-way partitions and their mean; average direction and
partition repeats within original triple before bootstrapping, 2,000 draws,
seed 3671. Useful-gain flag: mean delta >=+.001, lower bound >0 and all three
partition deltas positive. Neither flag automatically promotes a submission.
Intervals are descriptive/conditional on the fixed model and adaptive VALID
reuse, including the screening decision; not fresh-holdout or model-seed
robustness evidence. Partition repeats are not independent training runs
or an inference ensemble.

Per evaluated fold, report exact baseline/treatment metrics overall and
drug–drug, alpha counts, direction metrics, top-one recoveries/losses and
recoveries from baseline ranks >1 through 10. Reverse-only drug–drug report
MRR is descriptive and computed only after alpha selection; it is not a
gate or a complete model+retrieval score. No post-hoc subgroup promotion.

## Verification and stopping

Before execution test inverse-index involution/range, direct reversed-model
parity, raw target norm, candidate permutation/duplicates/pool extension,
query/cache chunks, frozen tensors, type offsets/list metadata, exact nested
CS1 baseline reconstruction, zero-alpha/no-change, non-target isolation,
fit/report poisoning isolation, sparse fallback, paired coverage, average
ties, screen gate and clustered repeated-partition statistics.

For each of 76 drug–drug directed relations, compare a representative query
over all 501 candidates to direct reversed model calls (atol=rtol=1e-5),
require exact cached-dot candidate permutation/chunk invariance, and record
max error. Save new raw/normalized scores, query/candidate IDs, per-fold
recipes/ranks, coverage, screen/summary and source/input/artifact hashes.
Unused fold rank slots stay explicitly NaN with zero coverage after a failed
screen; never fabricate an aggregate score for an incomplete partition.

Separate endpoint replay rebuilds VALID indexing, directly recomputes those
representative reversed scores, verifies all normalization, exact existing
baseline ranks, fit-only alpha recipes, new ranks, screen decision, summary
statistics, non-target invariance and immutable model/source/input hashes.
Progress at least every minute: stage, query count, LR=0, matched baseline
and any fixed-stage report MRR. Stop after the completed stage(s) and audit.
No TEST, full-VALID refit, additional experiment or submission update.
