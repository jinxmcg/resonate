# RF1: frozen-A rarity weighting and support-aware retrieval

Registered 2026-09-06 before new feature computation or result inspection.
User approved two separate post-training tests. Generate their independent
TRAIN-derived statistics together to share graph traversal, then evaluate
rarity first and support second. Do not combine the interventions.

## Fixed baseline and boundaries

Use only our H35F improved A, k=12, 4x4 blocks, 27,124,129 real parameters:
`biokg/results/h35f/campaign_s0/single.pt`, SHA256
`df7c783e7b8a6110f126bca1c60d89209735accbaa5ffc0b362f2cab52d9d2f1`.
Keep every model parameter frozen, with no gradients, optimizer or teacher.
No original-A or CFKD1 checkpoint, external trained artifact, B or inference
ensemble. Additional feature formulae are deterministic, not learned models.

Baseline is the audited FC1 improved pipeline, report-half MRR
0.8540538579928584. Read its verified fp32 VALID feature caches, metadata,
recipes and ranks. Use exactly the same full VALID and fit/report mask
(`default_rng(0)`, whole triples paired across directions). This is adaptive
development validation, already exposed to prior experimentation; it is not
a pristine holdout, seed-robust result or estimate of final TEST rank.

Directly load TRAIN, VALID and raw node counts only, through the existing
file-open guard. Build relation/direction adjacency from unique TRAIN edges
only; exclude query source from candidate holders. VALID supplies queries
and evaluation labels, never graph edges. No OGB dataset constructor,
processed graph, legacy TEST caches, TEST evaluation or full-VALID refit.
Every provided candidate is scored by the same rule; no answer-position,
candidate-order or negative-generation shortcut. Candidate duplicates
receive identical features.

## Test 1: rarity-weighted Jaccard

For each directed TRAIN relation, let N be the number of sources with at
least one TRAIN edge and df(v) the number of unique TRAIN sources linked
to target v. Fix `w(v) = 1 + log((N + 1) / (df(v) + 1))` (natural logarithm).
Neither N nor df includes validation edges or additional empty query rows.

Replace Jaccard between source neighbourhoods S and U with
`sum(w[v] for v in S intersection U) / sum(w[v] for v in S union U)`.
Zero union has similarity zero. Keep candidate holder max/top-three mean,
source exclusion, no-neighbour/missing-holder sentinel -1, and fp32
per-candidate-row z-normalization exactly as FC1. Replace only the Jaccard
pair; direct A scores and ordinary embedding analogy stay unchanged.
Uniform target weights recover ordinary Jaccard. No exponent/IDF grid.

## Test 2: support-aware use of ordinary retrieval

Do NOT use rarity-weighted Jaccard in this arm. Use FC1's original analogy
and ordinary Jaccard scores, plus deterministic candidate-specific support.

For each query and metric separately (embedding cosine before cubing;
ordinary Jaccard), compute mean mu and population standard deviation sigma
of similarity to all active TRAIN sources of that directed relation,
excluding the query source. These are query-conditioned TRAIN-graph/model
statistics, independent of the supplied candidate list and its labels.
Moments use float64 reductions of fp32 similarities, then cast to fp32.
For fewer than two background sources or sigma <= 1e-8, set evidence zero.

For each holder's similarity s, define bounded positive evidence
`e = max(s - mu, 0) / (max(s - mu, 0) + sigma + 1e-8)`.
For candidate c, exclude the query source, take its largest three e values,
pad missing values with zero, and fix `g(c) = sum(top3 e) / 3`.
Thus missing/all-weak support gives zero, one strong holder gives at most
one third, and several above-background holders give more support. No
uncapped holder count/popularity sum. Both members of a metric pair share
its gate. Jaccard has zero gate if the query has no TRAIN neighbours.

After the existing FC1 z-normalization, replace each retrieval member R by
`R_supported(c) = g(c)*R(c) + (1-g(c))*A(c)`, where A is normalized model
score. Do not normalize again after this interpolation. Unsupported
retrieval weight therefore returns to A rather than treating missing
evidence as a negative answer label. Gates zero/one recover A/original R.
The gate has no fitted threshold, confidence bin or relation-specific
learned parameters; it is not a repeat of the old failed three-margin-bin
gate. This is one fixed candidate-support hypothesis, not all possible gates.

Include one descriptive ablation: replace every gate by fixed 0.5, otherwise
identical. It asks whether simply weakening retrieval helps. It is not
matched to the observed gate mean and cannot isolate every gating mechanism.

## Selection and decisions

Each arm and the half-strength control gets the identical FC1 finite
global/family/directed-relation mixture family, with min 2,000 fit queries
per group, same fallbacks and tie-breaking. Existing fit_recipe is reused
unchanged. Only fit-half labels select weights. Freeze recipe before report
scoring. No gradients on VALID; no report-dependent settings or stopping.
Reapplying the unchanged baseline recipe must exactly reproduce its ranks.

Two primary contrasts: rarity minus unchanged baseline; support minus
unchanged baseline. Whole-triple paired bootstrap, 2,000 replicates,
seeds 3621/3622. Report per-contrast 97.5% intervals (quantiles .0125/.9875),
a Bonferroni adjustment for the two planned primary comparisons. Advance
flag requires delta >= +0.001 and adjusted lower bound > 0. No automatic
promotion, combination, new seed, training or TEST run, irrespective of flag.

Descriptive: fixed-original-recipe variants, fixed uniform mixtures,
half-strength control and support-minus-half-strength, per-feature ranks,
family/direction metrics, and top-one recoveries/losses. These do not drive
new switches on the report half. Descriptive intervals use 95%, seeds
3623/3624. Report fit and report numbers separately.

## Verification, compute and progress

Before real computation: synthetic weighted-intersection brute force and
uniform-weight equivalence, rarity ordering, duplicated-edge invariance,
background population independence from empty query rows, support formula
brute force, zero/one gates, missing/self holders, all-equal/empty background,
candidate permutation and duplicates, list-type metadata, fp32 finiteness,
fit/report isolation and official average-tie ranks.

Verify FC1 source/input/artifact hashes before use. Save protocol/source/input
hashes before the run. Recompute unchanged features on at least one batch
of every directed relation and compare with FC1; verify candidate permutation
there too. Keep model tensor hashes unchanged, gradients absent. Save graph
weights/background audits, generated feature/gate arrays, recipes, ranks,
summary and independent endpoint-audit output. Reproduce all saved metrics,
normalizations, support transforms, recipes and intervals independently.

The 1080 Ti computes only frozen embedding similarities; graph pooling and
mixture selection use CPU. Progress at least every 35 seconds with query
count, elapsed time, LR=0 (not applicable), and the fixed baseline MRR.
New pipeline MRR is pending until its predefined evaluation stage. Reuse
audited FC1 scores to avoid unnecessary recomputation of the same model.

TRAIN-only graph statistics and finite validation hyperparameter selection
follow [OGB's data-use protocol](https://ogb.stanford.edu/docs/leader_rules/).
This does not erase historical TEST exposure or independently certify the
eligibility of a future submission.
