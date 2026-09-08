# BioKG optimization queue

Updated: 2026-09-07. Goal: a reproducible, rules-compliant **single inference
model**, optionally distilled and/or augmented with TRAIN-graph features.
Work through these hypotheses sequentially; do not launch a broad sweep or
combine unproven changes. A clean top-three / second-place result is a useful
milestone, not a guaranteed outcome.

## Where we stand, and my best guess

The existing distilled student is about 0.8321 validation MRR alone. Its
feature-augmented pipeline reports 0.8532 on a held-out validation half and
0.8537 on full validation after hyperparameter selection (historical README
results, not a fresh audit). Neither number is directly interchangeable with
a fresh full-validation measurement. The published historical test result is
0.8528; do not reopen its test predictions for optimization.

The [official BioKG leaderboard](https://ogb.stanford.edu/docs/leader_linkprop/#ogbl-biokg),
checked on 2026-09-05, lists RelEns first at 0.9618 test MRR, ComplEx² second at
0.8583 (0.8592 validation), and UniBi third at 0.8550 test. Our reported test
gap to third is small, but that is context, **not a test-tuning objective**.
The previously discussed **0.9473 is an optimistic top-10-reranking bound for
the plain baseline**, not an achieved result or an expected improvement.

My leading diagnosis is a representation / ranking-information limitation,
not merely insufficient training or GPU speed:

1. **Retrieval uses a relation-blind similarity.** The current analogy feature
   compares entity embeddings with the same cosine for every relation. Yet
   being similar for drug interactions need not mean being similar for side
   effects. The trained relation operators offer a cheap alternative metric.
2. **The scorer uses one query vector.** It may struggle to rank several
   different modes of valid answers. A low-rank correction to that vector is
   still a linear readout against candidate embeddings; it does not resolve
   this expressivity issue by itself.
3. **Distillation may preserve shared blind spots.** The current teachers are
   variants of the same architecture. Their average may reduce noise without
   teaching the student genuinely new structural evidence.
4. **Most measured deficit is concentrated, not diffuse.** On the plain
   baseline, drug–drug, protein–function and drug–side-effect account for
   roughly 81% of the MRR shortfall. This must be remeasured on the strongest
   student + features before claiming it explains that pipeline's residuals.

H35 now verifies the same concentration on the frozen distilled **student
alone**: drug–drug 35.76%, drug–side-effect 23.47%, protein–function 22.37% of
its MRR deficit (81.60% combined). Its full-validation MRR is 0.8321749. The
FC1 below now measures the feature-augmented pipeline and its report-half
family/direction results; those historical deficit shares describe A alone.

These are hypotheses, not established causes. `1 - MRR` is not the fraction
of incorrect queries. More capacity could help, but earlier width sweeps,
generic two-hop features and simple feature recombination did not justify
prioritizing another blind sweep.

## Non-negotiable protocol

- User clarification, 2026-09-06: learn from other methods' public code and
  equations, with attribution and license compliance, but do not download or
  use their trained checkpoints, embeddings, prediction caches or learned
  combination weights. Any future teacher must be our own documented
  TRAIN-only training. The previously downloaded ComplEx-RP archive was
  deleted without extracting/loading its weights or scoring it.
- Follow the [standard OGB rules](https://ogb.stanford.edu/docs/leader_rules/):
  model fitting, learned metrics, teachers and structural graphs use TRAIN
  edges only. No gradients on validation, no validation edges as graph input.
- Validation permits ordinary, disclosed, finite hyperparameter selection.
  Split tuning/evaluation by whole triple: its head and tail queries stay
  together. These are development estimates, not a pristine final holdout
  after repeated adaptive experimentation.
- Do not load TEST edges, candidates, score caches or metrics in these trials.
  Freeze the entire final recipe before any authorized final test evaluation.
  Previous test exposure must remain disclosed; a fresh runner does not erase
  history or certify submission eligibility.
- Score every provided candidate symmetrically, both directions, with official
  OGB average-tie ranks and fp32 scores. Do not exploit positive column position,
  candidate ordering, or artifacts of negative generation.
- Keep one checkpoint at inference. Any teachers are training-only. Disclose
  all training stages, feature provenance, parameter accounting, tuning and
  compute; do not assume teacher parameters can silently be omitted.
- Record failures as well as successes. Confirm promising changes across
  seeds, then follow the [submission reporting requirements](https://ogb.stanford.edu/docs/leader_overview/).

## Code-only investigation after H35F

[Detailed findings](biokg/results/code_review/README.md). No new BioKG
evaluation, model training, external-weight use or submission change.

The clearest new finding is that, at equal embedding width, A's free 4x4
blocks can contain the diagonal ComplEx scorer and reproduce its candidate
ordering. Query normalization changes training logits and gradients, but is
a positive candidate-independent scale for a fixed entity/relation query.
A synthetic CPU check using our own random tensors verified identical
orders on 498 queries x 83 candidates. This does not show that our 144
complex coordinates can reproduce a 1,000-coordinate trained model.

Consequently, distinguish missing representational capacity from a training
objective that learns different representations. ComplEx-RP's released code
uses full-entity CE, relation CE and N3; our retained recipe uses sampled
typed CE, own-teacher KD and a query-to-positive embedding-distance term.
The latter is an explicit attraction objective, not merely norm control.
The resulting TRAIN-only gradient-conflict diagnostic is now complete (see
below); it weakens a large, pervasive trajectory-conflict explanation at
the current checkpoints. It does not measure earlier training dynamics.

Do not revive previous negative experiments as new ideas: relation CE,
N3, larger negative pools, additive entity bias and relation gains already
have recorded trials. Their unsuccessful transfer does not prove that the
underlying objectives are mathematically equivalent to our recipe.

The old A + existing features versus improved A + the same feature protocol
comparison is now completed as FC1 below. The subsequently user-approved, separately
controlled A-only training comparison CFKD1 is now complete below. No
external checkpoints, new teacher training or new architecture campaign
is authorized by this code review.

## TRAIN-only gradient diagnostic: completed

[Results and interpretation](biokg/results/gradient_diagnostic/README.md).
User-approved local 1080 Ti, original and improved A on 512 matched TRAIN
batches, no optimizer, VALID/TEST access or model changes. All 12 own model
states unchanged. 17 synthetic tests and endpoint checks passed; gradient
loop 57.19 seconds, peak allocated memory 2.10 GiB.

For improved A, weighted trajectory gradients have only 2.99% of the CE+KD
norm on entities/operators together, and mean cosine +0.09265. Operators
alone have mild opposition (cosine -0.04139; mean CE+KD descent projection
0.99254), with some stronger rare-relation batches. This does not establish
that removing trajectory would improve MRR; keep the loss unchanged.

The more conspicuous interaction is score temperature: CE and KD oppose
its scalar gradient in 89.45% of improved-A batches. The median share of KD's
squared Euclidean gradient norm there is 92.36%; KD's representation norm
averages 5.57% of CE's. These are parameterization-dependent snapshot
gradients, not Adam-step fractions or proof of ineffective distillation.
The subsequent confidence-versus-ordering investigation is complete below;
it does not support prioritizing a temperature-only change. Do not use
external trained artifacts or VALID fitting in this TRAIN-only diagnostic.
The separate finite-selection feature comparison was later completed as FC1.

## KD confidence versus ordering: completed

[Results and proposed isolated comparison](biokg/results/kd_calibration_diagnostic/README.md).
Local 1080 Ti, both own A snapshots and the same own teachers, 65,536 TRAIN
queries with a canonical-triple fit/report split. No model changes or
VALID/TEST access. All 12 model states unchanged; 25 tests passed.

For improved A, a global confidence scale fitted only on fit-TRAIN is
0.993944 and reduces report-TRAIN KD by just **0.4347%**. Even an optimistic
best scale fitted separately on each report query removes only **2.8654%**.
That second result is not a deployable policy or a generalization estimate.
Large temperature gradients therefore did not establish a large calibration
problem. Keep the temperature/loss unchanged; do not launch a temperature-only
training run on the strength of the previous gradient statistic.

High score correlation (0.95296) coexists with only 47.48% top-score overlap
on these large unfiltered TRAIN pools. This is not an error rate: most
disagreements occur where neither model's top answer is the sampled positive,
and other TRAIN positives may be among the candidates. Remaining KL is not
pure ranking error. The resulting candidate-focused KD proposal retained
the existing full KD, CE and trajectory. It was subsequently approved,
preregistered and tested with a matched unchanged-loss control as CFKD1 below,
without reviving H35D's failed rank-fusion targets.

## CFKD1: candidate-focused A-only training completed, not promoted

User subsequently approved the isolated test and confirmed k=12. The
[fixed protocol](biokg/CFKD1.md) and [results](biokg/results/candidate_focus/README.md)
record a 5,000-update matched unchanged-loss versus added conditional-KD
comparison, both starting from H35F improved A. Same own teachers, TRAIN
stream and 1e-4 cosine schedule; added weight 0.25, top-32 union with cutoff
ties and known TRAIN positive. No new capacity or branch. 34 tests and a
full-size 20-pair local GPU smoke passed before launch. TRAIN gradients,
read-only VALID, no TEST or external trained artifact.

Full-VALID MRR: starting A **0.8340991235**, unchanged-loss control
**0.8338693602**, focused **0.8339914437**. Primary focused-control delta
**+0.0001220835**, 95% whole-triple paired-bootstrap interval
**[+0.0000361302, +0.0002063263]**. This misses the preregistered +0.001
minimum and does not beat starting A, so no seed confirmation or promotion.
Both initial-comparison intervals include zero: the numerical decreases
are not established regressions. Small-probe improvements did not transfer
to full VALID. One adaptive development seed is not a general rejection of
candidate-focused distillation or evidence about increasing capacity.

Endpoint audit passed: hashes, matched initialization/updates, all A groups
trained, teachers frozen, LR/loss schedule, standalone candidate-symmetric
scoring and exact saved-metric/interval reproduction. Recorded compute
496.62 s (~8.28 min), peak allocation 2.18 GiB; 1080 Ti compute is free.
Keep H35F improved A. Both arms already used k=12, not k=8; no width increase
was tested. The resulting next comparison, old A versus H35F improved A
under the same feature protocol, is now completed as FC1 below. Do not infer its gain by adding model-only
MRR changes to the historical feature-pipeline test score. No next run or
submission change was launched; second place remains an unachieved goal.

## FC1: matched feature comparison completed, small positive gain

User approved measuring the pending feature-pipeline transfer. Both frozen
k=12 checkpoints are compared on full VALID with their own analogy features
and shared TRAIN Jaccard. Identical whole-triple fit/report split and finite
global/family/directed-relation weight selection; no gradient-based fitting.
30 synthetic tests passed, including a real-loader list-type regression
after the first launch stopped before scoring a query. Retry preserves the
same fixed experiment. [Protocol](biokg/FEATURE_COMPARE.md) and
[status](biokg/results/feature_compare/README.md). FP32/official average ties
are applied to both arms; historical fp16/pessimistic-tie output is not an
exact numerical baseline. Component swaps are diagnostic only. No TEST,
external checkpoints, model training or automatic promotion.

Matched report-half MRR (81,199 triples, both directions): original model
0.8317775870 -> improved model 0.8337635885; original complete feature
pipeline **0.8535134998** -> improved **0.8540538580**. Pipeline delta
**+0.0005403582**, 95% paired-triple interval **[+0.0002190631,
+0.0008650384]**. Positive in this development comparison but below the
predeclared +0.001 useful-gain threshold; no seed confirmation or promotion.

About 27% of the raw model gain survives after features. With original
blend weights fixed, swapping direct scores gives +0.000453, analogy alone
-0.000068 (interval crosses zero), and both +0.000484. Re-selecting blend
weights adds only +0.000056. There is no clear analogy benefit or established
geometry regression. A post-hoc saved-rank check finds that 1,181 of the
2,110 new model top-one successes (56%) were already solved by original
model+features. Thus overlap with existing retrieval is a measured factor,
not simply lack of learning in A. Do not turn report residuals into labels.

Both full-VALID plain-model rank vectors match archives exactly. Frozen
model hashes and source/input hashes are unchanged. Feature generation
811.04 s; through paired report/bootstrap 879.00 s; peak PyTorch allocation
0.38 GiB; GPU free. Independent saved-record audit passed: exact feature
normalization/ranks, fit-only recipes, pipeline/swap ranks and all summary
metrics/intervals reproduced, source/artifact hashes verified. Keep
existing H35F A and evaluate future changes by complete-pipeline gain.
No new run, full-VALID refit, TEST evaluation or submission change.

## RF1: rarity/support completed; useful blend-control result

User approved both post-training hypotheses. Keep improved A frozen and
test separately: (1) rarity-weighted Jaccard, based only on directed TRAIN
document frequencies; (2) candidate-specific support gates for ordinary
analogy/Jaccard, returning weak/absent retrieval evidence to A. No combined
arm. The second uses above-background TRAIN-holder similarities, not the
previously failed validation-confidence-bin gate. Include one descriptive
fixed-half-strength control. Baseline FC1 report MRR 0.8540538580.

[Protocol](biokg/RETRIEVAL_FOLLOWUPS.md),
[status](biokg/results/retrieval_followups/README.md). 41 synthetic tests
passed. Same finite fit-half recipe selection and paired reporting mask;
two adjusted primary intervals. No validation gradients, TRAIN graph
contamination, external checkpoint, TEST use or automatic promotion.
Neither primary variant advances. Report-half MRR: baseline **0.8540538580**,
rarity **0.8541798047** (delta +0.0001259467, adjusted 97.5% interval
[-0.0000812153, +0.0003217361]), support **0.8510670133** (delta
-0.0029868447, adjusted interval [-0.0035614879, -0.0024054833]).

The predeclared descriptive **half-strength + fit-only reselection** control
scores **0.8550978918**, delta **+0.0010440338**, descriptive 95% interval
**[+0.0006450757, +0.0014571628]**. This is a promising separate result,
not an automatically promoted primary success or evidence of seed robustness.

Crucial distinction: half-strength with old weights gives only **0.8497543394**.
The improvement requires recipe reselection. A convex blend of A and
`0.5*A + 0.5*R_j` is algebraically just another blend of the same original
five features. It changes the candidate mixtures reached by the finite
standalone-MRR-based search; it adds no new graph/model information. Global
effective A weight actually falls from 0.583175 to 0.533775. Do not describe
this as simply weakening retrieval or proof that a confidence gate worked.

Both proposed mechanisms remain unpromoted. Keep the unchanged baseline;
next recommended work is confirmation of the exact control recipe and a
separately registered finite blend-candidate comparison. No auto-sweep,
combination, new training, TEST or submission change. All 41 tests passed;
baseline and real audit-batch unchanged features reproduce exactly; frozen
model/source/input hashes verified. Generation 1,642.15 s, through all
comparisons/bootstrap 1,802.68 s, peak PyTorch allocation 0.23 GiB; GPU free.
Independent saved-record audit passed: source/artifact hashes verified;
normalization/support transforms, fit-only recipes, all saved ranks and
summary metrics/bootstrap intervals reproduced exactly, without loading
dataset splits. [Audit receipt](biokg/results/retrieval_followups/s0/endpoint_audit.json).

## BC1: fixed half-strength selection confirmation

User approved confirming the RF1 descriptive control. Registered
[BC1](biokg/BLEND_CONFIRMATION.md): same frozen improved-A score and four
TRAIN-derived cached retrieval features, exact RF1 0.5 interpolation, same
finite fit_recipe. No broader weight/strength search. Three fixed paired
partition seeds (0,1,2), each evaluated in both fit/report directions.
Seed0/fold0 must reproduce the existing recipes and ranks exactly. Report
each fold and partition plus within-triple-averaged paired bootstrap; no
best-split selection or score ensemble. Reused VALID is not a pristine
holdout; this is split stability, not model-seed robustness.

All six folds completed. Complete-partition deltas: seed0 **+0.0010806189**,
seed1 **+0.0008248413**, seed2 **+0.0009193754**; all six individual folds
also positive. Mean baseline **0.8545760316**, treatment **0.8555176435**,
delta **+0.0009416119**, descriptive within-triple 95% paired interval
**[+0.0006704451, +0.0012121762]**. Consistent-positive true, practical
+0.001 useful-gain flag false. Evidence of split stability on this reused
development data, not seed-robust generalization or second place. No automatic
promotion and no post-hoc threshold change. Direction deltas: tails
+0.0015750283, heads +0.0003081955; do not fit new switches from this summary.

50 synthetic tests passed. Original recipes/ranks reproduce exactly; all
source/input/checkpoint hashes unchanged. CPU 451.91 s through comparisons
and bootstrap, 467.12 s through final hashes; GPU unused. Independent audit
passed: all twelve fit-only recipes, report ranks, coverage, metrics and
clustered interval reproduced; source/input/artifact hashes verified.
No model or dataset split deserialization, validation gradients, TEST,
full-VALID refit or submission update. Stopped after six folds and audit.
Next research lead is a separately registered finite blend-candidate study,
not new architecture capacity or an automatic sweep.
[Status](biokg/results/blend_confirmation/README.md).

## CS1: candidate-side retrieval test

User approved the next retrieval hypothesis: compare each candidate with
the query's existing TRAIN answers, not only the query with the candidate's
TRAIN holders. Same own frozen A, signed-cube max/top3, no self-match,
unique TRAIN neighbors. [Protocol](biokg/CANDIDATE_RETRIEVAL.md).

Keep each BC1 half-strength fold recipe unchanged; blend in the fixed mean
of the two new normalized features with beta in {0,0.025,0.05,0.1,0.2},
selected on fit rows only with the existing min-2,000 hierarchy. Same six
folds across partition seeds0/1/2; matched baseline mean **0.8555176435**.
No path retrieval or broader old-feature weight search in this test.

Result: all six report folds completed, mean **0.8582166512** versus
**0.8555176435**, delta **+0.0026990077**, descriptive paired 95% interval
**[+0.0024912022, +0.0029036106]**. All three two-way partitions and all six
individual folds improve; both preregistered positive/useful-gain flags pass.
Tail MRR gains +0.0045832502; head gains +0.0008147652. Global beta is zero
in every fold, but 7–17 of 102 directed relation groups select a nonzero
beta. This supports complementary, relation-specific candidate retrieval.

60 synthetic tests passed. Initial CPU run stopped before evaluation to fix
repeated full-type-list conversion; sources and partial receipt retained,
all 60 tests passed again. Retry exactly reproduced both raw features for
all 7,164 queries from five completed original directed relations, all 501
candidates, max error 0.0. Successful CPU evaluation took ~17m35s; the occupied
1080 Ti was left untouched. Baseline report ranks exactly match BC1 and
model/source/input hashes are unchanged. Independent endpoint audit passed:
105 representative TRAIN-neighbor feature queries across 102 directed
relations, all normalization, six fit-only recipes, report ranks, coverage,
metrics and clustered interval reproduced; source/input/artifact hashes verified.
No new training, external trained artifact, validation graph input/gradients,
TEST, full-VALID refit or submission change. All validation is reused; this
is not a fresh holdout, model-seed robustness check or leaderboard score.
Stopped after the independent audit; no automatic promotion or next experiment.
[Status](biokg/results/candidate_retrieval/README.md).

## TP1: TRAIN-only typed paths — failed gate, stopped

User approved typed-path investigation and emphasized fail-fast behavior.
First stage is only three fixed drug–mediator–drug paths (protein,
sideeffect, disease), weighted by construction-TRAIN mediator degree.
Keep relation identities separate; choose seven finite path combinations
on fit-TRAIN, with relation/direction fallback. [Protocol](biokg/TYPED_PATH_PILOT.md).

Partition all TRAIN entity pairs across all relations/orientations into
90% construction, 5% fit and 5% report via fixed pair hash. This removes
held-out edges, reverse copies and alternate relations on the same pair.
Cap 256 sampled triples per target relation/role; both directions, 500
fresh typed negatives. No checkpoint comparator: current A saw TRAIN.
This is feasibility only, not an official MRR or pipeline gain estimate.

Fail-fast gate: report positive-path coverage >=10%, selected-path MRR
at least +0.01 over construction-TRAIN candidate popularity, descriptive
pair-cluster interval lower bound >0. Stop after this pilot and audit;
no automatic validation run or sweep even on a pass. No VALID, TEST,
external trained artifacts, GPU, model updates or submission changes.
Result: selected paths **0.2435084602** versus candidate-degree control
**0.4694555692** on 6,524 report-TRAIN triples (13,048 directed queries,
2,614 unique unordered pairs). Primary delta **−0.2259471090**, descriptive
pair-cluster 95% interval **[−0.2398670589, −0.2118654311]**. Coverage 96.0300%
passes, ranking signal fails. Side-effect paths dominate; shared-protein
coverage is only 8.2005%. Selected beats uniform +0.02506 and pooled +0.00176,
but neither overrides the failed primary gate. These are stratified TRAIN
pilot scores, not comparable to official VALID/TEST or our current pipeline.

**Stopped this exact variant before VALID**, as requested. No broader sweep
or follow-on run. Failure does not prove all typed paths useless or establish
their complementarity with A; this pilot deliberately loaded no checkpoint.
Current validation pipeline remains **0.8582166512**.

All 71 synthetic tests passed; CPU pilot completed in **28.19 seconds**.
Endpoint audit passed: grouped-pair split, graph, all samples/candidates,
203 direct feature examples (all 501 candidates, three paths, max error 0),
normalization, fit-only recipe, ranks, metrics and interval reproduced;
source/input/artifact hashes verified. Only TRAIN and raw counts opened.
Initial dense synthetic fixture lacked enough distinct unknown candidates;
the sampler failed closed, and the fixture was corrected before real data
without relaxing sampling rules. No VALID, TEST, GPU or submission changes.
[Status](biokg/results/typed_path_pilot/README.md).

## RS1: frozen reverse scoring — small gain, stopped at screen

User approved testing the same checkpoint's reverse-direction candidate score
to address the drug–drug top-ten fine-ranking weakness. This first test targets
drug–drug only; every other family and all old retrieval weights stay fixed.
See `biokg/REVERSE_SCORING.md` and `biokg/results/reverse_scoring/README.md`.

Add normalized score(candidate, learned inverse relation, source) with alpha
in {0, .025, .05, .1, .2}, using the existing finite fit-only selector. No
second model, new learned parameters, VALID graph edges, gradients or TEST.
CPU/four threads, LR=0. Verify exact baseline and reverse-model scoring first.

Fail-fast on seed0/fold0: full-pipeline gain >=+0.0005 and paired descriptive
95% lower bound >0; otherwise stop after that report half. If passed, complete
the remaining five existing folds, grouping partition/head/tail repeats within
each original triple. No automatic promotion, new experiment or submission.
Current complete pipeline aggregate 0.8582166512116752; matched first-fold
baseline 0.8578497365986548. Adaptive VALID reuse is explicitly disclosed.

Completed first fold: 0.8578497366 -> 0.8581763191, delta +0.0003265825,
descriptive 95% interval [+0.0001030824, +0.0005505058]. Below the preset
+0.0005 continuation gate, so five remaining folds not evaluated. DDI delta
+0.0016989029; tail alpha .05, head alpha zero. 591 top-one recoveries versus
513 losses (net +78). Limited complementarity, not proof of no benefit.
84 tests and endpoint audit passed; CPU 79.36 seconds, LR=0. No promotion,
TEST or model changes. Current aggregate pipeline remains 0.8582166512.

## TF1: TRAIN error forensics

User clarified: inspect one actual TRAIN drug–drug near miss to understand
why its correct answer loses. No additional scoring experiment or training.
Fixed sample of 128 TRAIN triples, both directions, whole drug catalog;
filter other known TRAIN positives including symmetric copies. Select first
filtered top-ten error, dissect candidate norm/alignment and 4x4 block margins,
and inspect TRAIN support with the focal edge/reverse excluded. Frozen own
model, LR=0, CPU only, no VALID/TEST/external trained artifacts. This is an
in-sample explanatory case study, not an official MRR or pipeline evaluation.
See `biokg/TRAIN_FORENSICS.md`.

Completed in5.69 seconds, 91 tests passed. First sampled case: TRAIN row1168263,
drug1381 --relation38--> drug786. Raw rank73 becomes10 after excluding other
known TRAIN positives; raw winner654 is itself valid in TRAIN. Remaining winner
1528 scores4.871401 versus correct4.694755. Its advantage is angular alignment
(0.469318 versus0.442798), not larger norm (0.907835 versus0.927317). Exact
margin decomposition: alignment+0.278230, norm−0.101584; total+0.176647.
13/36 blocks favor the answer,23 favor the competitor. Unit-target-norm
diagnostic rank3 does not fully fix it; reverse score also prefers competitor.

Removing the focal TRAIN pair/reverse leaves stronger best holder analogy
for the correct drug (0.833584 versus0.791847), and stronger candidate analogy
(0.757455 versus0.647325). This is evidence of a local structure/score mismatch,
not proof of its training cause or a failure of the full retrieval pipeline.
Only TRAIN/counts opened, immutable model/source/input hashes and direct
score/rank reconstruction verified. No extra model or submission changes.
[Case report](biokg/results/train_forensics/README.md).

## TF2: objective and existing-pipeline forensics — completed

User approved investigation 1, not k/block changes or retraining. Inspect
first eight TRAIN near-misses per direction from TF1, excluding repeated
unordered pairs. Rebuild exact existing retrieval on a focal-pair-removed
TRAIN graph with frozen seed0/fold0 BC1/CS1 weights and full drug catalog.
Check if these model-only errors survive the current pipeline.

On our unchanged H35F student and its original ten own teachers, inspect
CE/T=2 KD/weighted trajectory gradients against the temperature-independent
positive-minus-competitor margin. Focal loss versus whole2,048-row same-relation
batch,4,096 shared random typed negatives, plus a fixed hard-competitor-included
pool. No optimizer, parameter updates, VALID/TEST edges or new weight fitting.
These are conditional TRAIN diagnostics, not historical Adam replay or MRR
improvement estimates. [Protocol](biokg/OBJECTIVE_FORENSICS.md).

Results: current4x4 operators and full representation have corrective focal
total-loss directions on16/16 cases. Whole random-batch representation gradients
help13/16; operators alone help8/16. In three cases, other batch rows outweigh
the positive focal contribution. Competitor is present in6/16 random pools;
adding it to the10 missing pools makes all16 batch representation directions
helpful. This shared-pool intervention affects all2,048 rows; the effect is
predominantly from the other rows, not proof that row-local mining will work.

Teacher mean prefers the competitor on11/16 cases, but focal KD only opposes
the corrective representation direction on7/16 and does not dominate CE.
CE and weighted trajectory each help16/16 individually. No justification for
discarding KD/trajectory or claiming insufficient block expressivity from this
small first-order diagnostic. Actual Adam/history/generalization unmeasured.

Original case1381/r38/786: fixed current retrieval changes rank10->6 and
overtakes old competitor1528, but new winner1048 remains. Across16 selected
full-catalog TRAIN cases:7 ranks improve,5 worsen,4 tie; zero recover to rank1.
Not comparable to official VALID MRR; prior pipeline0.8582166512 unchanged.

117 tests and saved-record endpoint audit passed; main41.92 seconds,1.98GiB
peak GPU allocation. Student/all10 own teachers/source/input hashes unchanged;
no optimizer, gradients on VALID, TEST, external artifacts or submission edits.
No larger-k/8x8 or new training run launched.
[Detailed results and limitations](biokg/results/objective_forensics/README.md).

## TF3: protected shared competitor — completed, no training promotion

User approved the bounded next TRAIN-only diagnostic, not training. Keep
TF2's 16 cases and original batches, compare shared hard against a typed
random candidate with identical pool size and substitution multiplicity.
Test each with/without candidate-specific CE masking wherever that drug is
a known TRAIN answer. Retain KD/trajectory unchanged. Inspect focal margins,
background TRAIN answers and known-correct uses of the chosen drug, including
score/norm directional derivatives. Frozen own models, LR 0, no optimizer or
VALID/TEST access. Do not infer generalization or launch training automatically.

Completed all 80 arms on the 1080 Ti. Original random baseline gives corrective
focal gradients in 13/16; shared hard unprotected 16/16; shared hard protected
12/16; each matched random control 10/16. The chosen hard drug is a known TRAIN
answer in 55.39% of sampled batch-row occurrences. Unprotected hard harms the
mean margin on its known-correct uses in 15/16 case groups (baseline 6/16);
protection reduces this to 8/16 and improves the group mean versus unprotected
hard in all 16. Protection removes a false-negative CE penalty, but shared
parameters and unmasked KD/trajectory still couple queries. Focal improvement
versus random remains primarily competitor suppression, not correct-score lift.

Important limitation: 15/16 matched random candidates have zero TRAIN degree
for that relation, so hardness is not isolated from relation activity. Generic
background means change little; broader collateral/generalization unmeasured.
Do not promote this shared-hard variant. Possible separate next test: query-
specific, known-TRAIN-filtered relation-active hard candidates versus an
activity-matched random control, with KD retained. Not launched or approved here.

128 tests, CUDA synthetic check and endpoint audit passed; all 424 frozen-model
probe score/rank occurrences directly reproduced on CPU. Main 36.46 seconds,
1.663 GiB peak GPU allocation. Own model/teacher hashes unchanged, LR 0, previous
VALID pipeline MRR 0.8582166512 unchanged. GPU released, no training started.
[Protocol](biokg/PROTECTED_COMPETITOR.md),
[report](biokg/results/protected_competitor/README.md).

## TH1: small internal TRAIN reservation — created, no training yet

User approved reserving a small TRAIN part for the hide-a-link/predict-it
experiment. Fixed seed 36840 reserves 5% of unordered global entity pairs;
all relations, reverse copies and duplicate rows on a pair remain together.
No seed search. Official TRAIN untouched; future fit-only loader reads a
separate artifact, never the full TRAIN or reserved labels.

Fit: 4,526,393 rows / 2,114,965 pairs. Reserved: 236,285 rows (4.9612%) /
111,276 pairs (4.9984%). Every original row assigned exactly once; no pair
overlap. All 51 relations present in both parts. Keep the 1,112 held-out rows
with an endpoint absent from fit (781 entities); do not rebalance the split.
Rare relation coverage and this colder-node slice must be disclosed.

Future students, teachers and graph features must use fit-only provenance;
our existing all-TRAIN checkpoints are not clean warm starts or teachers here.
Learning targets for masked reconstruction must come from fit, not holdout.
Freeze the first recipe/endpoint/evaluation procedure before holdout metrics;
no metrics or candidate pools generated here. Repeated future feedback turns
the reservation into development validation, not pristine confirmation.

136 tests and exact artifact/fit-loader replay audit passed. CPU creation
15.46 seconds; no GPU/model/VALID/TEST access,
training or changes to the current 0.8582166512 pipeline. Existing training
runners are unchanged and must explicitly adopt the fit-only entry point.
[Protocol](biokg/TRAIN_HOLDOUT.md), [report and audit](biokg/results/train_holdout/README.md).

## MN1: masked-neighborhood single-model comparison — completed, not promoted

User explicitly narrowed the test to a fresh single model, no teachers or
distillation, and provided Vast RTX 5090 instance 50109054. Use TH1's fit-only
95% subset. Fresh k12/4x4 base: 50,000 steps. Fork model and optimizer states;
12,500 equal-budget ordinary versus masked-context continuation steps each.
Evaluate all reserved 5% rows only after both endpoints are frozen.

First concrete context: eight other same-relation drug–drug answers, target
excluded before sampling. Training-only query uses 25% normalized context;
loss is 75% ordinary CE + 25% assisted CE + original 0.1 trajectory. No remaining
neighbors and non-drug–drug rows use ordinary loss exactly. Inference stays
the ordinary single scorer; no extra parameters, ensemble or graph branch.
Unfiltered original negatives remain in both arms; this is not a TF3 fix.

Primary gate: masked minus continued-control internal-heldout MRR >=0.001,
lower pair-cluster bootstrap 95% bound >0, and masked >= frozen base. No early
checkpoint selection, official VALID/TEST or automatic follow-on/promotion.
Fit-only TRAIN probes provide progress, not a held-out score. A failed endpoint
stops this fixed version. All code/data receipts pinned before training.

151 local tests, 15 remote tests and full-size synthetic 5090 smoke passed.
Remote uv environment isolated and pinned; finite supervisor job has restart
disabled. No old checkpoint, full official dataset or external trained weight
transferred.

Completed endpoint MRR on 472,570 internal-heldout queries: fresh base
**0.6204363194**, ordinary continuation **0.6182218144**, masked continuation
**0.6173322471**. Primary masked minus control **-0.0008895674**, pair-cluster
95% interval **[-0.0009996977, -0.0007779612]**; gate failed. Drug–drug MRR:
base 0.2758951508, control 0.2703584013, masked 0.2661412842. This fixed
training-only mixture did not help. Ordinary continuation also lost to base;
do not interpret the result as proof that 4x4 capacity or training time is
the bottleneck. Single seed; the interval does not measure seed uncertainty.

Training 259.06 s, evaluation 27.07 s, endpoint replay audit 26.45 s on the
5090, excluding setup/startup overhead. Full candidate/rank/summary replay
passed, no holdout gradients or model changes during evaluation. All three
checkpoints and receipts backed up locally with verified hashes. Job exited,
GPU released; rented instance not stopped/destroyed. No next run, automatic
sweep, official VALID/TEST access or submission change. These internal scores
are not comparable to the existing official-VALID pipeline MRR.
[Protocol](biokg/MASKED_NEIGHBORHOOD.md),
[completed results](biokg/results/masked_neighborhood/README.md).

## NL1: same 4x4 matrices, nonlinear function — completed, not promoted

User approved changing the function, explicitly not increasing matrix size.
Fresh matched seed-0 pair, 50k steps each on TH1 fit, same initialization,
batches, CE + 0.1 trajectory and cosine optimizer schedule. Control uses
q=normalize(Hx); intervention q=normalize(z/(1+|z|)), z=Hx, with coordinatewise
complex magnitude and phase preserved. Fixed alpha 1, active at inference.
Same 27,124,129 real parameters; no B, KD, ensemble, graph context or external
weights. Initial and endpoint fit-only magnitude/attenuation diagnostics can
show whether free operator scaling weakens the activation, not select alpha.

Evaluate all reserved pairs only after both fixed endpoints. Reuse MN1's exact
candidate policy and pair-cluster bootstrap. Gate: nonlinear minus linear MRR
>= .001 and lower 95% bound > 0. No automatic alpha/LR/loss sweep or promotion.
64 local synthetic/regression tests and 27 remote tests passed; full-size
synthetic smoke passed (20 paired steps in 0.11978 s).

Completed internal holdout MRR: linear **0.6204363194**, nonlinear
**0.6194131842**. Primary delta **-0.0010231352**, pair-cluster 95% interval
**[-0.0015737196, -0.0005040952]**. Gate failed. Drug–drug delta -0.0037710912.
Final fit probe also slightly worse: 0.8825787593 versus 0.8812469370. Saturation
remained active (median attenuation 0.87872, mean nonlinear/linear query cosine
from the same nonlinear weights 0.99019); a changed function did not yield an
MRR improvement. Reject this fixed version, not every same-storage function.

Read-only full candidate/rank/bootstrap audit passed, including fresh initial
state regeneration. Linear endpoint model and optimizer hashes exactly match
the earlier MN1 base; batch and candidate streams match, confirming the control
without old checkpoints as training inputs. Single seed, not seed replication.
Training 330.75 s, evaluation 26.08 s, audit 25.52 s, excluding setup/startup/
backup. All artifacts backed up locally with verified hashes. Job exited,
GPU released; rental not stopped/destroyed. No follow-on sweep, official
VALID/TEST access or submission change. [Protocol](biokg/NONLINEAR_OPERATOR.md),
[completed results](biokg/results/nonlinear_operator/README.md).

## H37: exact typed squared likelihood — repair completed, overall gate failed

User approved correctness/cost pilot followed by one fresh matched pair if
practical. Same 27,124,129 parameters and 50k updates each, no NL1 activation,
B, ensemble, teachers or distillation. Control keeps sampled CE + .1 trajectory;
squared uses epsilon+(tau*dot)^2 and its exact typed conditional likelihood,
plus the same trajectory. Fixed epsilon1e-6. A scorer-and-objective experiment,
not an isolated squaring ablation or reproduction of ComplEx2. Gram statistic
recomputed with gradients each step; eligible-table gradients are dense.
Metadata entity IDs, including cold-in-fit rows, enter the denominator, not
held-out labels. Only TH1 fit may be read during training.

52 local tests passed, including explicit float64 probability/gradient parity,
fp32 full-model comparison, gradient support, sparse/dense optimizer rule and
checkpoint/candidate/evaluation checks. Finite remote job starts with tests
and synthetic worst-type pilot, gates at <=20ms/pair and <=8GiB allocated.
If these fail, no real-data training; if they pass, one fixed comparison.
Primary all-holdout gate delta>=.001 and lower95%>0. No interim held-out score,
alpha/LR/epsilon sweep, official VALID/TEST or submission change.
Remote pilot passed: 32 tests, explicit GPU probability/gradient agreement,
7.05535/7.19272 ms per paired step, peak allocation 595,214,336 bytes on the
synthetic largest type. Original real-data attempt then stopped on nonfinite
gradients before the first MRR probe. Bounded TRAIN-only replay reproduced at
step1426: one source squared norm underflowed to zero, causing288NaNs in its
entity gradient; weights, query, loss and H/tau gradients remained finite.
No held-out score or conclusion about MRR from this failed attempt.

H37N is a numerical repair, not a new loss/hyperparameter trial: clamp squared
norm to dtype tiny before sqrt in both arms, preserve existing epsilon and all
other settings. Local tests verify identical fp32 forward values, unchanged
ordinary gradients and finite zero/underflow gradients. Separate fresh retry
requires identical recorded failing-batch forward query/loss and finite
gradients, then the same GPU cost gate. No resume from the diagnostic state.
Original source/protocol/receipts unchanged; adapter/new source hashes pinned.
[Repair protocol](biokg/H37_NUMERICS.md), [completed retry](biokg/results/h37_stable/README.md).
[Protocol](biokg/H37.md), [status/results](biokg/results/h37/README.md).

H37N completed 50k updates per arm. All-query internal-holdout MRR:
linear0.6204363194, squared0.5741653145; delta−0.0462710049,
pair-cluster95%[−0.0497774451,−0.0425128631]. Fixed advance gate failed.
Drug–drug is a substantial exception: 0.2758951508 -> 0.5011380910 on
110,520 queries; the remaining362,050 queries fall from0.7256115437 to
0.5964577292. These exploratory slice results do not override the overall
gate or establish which changed component caused the difference. Final
in-sample fit-probe MRR also falls, 0.8825787593 -> 0.6590160339.

The recorded-batch numerical repair preserves forward query/loss but leaves
a finite entity-gradient norm of6.351189180416e12, which would dominate that
batch's global clip1. Its frequency and causal contribution to the MRR loss
are unmeasured. Proposed next diagnostic, not launched: fixed FIT-only
per-row norm / per-group gradient / clipping-factor measurements with
relation slices, before any stabilization trial. This is not evidence that
4x4 capacity is exhausted or that all squared objectives fail.

88 distinct local tests and39 remote tests passed. Endpoint audit passed;
baseline model/optimizer and batch/candidate streams match MN1 exactly.
Training365.69s, evaluation25.71s, audit25.45s. Endpoints, ranks, receipts and
failure artifacts backed up locally with verified hashes. On2026-09-07,
user requested stopping for the night: Vast50109054 confirmed stopped
(actual exited, intended stopped), disk retained, possible storage charges.
No follow-on training, official VALID/TEST access or submission changes.

## Sequential experiments

| Order | Hypothesis / change | First decision |
| --- | --- | --- |
| 1 — H34: screened, not promoted | Relation-aware analogy: replace `cos(E[h], E[u])` by `cos(H[r]E[h], H[r]E[u])` for TRAIN holders `u` of each candidate. Keep max / top-three pooling, model scores and Jaccard unchanged. No training. | Report-half delta −0.001267; bootstrap interval crosses zero. No evidence to advance this exact variant. |
| 2 — H35F completed; warm-up not promoted | Direct TRAIN CE/KD taught B to rank, but the complete warm-up strategy lost to both controls. | Full VALID: single 0.8340991, usual joint 0.8338250, warmup+joint 0.8321678. B alone 0.6751036; small +0.0003258 correction to its own A, insufficient net gain. No next run launched. |
| 3 — H36 | Structural cross-fitted distillation. Train teachers / build features without one TRAIN fold, predict that fold, teach the student with those soft targets and TRAIN labels. | First establish that the teacher adds information beyond the existing student. Then test whether the student retains the gain with teachers removed at inference. |
| 4 — H37/H37N completed; not promoted | Squared-score typed conditional likelihood with exact typed normalizer and unchanged 4x4 learned storage. Original numerical failure repaired in a separately recorded fresh matched pair. | Overall internal-holdout delta −0.0462710, paired 95% interval entirely negative. Drug–drug improves, other relations lose more. Investigate gradient/norm dynamics before any further trial; nothing launched. |

H35 has a multi-semantic precedent in [TransG](https://arxiv.org/abs/1509.05488),
not evidence of a BioKG gain. H37 is inspired by
[generative KGE circuits / ComplEx²](https://arxiv.org/abs/2305.15944), but is
our proposed adaptation, not a reproduction of that leaderboard entry.

## H34 preregistration: relation-aware analogy

Register before looking at H34 real-data results:

- Use released BioKG `dist_T2_s0.pt`, checksum verified; one frozen model.
  Read its training configuration and verify entity/relation indexing.
- First screen: 5,000 validation triples sampled without replacement by
  NumPy `default_rng(3400)`, sorted; both directions, all 500 official typed
  negatives. Use the local GTX 1080 Ti (user-approved); leave the rented GPU
  free for other work. CPU fallback does not change the protocol.
- A separate runner explicitly opens only TRAIN and VALID through the
  selective loader. Do not run the legacy end-to-end retrieval shell script:
  it includes test caching/evaluation, and legacy feature loaders call the
  all-split loader even for a validation request.
- Recompute fp32 student, global-cosine analogy (max/top3), relation-cosine
  analogy (max/top3), and Jaccard (max/top3) from TRAIN. Use unique TRAIN edges,
  exclude the query source from holders, preserve the legacy missing-holder
  sentinel -1 and signed cube for cosine features.
- Relation cosine uses normalized `model.hop(model.embed(...), directed_r)`
  for both source and holder; no newly learned projection. Audit how much the
  metric actually changes: a truly unitary H would preserve cosine. Report
  operator non-unitarity and feature differences instead of assuming novelty.
- Control: student + global analogy pair + Jaccard pair. Treatment: replace
  only the analogy pair. The student and Jaccard features are shared unchanged.
- Use a fixed triple-paired tuning/report split from `default_rng(3401)`.
  For each arm separately, select from the same finite global mixture grid:
  model weight in `{0.5, 0.75, 0.9, 1.0}`; share of remaining feature weight
  assigned to analogy in `{0, 0.5, 1}`; within each pair, equal max/top3 weights.
  Include the unchanged existing five-feature uniform mean as a control.
  Normalize each feature row by fp32 z-score. No per-relation fitting at this
  small-screen stage. This is a diagnostic control, not an exact recreation
  of the historical per-relation-tuned submission pipeline.
- Primary screen: held-out paired MRR delta. Also report fixed-uniform delta,
  model-only MRR, Hits@1/10, family/direction deltas, cosine change, ranking
  ties, and paired-triple bootstrap interval (1,000 replicates, seed 3402).
  The interval quantifies query variability, not seed uncertainty or immunity
  to adaptive validation reuse. No running score-driven changes.
- Advance to full validation if held-out delta is at least +0.001 and its
  paired bootstrap lower bound exceeds zero. Borderline results are
  inconclusive, not a success. On promotion, preregister the full-validation
  comparison using matched historical-style finite mixture choices; confirm
  seeds 0, 1, 2 before adopting it. A short-screen loss rejects only this exact
  zero-training metric variant, not all relation-aware retrieval ideas.
- Do not automatically start H35 while H34 is running. Record the outcome
  here before beginning the next hypothesis.

## Experiment ledger

- H33 random auxiliary: stopped at user request after the first completed
  full-length pair lost **0.0023838 MRR** (0.8130752 vs 0.8154589). Seed 1
  baseline completed; its random arm was interrupted. No three-seed claim.
  [Detailed receipt](biokg/results/h33/README.md).
- H34: completed on the local 1080 Ti in 26.86 seconds after loading. Matched
  report-half control MRR **0.8410783**, relation-aware **0.8398111**, delta
  **−0.0012672**, paired-triple bootstrap 95% **[−0.0033761, +0.0009694]**.
  Inconclusive and below the advance gate; **not promoted**. These are subset
  scores with global mixture selection, not the historical submission recipe.
  43 BioKG tests passed, plus post-run official-evaluator agreement for all
  seven features and both mixtures. [Detailed receipt](biokg/results/h34/README.md)
  and [immutable launch protocol](biokg/results/h34/protocol_at_launch.md).
- H35 completed: frozen student **0.8321749** full-validation MRR; single-bank
  refinement **0.8309517**, mean **0.8312715**, OR **0.8312383**, AND **0.8313425**.
  No promotion. A/B query cosine stayed near **0.9986**: little specialization
  under this frozen-table/first-bank, CE-only recipe. [Detailed receipt](biokg/results/h35/README.md).
- H35B completed, separately preregistered as adaptive follow-up: same-block
  composition **0.8303889**, cross-block shuffle **0.8306100**. No promotion.
  Same TRAIN stream and budget; identity initialization reproduced the frozen
  probe exactly; no test use. [Detailed receipt](biokg/results/h35b/README.md).
- All six H35/H35B adaptations are archived. **58 tests passed at that stage.**
  The two-bank architecture is not ruled out:
  embeddings/A were frozen and adaptation omitted the original distillation
  signal. A future two-bank test should isolate joint training with retained
  distillation against a matched single-bank control.
  Do not start another frozen-bank initializer/temperature/permutation sweep.
- H35C completed: all parameters jointly trained with original T=2 KD;
  single **0.8327348**, dual **0.8327025**, dual-minus-single **−0.0000323**
  (adjusted 97.5% interval **[−0.0001516, +0.0001029]**). No promotion.
  Bank-query cosine **0.99879**, B positive-score responsibility **47.71%**;
  no useful dual-bank gain. A-alone 0.8343624 is diagnostic, not a selected
  checkpoint. All ten teachers unchanged, every student parameter group
  changed; 68 tests and endpoint audit passed. 466.79 seconds training plus
  probes/audit/checkpoint writes; about 2.18 GiB peak allocated. No TEST.
  [Detailed receipt](biokg/results/h35c/README.md).
- User then explicitly approved different information and a different
  initialization for B. TRAIN-neighborhood context is one pending option; it
  must exclude the queried training edge, score validation candidates
  symmetrically, retain KD, and compare against a matched single.
- Subsequent RelEns paper review refined the diversity hypothesis: different
  scoring geometries and objectives can also supply complementary evidence;
  a new graph input is not mandatory. RelEns uses heterogeneous base models
  and candidate-rank aggregation, unlike our current same-family mean-logit
  distillation. Larger blocks alone do not ensure distinct useful experts.
- H35D preregistered after user approval: compare equal-weight
  mean-logit and rank fusion of our ten frozen teachers against the original
  student, on full VALID. Primary gate: rank-minus-logit MRR >= +0.001 and
  paired-triple 95% lower bound > 0. No weights fitted, no training or TEST,
  no automatic next architecture run. 78 synthetic tests and full-size local
  GPU smoke passed. [Fixed protocol](biokg/H35D.md).
- H35D completed: full-VALID mean logits **0.8427075**, rank fusion
  **0.8400257**, original student **0.8321749**. Rank-minus-logit delta
  **−0.0026818**, paired-triple 95% **[−0.0029067, −0.0024585]**: **not
  promoted**. Mean-logit teachers retain a **+0.0105326** gap over the student,
  not a guaranteed compressible gain. Student ranks match H35 exactly; all
  eleven models unchanged, no gradients or TEST. Evaluation 68.28 seconds,
  about 1.27 GiB peak allocated on the local 1080 Ti. No follow-up training
  started. [Detailed result and interpretation](biokg/results/h35d/README.md).
- H35E user-approved, preregistered and launched: three arms (single, private
  dot, private L1 distance), exactly 5,000 matched TRAIN updates each, original
  CE + T=2 mean-logit KD + A trajectory. All parameters co-adapt. A LR 0.0001;
  new B/maps/gate/temperature LR 0.001; cosine to zero. B is 32 real features,
  independently initialized; dot/distance initial tensors and parameter counts
  identical. Single 27,124,129 real parameters; each private-B arm 30,134,760.
  Primary geometry comparison plus single/frozen guards use three adjusted
  intervals; no adaptive sweep or automatic follow-up. 91 tests passed;
  full-size GPU smoke: 20 triplet updates in 5.14 seconds, 3.49 GiB peak,
  all groups updated, teachers unchanged, exact candidate-permutation scores.
  Estimated training about 22 minutes, local 1080 Ti only. No TEST.
  [Fixed protocol](biokg/H35E.md).
- During H35E, user raised joint-from-scratch training and random B/GAN-style
  initialization. Clarified that B's entity table is already independent random
  initialization; its relation scales start at identity and translation at zero,
  while only A is pretrained. Strong A plus a small initial residual gate could
  nevertheless disadvantage cold-start B; this is a hypothesis to inspect in
  endpoint gate/branch diagnostics, not an established cause. A possible future
  test is TRAIN-only B warm-up then joint training with matched-budget controls;
  a full joint-from-scratch run needs a longer separately registered budget.
  Neither follow-up is launched, and the in-flight H35E recipe is unchanged.
- H35E completed unchanged: single **0.8327347614**, dot **0.8326823323**,
  distance **0.8326803530**. Distance-minus-dot **−0.0000019793** (adjusted
  98.3333% interval **[−0.0000906463, +0.0000861238]**); distance-minus-single
  **−0.0000544085**. **No promotion.** The single arm exactly reproduced all
  H35C single-endpoint ranks. All 5,000 updates/arm completed; every student
  group changed; ten teachers unchanged; independent endpoint audit passed.
- H35E clarified the user's concern: A/B score correlation is near zero,
  unlike H35C's duplicate-query behavior, but B-alone MRR is only **0.01291 /
  0.01354** (dot/distance). Private temperatures fell from 1 to **0.356 / 0.335**;
  query-weighted gates fell from 0.05 to about **0.0405 / 0.0401**. Mean per-query
  gated B correction/A score-standard-deviation ratio is just **0.279% / 0.219%**.
  This is consistent with weak cold-start B being suppressed, not proof that
  joint-from-scratch training or B warm-up will succeed. Random B already
  existed; a future B learning phase needs its own TRAIN CE/KD and matched
  budget controls. No GAN objective and no follow-up run were started.
- H35E training/probes/audit/checkpoint writes took **1,203.38 s** (~20.06 min),
  peak **3.52 GiB** on the local 1080 Ti. Distance updates cost 740.49 s versus
  dot 172.48 s; equal parameters/updates do not imply equal compute. Local
  compute slot released; rented GPU untouched; no TEST. [Full result](biokg/results/h35e/README.md).
- User-requested post-hoc original-error audit: frozen student misses top-1 on
  75,691 directed VALID queries. Trained single fixes 3,501 original errors and
  breaks 3,246 originally correct predictions; dot fixes/breaks 3,547/3,320;
  distance 3,527/3,300. B yields some different recoveries but no net improvement:
  26–46 extra original-error fixes are outweighed by 54–74 extra breakages.
  These validation error cohorts are diagnostic only; do not use them as
  training supervision. Any error-targeting follow-up must use TRAIN-only
  targets or held-out TRAIN folds, and is not launched.
- H35F explicitly approved and preregistered: reuse dot-B architecture, width
  and initialization. Three arms get the same 15,000 TRAIN batches: single A;
  original joint A+B; direct B CE + T=2 mean-logit KD for 10,000 steps while
  A/gate freeze, then 5,000 joint steps with the original loss (no remaining
  B-only auxiliary loss). A LR 0.0001/B LR 0.001 share a 15,000-step cosine,
  no phase restart or optimizer reset. Freeze/schedule exposure is part of
  the strategy tested; total examples/calls match, not per-parameter updates.
- H35F launch checks passed: 101 tests, including 10 new warm-up tests; full
  synthetic GPU smoke for both phases passed direct-loss gradients, frozen
  A/gate hashes, state-preserving switch, teacher freezing and exact candidate
  permutation. 20 triplet updates per phase took 2.511/2.396 s; peak allocation
  2,771,709,440 bytes (~2.58 GiB), estimated training ~31 min. Local 1080 Ti,
  no TEST or rented GPU. Progress includes separate B-alone probe MRR and
  effective LR zero for frozen groups. [Fixed protocol](biokg/H35F.md).
- H35F completed: single **0.8340991235**, usual joint **0.8338250000**,
  warmup+joint **0.8321677853**. Warmup-minus-joint **−0.0016572147**, adjusted
  98.3333% interval **[−0.0020000205, −0.0013231639]**; warmup-minus-single
  **−0.0019313382**, **[−0.0022556648, −0.0016251523]**. Warmup was essentially
  unchanged from frozen (−0.00000708). **No promotion.**
- Direct warm-up taught B a real ranking signal: B-alone full VALID
  **0.6751036**, versus usual-joint B **0.0121618**; A/B candidate-score
  correlation **0.7510**, mean B correction/A score spread **4.61%**. Adding
  warmed B to its own A improves MRR **+0.0003258**, recovering 1,530 top-1
  queries while losing 1,320, but cannot bridge the gap to the single control.
  B alone is not competitive with A as a replacement. No B-only export.
- H35F boundary audit: original A and gate unchanged through 10,000 steps;
  all B groups changed; correct Adam counters, no optimizer/LR restart. All
  15,000 optimizer calls/arm completed; warmup A/gate 5,000 parameter updates,
  B 15,000. All teachers unchanged, no gradients; 101 tests and independent
  endpoint audit passed. Training/probes/audits/writes **1,679.35 s** (~28 min),
  peak **2.58 GiB**. Local GPU compute slot released; no TEST or rented GPU.
- Strongest H35F arm was continued single-A training, +**0.0019243** vs frozen
  (post-hoc descriptive 95% **[+0.0016413, +0.0022317]**). It exceeds prior
  5,000-update single by +0.0013644, but schedules differ. This control result
  is a candidate for separate confirmation, not an automatically promoted
  submission. The warmup comparison tests a full strategy with different A
  update/schedule exposure, not warm-up alone. [Full result](biokg/results/h35f/README.md).
