# HC1: the learned combiner fit on a TRAIN holdout, validation-only test

Registered 2026-09-08 00:30 (before any holdout cache, teacher or blend result
exists). User approved this variant: one 97% teacher per seed, no twin
students, combiner fit on the teacher's holdout scores, applied to the
released 100% student; validation decides whether it transfers.

## Why

Rows F / C-F / ensemble (`README.md`, wikikg2 table) are not filed because
`learned_blend.py` fits ~1,500 per-(relation, direction) weights by Adam on
validation labels, which OGB's rule excludes ("standard hyper-parameter tuning;
not allowed: gradient-based search, use as model input"). Fitting the same
combiner on reserved TRAIN rows is training on training data. The deployed
system stays the released one (100% student + members on all of TRAIN); only
the combiner's fit data changes. Hold-out stacking: the fit features come from
twins that never saw the held-out rows.

## Fixed design

* Holdout: `holdout_wiki.py`, pair-grouped (min·N+max SplitMix64, seed 36840,
  the TH1 hash), buckets 97–99 of 100 (~3% of pairs), every row of a reserved
  pair held out together; 500 uniform negatives per direction, seed 36840;
  relations that would vanish from fit keep their pairs in fit (listed in the
  receipt). No search over seeds or cutoffs. Fit-TRAIN is what `load()` hands
  every script when `WIKI_HOLDOUT` is set; `valid` and `test` are untouched.
* Fit world (5090, box 50207787): teacher `model_fit97_s0.pt` = the row-A
  recipe unchanged (800k steps, k=8, block 64, rev-frac 0.75, rowadagrad,
  table-lr 0.6, seed 0) on fit-TRAIN, `--eval valid` (validation as a monitor
  only). Members on the holdout split from fit-TRAIN: `analogy_f0_t3`,
  `holders`, `cn_aa`, `linked`, `cn3_aa`, `typed` (table `typed_lo_fit97.npz`
  built from fit-TRAIN). The four self-augmented members are excluded from
  HC1 (their proposals come from a 100% teacher and their thresholds from
  validation); seven members in both worlds.
* Apply world: released `model_dist_s1.bf16.pt` (student, seed 1; seed 0 is
  not in the release) and `model_wiki_s0.bf16.pt` (teacher, seed 0, the
  teacher→teacher diagnostic), members on validation from all of TRAIN:
  `analogy_d1_t3` / `analogy_s0_t3`, `holders`, `cn_aa`, `linked`, `cn3_aa`,
  `typed`. Existing valid caches (jinx, `/mnt/geocore/wiki/ens_cache`) are
  reused where present; missing ones are built with the unchanged scripts.
* Combiner: `learned_blend.fit_all` unchanged (global → direction → relation,
  Adam, l2 1e-3, 300/200 steps). Guard chosen inside the holdout by
  cross-fitting on halves (both folds) over {250, 500, 1000, 2000}; ties go to
  the larger guard. Then one fit on the full holdout, applied as is to
  validation (`holdout_blend.py`). No test cache is read or written.

## Gate (fixed before any number is seen)

On the held-out half of validation (seed 0, the half `learned_blend.py search`
reports), with the same seven members and the student as model channel:

    G    = learned combiner fit on the other half (validation labels) − selection blend fit on the other half
    PASS = holdout-fit combiner ≥ selection blend + 0.5·G

i.e. the holdout fit must keep at least half of the gain that fitting on
validation labels gives over the allowed selection blend. The full-validation
Evaluator number is reported alongside (nothing was fit on it).

Interpretation, fixed now: PASS on the student → the recipe is filable as it
stands (combiner trained on TRAIN rows; one test read per seed would follow
under a separate registration). FAIL on the student but PASS on the
teacher→teacher diagnostic → the model channel's calibration is the problem;
the twin-student route (fit students on fit-TRAIN) is the next step. FAIL on
both → the holdout (older edges, time split) does not stand in for validation
and the combiner stays unfiled; nothing further is tuned on this evidence.

## Data boundary

Fit world reads official TRAIN only (through the holdout file) and validation
as the teacher's monitor. Apply world reads validation to report. Test is not
opened. Every holdout number is a TRAIN number; every validation number is
reported once, after the fit is frozen.

## Cost

One 5090 at $0.404/h (box 50207787): teacher ~40 min, members ~30 min
concurrent, retrieval ~20 min after the teacher; ~1.5 h. The 1080 Ti on jinx
scores the released student/teacher on validation (fp32 tables) and runs the
fit + report.

## HC1 RESULT (2026-09-08, box 50209059, 5090): FAIL on the student

Fit world as registered: teacher `model_fit97_s0` on fit-TRAIN reached
validation MRR 0.6994 (the 100% teacher: 0.7041, −0.0047); probes tracked
the 100% teacher within −0.004 throughout. Guard sweep on holdout halves:
250 → 0.8028, 500 → 0.8018, 1000 → 0.7998, 2000 → 0.7990; guard 250, 192
local groups, in-sample holdout 0.8039. Applied to the released student
(s1) with the seven members on all of TRAIN, full validation, official
Evaluator: **MRR 0.6967**, hits@1 0.6216, hits@10 0.8431 (tail 0.942, head
0.452). On the held-out half of validation (seed 0): model alone 0.7186,
selection blend 0.7542, validation-fit learned combiner 0.7826,
holdout-fit combiner 0.6970. Gate bar 0.7684 → **FAIL** (below the model
alone). Receipts: `results/hc1/dist_s1.{json,log}`, weights
`results/hc1/weights_dist_s1.npz`, split `results/hc1/holdout_wiki.json`.

Cause, visible in the split's own receipts: the pair-grouped holdout
holds out every TRAIN row of a reserved pair, so a held-out query is never
already linked to its answer in fit-TRAIN (0% of holdout rows). Validation
queries are linked to their answer under another relation 9.3% of the
time, test 12.1%, and 10.6% of TRAIN rows sit on multi-row pairs. The
graph members encode exactly that link: on the holdout `linked` scores
0.002 alone and the fit gives it −4.04 in the head direction (validation
fit: +0.12); `holders` and `analogy` also flip sign. The holdout was the
wrong analogue of the time split for this dataset by construction, not
a calibration or drift finding. Teacher→teacher diagnostic (same fit applied to the released 100% teacher s0):
full validation **0.6874** (alone 0.7041); on the half: model alone 0.7042, selection 0.7450,
validation-fit combiner 0.7740, holdout-fit 0.6876, bar 0.7595 → FAIL, the same −200% of
the gain. Both channels fail identically, which is what a defect in the split, not in the
model channel, predicts. Receipts: `results/hc1/wiki_s0.{json,log}`, `results/hc1/teacher_s0.log`.

## HC2 (registered 2026-09-08 23:35, user approved): row-level holdout, one attempt

Change, and the only change: `holdout_wiki.py --grouping row` buckets
TRAIN rows independently (same hash, seed 36840, cutoff 97), so other rows
on the same pair stay in fit-TRAIN. A row-level 3% holdout has 10.3% of
its rows linked to their answer in fit-TRAIN, the validation statistic.
Everything else as HC1: same recipe teacher on the new fit-TRAIN, same
seven members, same guard sweep inside the holdout, same apply world
(student s1, teacher s0 diagnostic), same gate on the same validation
half. This is a second attempt after seeing HC1's validation number; the
defect it corrects was diagnosable from the split alone, and no
validation label informs the change. If HC2 fails the gate too, the
holdout route is closed: the combiner stays unfiled and the paper keeps
the rows as reported.

## HC2 RESULT (2026-09-08 00:12, box 50209059): FAIL on the student; the holdout route is closed

Row-level holdout: 484,833 rows (3.0%), 49,831 (10.3%) linked to their answer
in fit-TRAIN (receipt `results/hc2/holdout_wiki_row.json`). Teacher
`model_fit97r_s0` on its fit-TRAIN: validation 0.6977 (100% teacher 0.7041,
−0.0064). The defect of HC1 is gone in the fit world: `linked` scores 0.102
alone on the holdout (HC1: 0.002) and takes +0.18 (tail) in the fit. Guard
sweep on holdout halves: 250 → 0.8052, 500 → 0.8034, 1000 → 0.7999,
2000 → 0.7949; guard 250, 198 local groups, in-sample holdout 0.8064.

Applied to the released student (s1), full validation, official Evaluator:
**MRR 0.7266**, hits@1 0.6529, hits@10 0.8713 (tail 0.965, head 0.489). On
the held-out half: model alone 0.7186, selection blend 0.7542,
validation-fit learned combiner 0.7826, holdout-fit combiner 0.7267. Bar
0.7684 → **FAIL** (keeps −97% of the combiner's gain over selection: it no
longer hurts, +0.008 over the model alone, but it stays below the allowed
selection blend). Receipts: `results/hc2/dist_s1.{json,log}`,
`results/hc2/weights_dist_s1.npz`, `results/hc2/teacher_s0.log`.

Teacher→teacher diagnostic (released 100% teacher s0): full validation **0.7171**
(alone 0.7041); on the half: model alone 0.7042, selection 0.7450, validation-fit combiner
0.7740, holdout-fit 0.7172, bar 0.7595 → FAIL (−96% of the gain). Same pattern as the student,
so the model channel's calibration is not the cause. Receipts: `results/hc2/wiki_s0.{json,log}`.

Reading: what remains is the head direction. On held-out TRAIN rows the
tail direction is nearly solved by the model (0.975 alone) and the
head-direction fit gives `analogy` and `linked` the opposite sign to the
validation fit (holdout +0.15 / −0.31; validation +0.24 / +0.12). Older
edges and newer edges want different head-side weights; that is the time
split, not a construction error, and no holdout of May-2015 edges will
stand in for August-2015 queries on this axis. Decision, as registered:
the holdout route is closed; the learned combiner stays unfiled and the
paper keeps rows F / C-F / ensemble as reported-not-filed. The routes that
remain are outside this registration: a zero-order (non-gradient) search
of the same per-relation weights on validation, which is what the board's
current first entry (RelEns, TPE on validation MRR) does, or asking the
OGB maintainers directly. Both are the user's call.

## HC3 (registered 2026-09-08 00:40, user approved after the degree finding): HC2 fit rows, degree-matched weights

Finding that motivates it (query-side and answer-side entity degrees, in the
graph the members read): heads look alike in every split (median 6–8); tails do
not. Validation tails: p25 / median / p90 = 24 / 642 / 82k; test 49 / 1,230 /
55k; training rows and the HC2 holdout 150 / 4,600 / 1.2M. The row holdout
also over-samples cold heads (7.0% with degree ≤ 3 vs 1.3% on validation). In
the head direction the tail is the query entity and the members are built from
its neighbourhood, so HC2's head-direction weights were fit on hubs and applied
to modest entities.

Change, and the only change: the HC2 fit world (same teacher `model_fit97r_s0`,
same seven holdout caches, nothing retrained) with an importance weight per
held-out triple, applied to both of its directions:
w = P_valid(cell) / P_holdout(cell), cell = (⌊log2(deg(head)+1)⌋, ⌊log2(deg(tail)+1)⌋),
each capped at 20; validation degrees in full TRAIN, holdout degrees in
fit-TRAIN; cells with no holdout mass are dropped (their validation mass is
reported); weights clipped to [0, 50] and scaled to mean 1; effective sample
size reported (`hc3_weights.py`). The listwise fit uses the weighted
cross-entropy (`learned_blend.fit_weights(w=...)`); the guard is chosen by the
weighted cross-fit MRR on holdout halves over the same {250, 500, 1000, 2000}.
Apply world, members, gate and validation half unchanged from HC1/HC2.

Disclosure: the weights are derived from validation's aggregate degree
histogram — a query-side statistic for head queries, an answer-side one for
tail queries; no validation label enters row by row and nothing is fit on
validation. This is a third attempt after two failures, run because the
cause it addresses was measured, not because a number was close. If it
fails the gate, degree is not the whole difference between old and new
edges, and no further holdout variant will be run.

## HC3 RESULT (2026-09-08 01:05, box 50209059): FAIL on both; the holdout route is closed for good

Weights (`results/hc3/hc3_weights.json`): effective sample size 269,582 of
484,833 triples, max weight 16.9, none clipped, 0.04% of validation mass in
cells absent from the holdout; the L1 distance between the validation and
holdout (head-degree, tail-degree) histograms went from 0.677 to 0.0007.
Guard sweep by weighted cross-fit: 250 → 0.8429, 500 → 0.8416, 1000 →
0.8374, 2000 → 0.8313; guard 250, 198 local groups.

Student s1, full validation, official Evaluator: **MRR 0.7313** (HC2
0.7266), hits@1 0.6589, hits@10 0.8729. On the held-out half: model alone
0.7186, selection 0.7542, validation-fit combiner 0.7826, holdout-fit
0.7315 → bar 0.7684, **FAIL** (keeps −80% of the gain over selection).
Teacher s0: full validation **0.7220** (HC2 0.7171); half: 0.7042 / 0.7450 /
0.7740 / 0.7221 → bar 0.7595, **FAIL** (−79%). Receipts:
`results/hc3/{dist_s1,wiki_s0}.{json,log}`, `results/hc3/weights_*.npz`.

Reading: degree matching recovers about +0.005 on both channels and no more.
The head-direction weights moved toward the validation fit (linked −0.31 →
−0.22, analogy −0.24 → +0.06 vs +0.12 / +0.24 on validation) but did not
get there. Degree is part of what separates May-2015 edges from August-2015
queries, not most of it. As registered: no further holdout variant. The
combiner fit on validation labels stays reported-not-filed; the ways to a
filing are the selection blend (allowed as tuning; 0.7542 on the half) and a
gradient-free search of the per-relation weights on validation MRR (the
RelEns precedent). A combiner trained inside the model would need to
condition on the pair's neighbourhood, not the relation alone, and to learn
from the training edges that resemble new ones; those are not identified by
degree.
