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
