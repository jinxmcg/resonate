# CK1: the width curve below k=8 on ogbl-wikikg2 (registered 2026-09-08 02:20, user approved)

The k=8 model is 329M parameters, 97% of it the entity table (2.5M × k²
complex). The measured curve is flat from k=8 to k=12 (0.6945 vs 0.6879–0.6961
at the 200k screen); nothing below k=8 has been run. TripleRE + NodePiece
sits at 0.6866 with 36M and StarGraph + TripleRE at 0.7286 with 93M.

Screen, same recipe as the row-A teachers except k and steps: dense operator
(block size = k²), rev-frac 0.75, RowAdagrad table lr 0.6, neg 4096, seed 0,
**200k steps**, `--eval valid` only (the setting of the paper's curve table,
reference k=8 dense 200k = 0.6945).
* k=6: width 36 complex, ~180M parameters.
* k=4: width 16 complex, ~80M parameters.

Bar, fixed now: a width is carried to the full 800k run (and then members +
reverse + selection blend on validation, one compact row candidate) if its
200k validation MRR is ≥ 0.6945 − 0.020 = 0.6745, i.e. within the 0.02 that
k=8 costs on biokg. Below that the width is recorded as a curve point only.
No test read in CK1.

## CK1 RESULT (2026-09-08 02:23): k=6 PASS 0.6764 (−0.018 vs k=8 at 200k, ~180M); k=4 FAIL 0.6454 (−0.049, ~80M): the width curve bends between k=6 and k=4. Logs: `results/ck1/`.

## CK2 (registered 02:25): the compact screens as ensemble partners of the T=2 student
Validation halves, selection blend (guard 250, seed 0): student alone; k6 / k4 alone;
student + k6, student + k4, student + k6 + k4 as two/three-model score ensembles (uniform
z-mean and the per-relation selected weighting); the same with the nine members. A pairing
is "useful" if its selection-blend held-out MRR beats the student's by ≥ +0.002 with the
same members. Informational at 200k; the full-strength check is CK3.

## CK3 (registered 02:25): the full k=6 model
`model_k6.pt`: the row-A recipe with k=6, block size 36 (dense), 800k steps, seed 0,
`--eval valid`. Then its members on validation (analogy tag k6, reverse tag k6) and the
selection blend for: k6 + nine members (the compact row candidate, ~180M) and student + k6 +
nine members (the two-width ensemble, ~510M). Bars: the compact row is a filing candidate if
its held-out MRR is within 0.02 of the student's row (0.7711); the ensemble is a filing
candidate if it beats the student's row by ≥ +0.005. No test read tonight.

## CK2 RESULT (2026-09-08 02:35): compact screens as ensemble partners — below the bar once the members are in

Selection blend, held-out half (200k screens; `results/ck2/selection.log`):

| set | model-only (uniform) | selected per-relation | head |
|---|---|---|---|
| student | 0.7186 | 0.7186 | 0.476 |
| student + k6 | 0.7199 | 0.7232 | 0.482 |
| student + k4 | 0.7177 | 0.7221 | 0.481 |
| student + k6 + k4 | 0.7207 | 0.7247 | 0.484 |
| student + 9 members | | 0.7711 | 0.571 |
| student + k6 + 9 | | 0.7721 | 0.572 |
| student + k4 + 9 | | 0.7713 | 0.571 |
| student + k6 + k4 + 9 | | 0.7726 | 0.572 |

As bare ensembles the compact widths add +0.004 to +0.006 to the student; once
the nine members are in they add +0.001 to +0.0015, under the +0.002 bar. The
members already carry most of what a second width contributes. Informational
at 200k; CK3 (the 800k k=6) is the full-strength check and is queued.

## CK3 RESULT (2026-09-08 03:12): the full k=6 model

`model_k6.pt` (800k, seed 0): validation MRR 0.6829, hits@1 0.6171, hits@10
0.8123; the k=8 teacher of the same recipe reads 0.7041 (−0.021). Selection
blend, held-out half (`results/ck3/selection.log`):

| row | MRR | head |
|---|---|---|
| k6 alone | 0.6826 | 0.424 |
| k6 + nine members (analogy_k6, holders, cn_aa, linked, cn3_aa, typed, rev_raw_k6, rev_nov_k6) | **0.7529** | 0.545 |
| student + k6 | 0.7241 | 0.484 |
| student + k6 + nine | 0.7725 | 0.572 |
| student + k6 + eleven (both analogy and both reverse pairs) | 0.7750 | 0.578 |

Bars: the compact row is within 0.02 of the student row (0.7711 − 0.7529 =
0.018) → a filing candidate at ~180M (the student row: 329M). The two-width
ensemble reaches +0.004 with all members, under its +0.005 bar; recorded, not
promoted. The k6 row would need its own T=2 distillation to be the "distilled
compact" analogue of the student row; not run tonight.

## CK4 (registered 2026-09-08 04:10): the distilled compact model

`model_dist_k6.pt`: the row-C recipe (T=2 student distilled from the ten
released k=8 teachers, 400k steps, seed 0) with k=6, block size 36; logits
distillation does not depend on the student's width. `--eval valid` only (the
row-C script's `--eval both` is NOT used). Then its members (analogy tag k6d,
reverse tag k6d) and the selection blend: k6d + nine members. Bar: within
0.02 of the student row (0.7711) → it replaces row D as the compact filing
candidate; also reported against the undistilled k6 row (0.7529).

## CK4 RESULT (2026-09-08 04:58): the distilled k=6 student

`model_dist_k6.pt` (T=2 from the ten k=8 teachers, 400k, seed 0): validation
MRR **0.7048**, hits@1 0.6376, hits@10 0.8378 — above the undistilled k=6
(0.6829, +0.022) and level with a k=8 teacher (0.7041) at 180M parameters; the
k=8 student is 0.7190. Selection blend, held-out half
(`results/ck4/selection.log`): k6d alone 0.7046; k6d + nine members
**0.7543** (head 0.542), against 0.7529 for the undistilled k6 row and 0.7711
for the k=8 student row. Bar met (0.7711 − 0.7543 = 0.017 < 0.02): the
distilled k=6 row replaces row D as the compact filing candidate. Note for the
paper: the +0.022 the distillation adds to the model becomes +0.0014 in the
blended row; the members carry most of what the wider or better-trained
model would add.

## CP1 (registered 2026-09-08 05:40, user's idea): compress the trained full-width table by degree tier

Train wide, then narrow the tail: take the released k=8 T=2 student (s1),
group entities by training degree (tiers < 8, 8–63, 64–1023, ≥ 1024), and in
each tier keep only the top principal directions of the trained rows (real
view, centred), i.e. a narrow coefficient vector per entity plus one shared
projection per tier (`compress_wiki.py`). No training. Configurations are
listed by tier widths in complex dimensions, e.g. (8, 16, 36, 64) = 8 for the
tail up to the full 64 for hubs. The full width (64, 64, 64, 64) must
reproduce the student's 0.7190 (sanity). Report validation MRR (official
Evaluator), head/tail, the parameter count and the variance kept per tier.
Bars: within 0.01 of 0.7190 under 100M parameters → a candidate for a short
fine-tune of the narrow coefficients (CP2, to be registered); within 0.02 →
the tiered training screen (adaptive-width table) is worth running. Precedent:
adaptive input embeddings (Baevski & Auli 2019), mixed-dimension embeddings
(Ginart et al. 2019). Validation only.

## CP1 RESULT (2026-09-08 05:50, jinx 1080 Ti): post-hoc tiered compression FAILS

Student s1, tiers by degree < 8 / 8–63 / 64–1023 / ≥ 1024 = 1,674,500 /
808,275 / 16,593 / 1,236 entities (`results/cp1/cp1_student_s1.log`):

| widths (complex, tail → hubs) | params | valid MRR | variance kept per tier |
|---|---|---|---|
| 64, 64, 64, 64 | 320M | 0.7187 (sanity: the student) | 1, 1, 1, 1 |
| 16, 32, 64, 64 | 108M | 0.5876 | 0.49, 0.72, 1, 1 |
| 8, 16, 36, 64 | 54M | 0.4834 | 0.34, 0.49, 0.78, 1 |
| 8, 8, 16, 64 | 40M | 0.4367 | 0.34, 0.33, 0.53, 1 |
| 4, 8, 16, 64 | 27M | 0.4182 | 0.24, 0.33, 0.53, 1 |
| 4, 4, 8, 64 | 20M | 0.3821 | 0.24, 0.23, 0.37, 1 |
| 2, 4, 8, 64 | 14M | 0.3753 | 0.17, 0.23, 0.37, 1 |

No bar is met. The trained tail rows are close to isotropic (16 of 128 real
directions hold a third of the variance), so there is no shared narrow
subspace to project onto after training; the width must be trained narrow,
with the tier's subspace learned jointly (adaptive-width table, a trainer
change, not registered tonight). The uniform-width points bound it: k=6 costs
0.02 at 180M, k=4 costs 0.05 at 80M. Decision for the two wikikg2 entries:
row C (ensemble) and row A (the k=8 student row); the distilled k=6 row stays
a curve point for the paper.

## CP2 (registered 2026-09-08 06:20, user's idea "capture the essence wide, then compact where not needed"): the tiered refit

`resonate_tiered.py` + `train_wiki.py --tiered-from`: the k=8 T=2 student's
operators and temperature are copied and frozen; every entity gets a row of
width by training degree (tiers < 8 / 8–63 / 64–1023 / ≥ 1024), a coefficient
vector per entity plus one shared projection and offset per tier, initialised
from CP1's per-tier PCA. Only the coefficients (RowAdagrad, lr 0.6) and the
four projections/offsets (Adam) train, on training edges, with the standard
loss and the student as T=2 distillation teacher, 200k steps, seed 0, neg
4096, `--eval valid`. Configurations: widths (8, 16, 36, 64) = 54M table and
(16, 32, 64, 64) = 108M. Bars as CP1: within 0.02 of the student (0.7190) →
worth the full run and its members; within 0.01 → replaces row A as the
single-model entry. Validation only.

## CP2 RESULT (2026-09-08 10:00): the tiered refit sits on the uniform-width curve

Refit from the k=8 student (operators frozen, T=2 distillation from the
student, 200k steps; `results/cp2/`): widths (8, 16, 36, 64), 54M table + 8.8M
operators ≈ 63M: PCA init 0.4834 → probes 0.6296 / 0.6369 / 0.6415 → **valid
0.6424**. Widths (16, 32, 64, 64), 108M + 8.8M: **valid 0.6792**. Reference: the student 0.7190; the uniform-width
points trained from scratch, k=4 (80M) 0.6454 and k=6 (180M) 0.6829. The
tiered rows land on that same curve: width by degree buys nothing over width
alone, and neither meets the 0.02 bar. Reading: a per-entity row costs its
width whatever the entity's degree; the parameter count on this graph falls
only by sharing structure across entities (anchor/hash encoders), which is
the next research item. Rows C and A remain the two wikikg2 entries.

CP2 diagnostic (per-tier, `tier_diag.py`): the loss is where a TAIL entity is
the ANSWER — answer tier < 8: 0.4658 (student) → 0.3283 (54M) / 0.4019 (108M);
answer tier 8–63: 0.7016 → 0.6156 / 0.6491; hub answers unchanged (0.99);
tail entities as queries lose far less (0.9195 → 0.8449 / 0.8732). Narrow
rows fail as candidates: 1.67M rows in ONE shared 16-real subspace cannot be
told apart among 500 decoys that are mostly tail entities too. The width is
not the defect; the single subspace per tier is.

## CP3 (registered 2026-09-08 10:40, user's reading: "the range is huge"): many subspaces per tier

As CP2 (narrow coefficients per entity, operators frozen, refit with the
student as T=2 teacher, 200k steps, seed 0), but each tier gets K learned
subspaces instead of one: entities are assigned to a subspace by k-means on
their wide rows (K = 256 for the two tail tiers, 16 for the 64–1023 tier, 1 for
the hubs), each subspace has its own projection and offset initialised by
PCA of its cluster, and the assignment is fixed. Parameters: coefficients as
CP2 + K·(d·2M + 2M) per tier (≈ 0.5M for K = 256, d = 16). Configuration
(8, 16, 36, 64) as CP2's 54M. Bars as CP2 (within 0.02 of 0.7190 → worth the
full run; within 0.01 → replaces row A). The per-tier diagnostic is reported
again. Validation only.

CP3 init check (jinx, 2026-09-08 11:00): with K = 256 / 256 / 16 / 1 subspaces
and widths (8, 16, 36, 64), the k-means + per-cluster PCA initialisation alone,
no training, reads **0.6831** on validation (55.8M table parameters), against
0.4834 for one subspace per tier (CP1) and 0.6424 after CP2's 200k-step refit.

## CP3b (registered 11:05): init-only sweep over K and widths
Same construction, `--steps 0`, validation MRR and parameter count for
K ∈ {256, 1024, 4096} on the two tail tiers (16 / 1 above) × widths
(8, 16, 36, 64), (4, 8, 36, 64), (4, 4, 16, 64). Informational: maps the
size–accuracy frontier of clustered narrow rows before any refit; the refit
(CP3) is run on the registered K = 256 point, and the best CP3b point may be
refit afterwards under the same bars.

## CP3b RESULT (2026-09-08 11:10): the frontier of clustered narrow rows, no training

`results/cp3b/summary.log` (init only; the student is 0.7190 at 320M):

| K (two tail tiers) | widths | table params | valid MRR |
|---|---|---|---|
| 256 | 8, 16, 36, 64 | 55.8M | 0.6866 |
| 256 | 4, 8, 36, 64 | 28.7M | 0.6652 |
| 256 | 4, 4, 16, 64 | 21.2M | 0.6426 |
| 1024 | 8, 16, 36, 64 | 60.7M | 0.7020 |
| 1024 | 4, 8, 36, 64 | 31.3M | 0.6875 |
| 1024 | 4, 4, 16, 64 | 23.0M | 0.6711 |
| 4096 | 8, 16, 36, 64 | 80.4M | **0.7109** |
| 4096 | 4, 8, 36, 64 | 41.5M | **0.7029** |
| 4096 | 4, 4, 16, 64 | 30.1M | 0.6887 |

The number of subspaces is the lever, not the width: 256 → 1024 → 4096
gains +0.015 and +0.009 at fixed widths, while halving the tail width costs
0.008 to 0.016. K = 4096 with (8, 16, 36, 64) is within 0.01 of the student
(CP3's "replaces row A" bar) with a quarter of the table; K = 4096 with
(4, 8, 36, 64) is within 0.02 at an eighth. Operators add 8.8M to each.

## CP3c (registered 11:15): the two K = 4096 points as models
Rebuild and keep both (`model_cp3_k4096_w8.pt`, `model_cp3_k4096_w4.pt`,
same seed → same k-means), give each its members on validation (analogy from
its own table, reverse through the copied operators, the shared members) and
run the selection blend: compact rows "cp3 + nine". Gentle refit of the 41M
point: 50k steps, table lr 0.1, operators frozen, student as T=2 teacher
(`model_cp3_k4096_w4_refit.pt`), reported against its init. Per-tier
diagnostic for all three. Bars: a compact row within 0.02 of the student row
(0.7711 standard / 0.7734 rich) is the second wikikg2 entry in place of row A,
the smaller of the two that meets it preferred. Validation only.

## CP3d (registered 2026-09-08 11:40, user: "still coarse"): finer tiers for the tail
Degree histogram (entities / share of validation answers): 1–2 20k / 0.5%,
3–4 236k / 2.0%, 5–7 1.42M / 32.2%, 8–15 671k / 20.0%, 16–63 137k / 10.0%,
64–1023 17k / 13.9%, ≥ 1024 1.2k / 21.3%. Init-only sweep (as CP3b) with
cutoffs (5, 8, 16, 64, 1024), K per tier scaled to its size (512, 4096, 2048,
512, 16, 1), widths (2, 4, 8, 16, 36, 64) ≈ 40M and (4, 8, 16, 32, 36, 64) ≈ 65M,
and (2, 4, 4, 8, 36, 64) ≈ 30M; plus the same three with K doubled where the
tier allows (1024, 4096, 4096, 1024, 16, 1). Reported against CP3b's points
of equal size. Informational; the best point joins CP3c's member/selection
check if it beats CP3b at equal size.

## CP3d RESULT (2026-09-08 12:30): finer tiers gain at the small end only

Init-only (`results/cp3d/summary.log`), cutoffs (5, 8, 16, 64, 1024):

| K (first four tiers) | widths | table params | valid MRR | coarse point of equal size (CP3b) |
|---|---|---|---|---|
| 512, 4096, 2048, 512 | 2, 4, 4, 8, 36, 64 | 30.0M | 0.6960 | 0.6887 (30.1M) |
| 1024, 4096, 4096, 1024 | 2, 4, 4, 8, 36, 64 | 33.8M | **0.6991** | |
| 512 … | 2, 4, 8, 16, 36, 64 | 40.7M | 0.7007 | 0.7029 (41.5M) |
| 1024 … | 2, 4, 8, 16, 36, 64 | 47.6M | 0.7022 | |
| 512 … | 4, 8, 16, 32, 36, 64 | 78.9M | 0.7084 | 0.7109 (80.4M) |
| 1024 … | 4, 8, 16, 32, 36, 64 | 92.4M | 0.7091 | |

At ~30M the finer split of the tail (degree 1–4 at width 2, 5–7 at 4) is
worth +0.007 to +0.010; from 40M up the schedules are equal and the curve
saturates near 0.71 by 80M. Best small point: 33.8M table + 8.8M operators
≈ 43M at 0.6991, within 0.02 of the student (0.7190).

## CP3 RESULT (2026-09-08 12:45): the K = 256 refit adds +0.012 over 200k steps
`model_cp3_k256.pt` (widths 8, 16, 36, 64, K = 256/256/16/1, 55.8M table):
init 0.6866 → probes 0.6586 (50k, the high-rate dip) / 0.6878 / 0.6956 →
**valid 0.6983**. CP3c's gentle refit (K = 4096, widths 4, 8, 36, 64, 50k steps,
table lr 0.1) went 0.7029 → **0.7044**: too short to matter. The refit works
given a full schedule; the dip is the high early rate.

## CP3e (registered 12:47): full refit of the two K = 4096 models
As CP3 (200k steps, table lr 0.6 cosine, operators frozen, student as T=2
teacher) on `model_cp3_k4096_w4.pt` (41.5M table) and `model_cp3_k4096_w8.pt`
(80.4M). Bars unchanged; the refit models then get members and the selection
blend as in CP3c. Validation only.

## CP3f (registered 2026-09-08 13:05): train narrow, with the wide model's clusters only
As CP3e on the 41M configuration (K = 4096, widths 4, 8, 36, 64, operators
frozen, student as T=2 teacher, 200k steps), but the narrow rows and the
cluster subspaces start at random (`--tiered-random-init`); only the cluster
assignment is taken from the wide model. Answers whether training narrow
reaches the compress-and-refit number (CP3e's 41M result) in the same
budget. Informational; validation only.

## CP3c / CP3e RESULTS (2026-09-08 13:30): the compact rows with the evidence layer

Selection blend, held-out half, nine members (own analogy, holders, cn_aa,
linked, cn3_aa, typed, own rev_raw / rev_nov); `results/cp3c/selection{2,3}.log`,
`results/cp3e/selection.log`:

| model | table | model alone (valid) | row, standard | row, rich |
|---|---|---|---|---|
| k=8 T=2 student (reference) | 320M | 0.7190 | 0.7711 | 0.7734 |
| K=4096, widths 8/16/36/64, init only | 80.4M | 0.7109 | 0.7622 | 0.7663 |
| K=4096, widths 4/8/36/64, init only | 41.5M | 0.7029 | 0.7548 | 0.7594 |
| same, gentle refit (50k, lr 0.1) | 41.5M | 0.7044 | 0.7601 | 0.7639 |
| same, full refit (200k, lr 0.6) | 41.5M | **0.7074** | 0.7629 | **0.7667** |

Both K = 4096 models meet the CP3c bar (within 0.02 of the student row): the
fully refit 41M-table model (≈ 50M with operators) is 0.0067 under the
student row with the rich selection, and above the un-refit 80M model.
Reading: the refit is worth +0.0045 on the model and +0.007 on the row; the
evidence layer narrows the gap to the wide model from 0.012 (alone) to
0.007 (row). The 80M full refit and the per-tier diagnostic are pending.
Decision candidate for the second wikikg2 entry: the refit 41M model + nine
members, rich selection (≈ 50M parameters), pending the 80M number.

## CP3e RESULT (2026-09-08 14:05): full refits of the two K = 4096 models, and the decision

| model | table | total | alone | row std | row rich |
|---|---|---|---|---|---|
| student k=8 (reference) | 320M | 329M | 0.7190 | 0.7711 | 0.7734 |
| widths 8/16/36/64, full refit (`model_cp3_k4096_w8_full.pt`) | 80.4M | 89M | 0.7105 (init 0.7109; the deeper high-rate dip, 0.597 at 50k, did not fully recover) | 0.7645 | **0.7684** |
| widths 4/8/36/64, full refit (`model_cp3_k4096_w4_full.pt`) | 41.5M | 50M | 0.7074 (init 0.7029) | 0.7629 | **0.7667** |

Per-tier diagnostic (`results/cp3c/tier_diag5.log`): the clustered models
keep tail answers at 0.43–0.45 (CP2's single subspace: 0.33; student 0.47) and
lose nothing above 64 edges; the refit moves tail answers up (0.4262 →
0.4473 for the 41M table) at a small cost in the 8–63 group. Decision: the
second wikikg2 entry is the 50M model (refit 41M table + operators) with its
nine members and the rich selection, 0.7667 held-out, 0.0067 under the 329M
student's row; the 89M model buys +0.0017 for +40M and is kept as a curve
point. Both remain "validation only" until the user's test-read go.
