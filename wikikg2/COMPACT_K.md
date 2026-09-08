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
