# TR1 proposal: test reads for the allowed rows (DRAFT for the user's approval; nothing here has run)

User's decision (2026-09-08): at most two wikikg2 entries — C (the score entry) and the compact
model E (the architecture entry). Reads: one for C, one for E (seed 0 exists).

Written 2026-09-08 04:00 while the user slept. No test cache exists on the box
for any of these rows; building them is part of the read and waits for the go.

Rows, all "selection blend on full validation, frozen, applied once to test"
(`blend_wiki.py freeze --test`, guard 250, the same members as measured):

| row | validation held-out | test estimate | params | reads |
|---|---|---|---|---|
| A. T=2 student s1 + nine members, rich selection | 0.7734 | ~0.715 | 329M | superseded by E for the two-entry plan; kept as reference |
| E. compact: K=4096 clustered narrow rows (widths 4/8/36/64) refit from the student, operators frozen, + nine members, rich selection (`model_cp3_k4096_w4_full.pt`) | 0.7667 | ~0.70 | 50M | 1 (seed 0) |
| B. seven-student ensemble + members + reverse | 0.7834 | ~0.727 | 2.3B | 1 |
| C. ten-teacher ensemble + members + reverse | 0.7909 | ~0.735 | 3.29B | 1 |
| D. distilled k=6 (T=2, CK4) + its nine members (compact) | 0.7543 | ~0.70 | 180M | 1 (seed 0 only exists) |

Each read: build the row's test caches (ensemble scores and reverse members
via `ens_cache.py --split test`; per-model `cache_wiki.py`, `reverse_wiki.py`,
`retrieval_wiki.py --split test`; `typed_paths.py --split test`; the shared
`cn_wiki`/`cn3_wiki` test caches), freeze the selection on full validation,
apply once, record the official Evaluator MRR with hits@1/3/10 in
`results/tr1/`. SEL1's rich family is used for row A (met its bar, +0.0023, stable over three half-seeds); rows B–D use the standard family.

Disclosures to carry into README/paper: selection guard 250 chosen on
validation halves (the earlier test-mix-informed guard is not used); the
reverse members are functions of the frozen models; ensembles are marked as
ensembles with summed parameters; the validation-fit learned combiner rows
stay reported-not-filed.

## TR1 APPROVED (2026-09-08 15:25; the user: "ok let's do prepare the submissions today")

Rows read, each once through the official Evaluator after its selection is frozen on
full validation:
* **C**: `ens10t` (score average of the ten released teachers) + analogy_d1_t3 +
  analogy_s0_t3 + holders + cn_aa + linked + cn3_aa + typed + rev_raw_ens10t +
  rev_nov_ens10t; standard selection family, guard 250 (`scripts/read_c.sh`, box 1).
  One read. The leave-one-out spread is optional and separate.
* **E, seven seeds** (the released T=2 students s1, s4–s9): each compressed with its
  own k-means clusters (K = 4096/4096/16/1, widths 4/8/36/64, per-cluster PCA), refit
  200k steps with that student as T=2 teacher and its operators frozen
  (`model_cp3s<s>_w4_full.pt`; seed 1 = the existing `model_cp3_k4096_w4_full.pt`),
  then its nine members (own analogy, holders, cn_aa, linked, cn3_aa, typed, own
  rev_raw / rev_nov); rich selection family, guard 250 (`scripts/e_seed.sh`; seeds
  1, 4, 5, 6 on box 1, 7, 8, 9 on box 2). Seven reads; reported as mean ± std.
Test caches are built inside these scripts and deleted per seed afterwards; no
other test access. Receipts: `results/tr1/`.

### C READ (2026-09-08 13:46 box time): test MRR **0.7320**
hits@1 0.6656, hits@3 0.7641, hits@10 0.8643; tail 0.9691, head 0.4949;
frozen selection in-sample validation 0.7914 (held-out estimate 0.7909; gap to
test 0.059). Global pattern chosen: soft20. On the board of 2026-09-04 this is
2nd, behind RelEns 0.7392 and ahead of StarGraph + TripleRE + Text 0.7305.
Receipts: `results/tr1/C.{json,log}`, `results/tr1/frozen_C.npz`.

### E READS (one per seed; rich selection frozen on full validation)
| seed | test MRR | hits@1 | hits@3 | hits@10 | tail / head | in-sample valid |
|---|---|---|---|---|---|---|
| s1 | **0.7116** | 0.6450 | 0.7433 | 0.8429 | 0.9621 / 0.4610 | 0.7675 |
| s4 | **0.7090** | 0.6436 | 0.7390 | 0.8379 | 0.9560 / 0.4619 | 0.7681 |
| s6 | **0.7085** | 0.6433 | 0.7383 | 0.8369 | 0.9569 / 0.4600 | 0.7651 |
| s5 | **0.7105** | 0.6454 | 0.7405 | 0.8391 | 0.9576 / 0.4634 | 0.7665 |
| s8 | **0.7087** | 0.6437 | 0.7388 | 0.8367 | 0.9570 / 0.4604 | 0.7686 |
| s9 | **0.7107** | 0.6455 | 0.7408 | 0.8391 | 0.9585 / 0.4628 | 0.7657 |
| s7 | **0.7083** | 0.6434 | 0.7388 | 0.8355 | 0.9567 / 0.4599 | 0.7661 |

**E, seven seeds: test MRR 0.7096 ± 0.0013** (hits@1 0.6443 ± 0.0010, hits@3
0.7399 ± 0.0018, hits@10 0.8383 ± 0.0024); frozen-selection validation
0.7668 ± 0.0013 in-sample (held-out estimate on seed 1: 0.7667). Exact
parameter count from the checkpoint: 50,244,249 real parameters (coefficients
27,681,304 + projections 12,746,752 + offsets 1,050,752 + operators 8,765,440
+ temperature). Receipts: `results/tr1/E_s*.{json,log}`, `frozen_E_s*.npz`;
checkpoints `model_cp3s{1,4,5,6,7,8,9}_w4_full.pt` on release v2.0-two-boards
(SHA256SUMS_compact). On the 2026-09-04 board this is 8th, between StarGraph +
TripleRE (2022) 0.7201 and CompoundE3D 0.7006, and the best entry under 90M.

## TR2 (registered 2026-09-08 evening, pending the user's go): the ten-seed requirement

OGB: "average and unbiased standard deviation must be taken over 10 different
random seeds". E has seven released students; C is one read.
* **E, seeds 0, 2, 3**: retrain the lost students with the row-C recipe
  (`train_wiki.py --distill <ten teachers> --distill-T 2.0`, 400k steps, seed s,
  `--eval valid`), then exactly the E procedure (`e_seed.sh`): compress with the
  student's own clusters, refit 200k, nine members, rich selection frozen on full
  validation, one read each. Reported together with the seven as mean ± std over
  ten.
* **C, ten leave-one-out reads**: for i in 0…9 the score average of the nine
  teachers without seed i (and their averaged reverse members), the same seven
  other members, standard selection frozen on full validation, one read each;
  reported as mean ± std over the ten, next to the full ten-teacher read 0.7320
  (the protocol already used for the unfiled ensemble row). Thirteen reads in
  total; no other test access.
