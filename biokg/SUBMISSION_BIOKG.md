# ogbl-biokg filings (2026-09-09)

Two entries, the Pareto frontier of the sparse ladder: the best MRR (C) and the
best MRR per parameter (C′). Rows A and B are not filed — B is dominated by C′
on both axes the board reports (lower MRR at 2.84× the parameters) and A is a
rung below both; they remain the development record in `README.md` and the paper.

Both entries are by allowed means: validation is used only to select among a
fixed candidate list of per-(relation, direction) weight patterns
(hyperparameter tuning, not gradient search), no gradient touches validation
labels, and each row takes one test read per seed through the official OGB
Evaluator, ten seeds, no read repeated. Team: Cristian Malaia. Code and
checkpoints: https://github.com/jinxmcg/resonate (release `v2.0-two-boards` for
the row-A models and T=2 students, `v2.1-compact` for the ten compact
checkpoints and their SHA256SUMS).

**The form:** https://forms.gle/7PB7375i5P1rHgng6 — one submission per entry.
Results are posted after a validity check, typically within a week.

Fields that are the same on both submissions:

| form field | value |
|---|---|
| dataset | ogbl-biokg |
| OGB version | 1.3.6 (`ogb>=1.3.6` in `pyproject.toml`, 1.3.6 pinned in `uv.lock`) |
| official / unofficial | **official** — submitted by the authors of the method |
| external data | none (no pre-trained models, no external datasets; the retrieval features read the ogbl-biokg training split only) |
| ensemble | no |
| runs | 10 random seeds, `torch.mean` and unbiased `torch.std` over the ten |
| team | Cristian Malaia |
| contact | cristian.malaia@gmail.com |
| code | https://github.com/jinxmcg/resonate (README carries the reproduction commands; `VALIDATE.md` carries them with their measured cost and output) |
| paper | Beyond Link Prediction: Compact Knowledge Representations for Prediction, Retrieval, and Direct Access — `paper/resonate.pdf`, linked **at a pinned commit SHA**, not at `main` (see "Open before the form goes in"); not on arXiv by choice |
| hardware | 1× RTX 5090 (32 GB, rented) for training; 1× GTX 1080 Ti for the compression, retrieval features, blend and reads |

## Entry C — ResonatE distilled + retrieval

| form field | value |
|---|---|
| method name | ResonatE (distilled) + retrieval |
| test MRR | **0.8528 ± 0.0002** (10 seeds, one read each; hits@1 0.7964, hits@3 0.8933, hits@10 0.9539) |
| validation MRR | 0.8537 ± 0.0002 (the frozen selection, in-sample); held-out estimate 0.8532 ± 0.0003 |
| parameters | **27,124,129** (table 27,006,624 + relation operators 117,505) |
| training cost | ~4.5 min per seed for the ten source models, ~9 min per seed for the students, on one RTX 5090; retrieval features ~40 min per seed on CPU plus 88 min once for the shared Jaccard pair |

Per seed: 0.8531 0.8527 0.8526 0.8529 0.8527 0.8528 0.8529 0.8527 0.8531 0.8530.
The distilled model alone reads 0.8321 ± 0.0003 on validation.

**Description (for the form).** A ResonatE model — a unit-norm complex entity
table with per-relation block operators, unitary at initialisation, trained with
row-sparse gradients and row-wise Adagrad (k=12, 4×4 blocks) — distilled at
temperature 2 from ten independently seeded models of the same shape and
parameter count. Its scores are combined with four label-free retrieval features
read from the training graph only (two analogy members and two Jaccard members)
by a per-(relation, direction) selection among a fixed list of weight patterns,
the selection made on validation and then frozen and applied once to test.

## Entry C′ — ResonatE compact (degree-tiered table) + retrieval

| form field | value |
|---|---|
| method name | ResonatE compact (degree-tiered table) + retrieval |
| test MRR | **0.8468 ± 0.0003** (10 seeds, one read each; hits@1 0.7888, hits@3 0.8880, hits@10 0.9513) |
| validation MRR | 0.8481 ± 0.0003 (the frozen selection, in-sample); held-out estimate 0.8478 ± 0.0003 |
| parameters | **9,555,497** (table 9,437,992 + relation operators 117,505) |
| training cost | entry C's, plus 10 s per seed for the compression on a GTX 1080 Ti, which also took the members, the blend and the ten test reads |

Per seed: 0.8466 0.8470 0.8466 0.8473 0.8466 0.8460 0.8467 0.8469 0.8471 0.8467.

**Description (for the form).** Entry C's distilled model with its entity table
replaced *after training* by degree-tiered coefficients over per-tier subspace
banks: entities are split into six tiers by training degree (cutoffs 5, 8, 32,
128, 1024), each tier's rows are clustered by k-means (K = 256, 256, 256, 32, 1,
1), and each cluster gets a PCA basis of width 2, 4, 8, 48, 144, 144, so a row is
stored as coefficients in its cluster's subspace and reconstructed as a matmul.
No training follows the compression; the relation operators are copied and
frozen. The table falls from 27,006,624 to 9,437,992 parameters, 64.8% smaller
overall, for 0.0061 test MRR. The same four retrieval features and the same
frozen selection blend as entry C are applied on top.

## Tuned hyperparameters, with the selected values

Everything below was selected on the validation split. Values marked * differ
from the code's defaults; the rest are the defaults, listed so the setting is
complete. Reproduced exactly by `scripts/run_campaign_sparse.sh`,
`scripts/run_distill_sparse.sh` and `scripts/run_cpb3_campaign.sh`.

**Training (both entries share it)** — `train_biokg_comp.py`:
`--steps 50000`*, `--batch 2048`, `--neg 4096`, `--k 12`, `--block-size 4`*,
`--shell sparse`* (real-view entity table, row-sparse gradients, row-wise
Adagrad), `--table-lr 0.3`*, `--table-dtype fp32`, `--lr 5e-3` with
`--sched cosine` (Adam on everything but the table), `--lam 0.1`,
`--direction-sampling uniform`, `--mining-mode none`, no auxiliary terms
(`--aux-rp 0`, `--n3 0`, `--compose 0`), seeds 0–9.

**Distillation** — the same call plus `--distill` over the ten row-A
checkpoints, `--distill-w 1.0`, `--distill-T 2.0`*. T ∈ {1, 2} was the search;
T=1 gives 0.8509 ± 0.0003 on test and is reported as the ablation, T=2 was
selected on validation.

**Compression (entry C′ only)** — `compress_biokg.py`, degree cutoffs
(5, 8, 32, 128, 1024), per-tier subspace widths (2, 4, 8, 48, 144, 144),
per-tier cluster counts K = (256, 256, 256, 32, 1, 1). Selected on validation
from the CP-B1 width curve; the three-point check on the distilled model
(9.56M → 0.8176 > 8.92M → 0.8151 > 11.75M → 0.8142) confirmed the schedule
transfers, and it was frozen before the blend was computed.

**Blend (both entries)** — five members (the model, two analogy features, two
Jaccard features), `--norm z`, per-(relation, direction) selection among a fixed
candidate list of weight patterns with the global pattern `soft20`, groups below
`--min-rows 4000` falling back to their family or the global pattern. The
held-out estimate uses `--min-rows 2000` on validation halves with `--seed 0`.
Weights are fitted on the full validation split, frozen, and applied once to
test.

## What the two entries claim together

The compression costs 0.0143 MRR on the model's own scores but only 0.0061 after
the blend: the analogy and Jaccard features read the training graph rather than
the entity table, so they absorb 57% of the loss. The compact row's seed spread
(±0.0003) is within a whisker of the uncompressed row's (±0.0002), so the
compression adds essentially no variance of its own.

## Reproduction

```
scripts/fetch_checkpoints.sh sparse     # dist_T2_s{0..9}.pt, entry C
scripts/fetch_checkpoints.sh compact    # tiered_T2_s{0..9}.pt, entry C′
uv run python verify_compact.py --device cuda    # re-scores all ten compact tables
python scripts/summarize.py             # entry C's row from results/sparse/
python scripts/summarize_cpb3.py        # entry C′'s row from results/cpb3/
```

`verify_compact.py` rebuilds each tiered table, re-scores it with the official
Evaluator against the value logged when it was built, and asserts the 9,555,497
parameter count. All ten reproduce with max |d| = 0.00004 on both the GTX 1080 Ti
that built them (torch 2.6.0+cu124) and a rented RTX 5090 (torch 2.11.0+cu128),
the per-seed deviations identical to five decimals.

## Disclosures

* **Validation is optimistic by a known amount.** Blend weights are frozen on the
  full validation split, so the validation numbers above are in-sample. The
  held-out estimates (same procedure fit on one random half of validation, scored
  on the other) are 0.8532 for C and 0.8478 for C′. Held-out minus test is
  +0.0004 for C and +0.0011 for C′ on average, in the same direction on all ten
  seeds.
* **The compression is device-deterministic but not device-portable.**
  `compress_biokg.py` seeds its k-means from the device RNG, so re-deriving a
  compact table on a different GPU yields a different (equally good) table, not
  the released one. The ten checkpoints therefore ship as files. The released
  *files* are device-independent — rebuilding a row is `coef @ P + mu` — which is
  what the cross-machine check establishes.
* **Reads spent.** Ten per row, one per seed, all reported. The seed-0 read from
  the earlier CP-B2 chain (0.8433, the compact *undistilled* model) belongs to a
  different model and is reported in `COMPACT_B.md`, not pooled here.
* **Not filed.** Row A, the single 27.1M model, 0.8158 ± 0.0006; row B, that
  model with the same retrieval features, 0.8463 ± 0.0004. Both are dominated by
  the filed rows and are kept as the development record. Also not filed: the
  first (dense-Adam) ladder in `results/dense/`, superseded by this one.
* **CP-B4 ran and was dropped; C′ as filed stands.** The refit of the compact
  table's coefficients (lr 5e-4, everything else frozen) cleared its
  pre-registered bar on seed 0 — 0.8539 held-out on validation, above entry C —
  and earned the ten-seed campaign. The campaign landed and the row did not hold:
  under the registered rule ("reported as C″ only if the mean beats 0.8468 by
  more than the pooled seed spread; otherwise C′ stands as filed and the refit is
  a validation curiosity"), C′ stands. **The individual refit models were good;
  the combination was what failed** — each refit table improved on its own scores
  much as seed 0 promised, but the blended row did not follow. Nothing about the
  filed entry changes: the ten checkpoints, the ten test reads and the 0.8468
  above are the init-only compact tables, untouched by CP-B4. The campaign's
  receipts and the reading of why the blend failed belong in `COMPACT_B.md`.

## Board position

Against the ogbl-biokg leaderboard as fetched 2026-09-09 (12 entries, unchanged
since 2026-09-04):

| # | method | test MRR | params |
|---|---|---|---|
| 4 | AutoBLM-KGBench | 0.8536 | 192,047,104 |
| **5** | **C. ResonatE distilled + retrieval** | **0.8528** | **27,124,129** |
| 6 | ComplEx-RP (1000dim) | 0.8492 | 187,750,000 |
| **7** | **C′. ResonatE compact + retrieval** | **0.8468** | **9,555,497** |
| 8 | TripleRE | 0.8348 | 469,630,002 |

C is the 4th single model. The smallest entry currently listed is AutoSF at
93,824,000 parameters: C is 3.5× smaller than that, C′ is 9.8× smaller, and C′
is 49× smaller than TripleRE, which it beats by 0.0120.

## Open before the form goes in

* **The paper link is the repository PDF, not arXiv** (decided 2026-09-09: the
  preprint still needs changes and will not be posted first). OGB prefers an
  arXiv link and requires a technical report when the method has original
  components, so this is the one field where the submission may draw a query.
  Two things make it hold up: the PDF was **rebuilt from the current `.tex`
  before filing** — the version committed on 7 September was 18 pages and
  contained no mention of the compact row, so it described neither entry C′ nor
  its construction; the current one is 20 pages and carries
  §"Compressing the entity table after training", the C′ row and the CP-B4
  reading. And the link given on the form should be **pinned to a commit**,
  `https://github.com/jinxmcg/resonate/blob/2689e73a745cd59678db10b77ae0c14c66a16a30/paper/resonate.pdf`
  (the commit carrying the PDF rebuilt on 9 September with the comparison fix and the Acknowledgements section), not
  `main` — otherwise later edits to the preprint silently change the report the
  filed entries point at. That SHA is the one to paste on the form.
* **CP-B4.** Its ten-seed campaign would replace entry C′ at the refit numbers
  (seed 0: 0.8539 held-out on validation, above entry C) at the same 9,555,497
  parameters. Confirm whether it finished before filing — if it did and it holds,
  the honest filing is one entry, not two.
* **The repository status banners.** `README.md` and `VALIDATE.md` both open with
  "nothing has been filed for BioKG, WikiKG2, or STaRK-Prime", and the
  proposed-entries table still lists four biokg rows. Update them to two filed
  rows *after* the form is submitted, not before.

## Records

`biokg/COMPACT_B.md` (CP-B1 through CP-B4, every pre-registration and bar in the
order written), receipts in `biokg/results/sparse/` and `biokg/results/cpb3/`,
reproduction recipe and both verification runs in `VALIDATE.md`.
