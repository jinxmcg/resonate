# CP-B1: the tiered-table width curve on ogbl-biokg (registered 2026-09-08 12:50, user's idea)

The wikikg2 compression line (`../wikikg2/COMPACT_K.md`, CP1–CP3e) ends with
clustered narrow rows: entities are grouped by training degree, each tier gets
K learned subspaces (k-means on the trained rows + per-cluster PCA), and each
entity keeps only a narrow coefficient vector. On wikikg2 that reached 0.7109
against the student's 0.7190 at a quarter of the table, and the reading was
"the number of subspaces is the lever, not the width".

biokg is the opposite shape of graph: **93,773 entities against wikikg2's
2.5M**, and far less skewed degrees (median 22, mean 102). Both facts matter,
and they pull in opposite directions:

* a per-tier subspace bank costs `K · (d + 1) · 2M` whatever the graph, but
  here it is amortised over 93,773 rows instead of 2.5M, so K is roughly 27x
  more expensive per entity;
* the degree tail is much thinner in answers — the 24.3% of entities with
  degree < 5 are only 0.5% of validation answers.

Registered: an init-only sweep (no training) over tier cutoffs, widths and K,
validation only, official OGB Evaluator, against the released single model
`checkpoints/sparse_s0.pt` (k=12, 4x4 blocks, 27,124,129 parameters). Bar as
CP1/CP2: **within 0.02 of the single** is a curve point worth carrying;
**within 0.01** would make it a candidate to replace the single-model row.
Target size 8–12M. No test read.

Port: `resonate_tiered_biokg.py` re-parents `resonate_tiered.TieredTableResonatE`
off `resonate.ResonatE` (biokg's model class, complex `E`, typed negatives)
instead of the wikikg2 sparse shell — `E` becomes a property over `table()`,
`rows()`/`embed()` go through the tiered reconstruction, operators and
temperature are copied from the wide model and frozen. The table mathematics
and `init_from_wide` (k-means + per-cluster PCA on the real view) are the
wikikg2 file's, unchanged. `compress_biokg.py` is the sweep driver.

## Tiers

Degree histogram on the training graph (both directions), cutoffs
(5, 8, 32, 128, 1024), and each tier's share of the 325,772 validation
answers (both directions):

| tier | degree | entities | share | validation answers |
|---|---|---|---|---|
| 0 | 1–4 | 22,766 | 24.3% | 0.5% |
| 1 | 5–7 | 3,666 | 3.9% | 0.4% |
| 2 | 8–31 | 28,676 | 30.6% | 5.7% |
| 3 | 32–127 | 29,285 | 31.2% | 19.3% |
| 4 | 128–1023 | 8,038 | 8.6% | 29.9% |
| 5 | ≥ 1024 | 1,342 | 1.4% | 44.3% |

Against wikikg2, where 1.67M of 2.5M entities sit below degree 8 and carry 32%
of the answers, biokg's mass is in the middle and its answers are in the top
two tiers.

## CP-B1 RESULT (2026-09-08 13:10, jinx 1080 Ti): 27.1M → 9.6M for 0.010

Init only, no refit. `results/cpb1/sweep.log` (round 1), `sweep2.log` (round 2).
Reference: the wide model re-scored through the same path reads **0.8155** on
validation (test 0.8158 ± 0.0006 over the ten seeds; this is seed 0). Widths
are complex dimensions per tier, so 144 = the full row; K is subspaces per
tier. "Total" adds the 117,505 frozen operator parameters.

Sanity: widths (144 ×6), K = 1 reproduces the wide model exactly — 0.8155,
hits@1 0.7453, hits@10 0.9414, variance kept 1.0 in every tier.

**Round 1 — K at fixed widths (2, 4, 8, 32, 96, 144):**

| K per tier | table | total | valid MRR |
|---|---|---|---|
| 1, 1, 1, 1, 1, 1 | 4.55M | 4.67M | 0.7180 |
| 16, 16, 16, 8, 4, 1 | 4.98M | 5.10M | 0.7696 |
| 64, 64, 64, 16, 4, 1 | 5.56M | 5.68M | 0.7824 |
| 256, 256, 256, 64, 8, 1 | 8.39M | 8.51M | 0.8006 |
| 1024, 1024, 1024, 256, 16, 1 | 19.29M | 19.41M | 0.8126 |

K is still the lever it was on wikikg2 — one subspace per tier loses 0.098 and
clustering buys most of it back — but here it prices itself out. 64 → 256
costs +2.8M for +0.018; 256 → 1024 costs +10.9M for +0.012. The knee is at
K ≈ 256. wikikg2's winning point, K = 4096, is not reachable at all: at these
widths it would cost 56.7M, more than twice the 27.0M table it compresses.

**Round 2 — where the width goes (tiers 4–5 at full 144):**

| widths | K per tier | table | total | valid MRR |
|---|---|---|---|---|
| 1, 2, 4, 24, 144, 144 | 256, 256, 256, 64, 1, 1 | 6.72M | 6.84M | 0.7923 |
| 2, 4, 8, 16, 144, 144 | 256, 256, 256, 64, 1, 1 | 7.28M | 7.40M | 0.7869 |
| 1, 2, 4, 32, 144, 144 | 256, 256, 256, 64, 1, 1 | 7.48M | **7.60M** | **0.7973** |
| 2, 4, 8, 24, 144, 144 | 256, 256, 256, 64, 1, 1 | 8.04M | 8.16M | 0.7974 |
| 2, 4, 8, 32, 144, 144 | 256, 256, 256, 64, 1, 1 | 8.81M | 8.92M | 0.8030 |
| 2, 4, 8, 48, 144, 144 | 256, 256, 256, 32, 1, 1 | 9.44M | **9.56M** | **0.8060** |

**Round 1, the wider schedules, for comparison:**

| widths | K per tier | table | total | valid MRR |
|---|---|---|---|---|
| 2, 4, 4, 16, 48, 144 | 256, 256, 256, 64, 8, 1 | 5.06M | 5.17M | 0.7681 |
| 2, 4, 8, 24, 72, 144 | 256, 256, 256, 64, 8, 1 | 7.13M | 7.25M | 0.7913 |
| 4, 8, 16, 36, 72, 144 | 256, 256, 256, 64, 8, 1 | 10.92M | 11.04M | 0.8001 |
| 4, 8, 16, 48, 144, 144 | 256, 256, 256, 16, 1, 1 | 11.63M | 11.75M | 0.8033 |
| 4, 8, 24, 72, 144, 144 | 256, 256, 256, 16, 1, 1 | 14.90M | 15.02M | 0.8115 |

Frontier: **9.56M at 0.8060** (−0.0095, inside the 0.01 bar) and **7.60M at
0.7973** (−0.018, inside 0.02). Saved as `checkpoints/cpb1_s0_1.pt` and
`cpb1_s0_0.pt`.

## Readings

**1. The budget belongs in the high-degree tiers, and it is not close.**
The clearest pair: (4, 8, 16, 36, 72, 144) at 11.04M reads 0.8001, while
(2, 4, 8, 32, 96, 144) at 8.51M reads 0.8006 — 2.5M fewer parameters for the
same MRR. The first widens tiers 0–2 (6.6% of answers) and narrows tier 4
(29.9% of answers). Holding tiers 4–5 at full width and cutting the tail to
(1, 2, 4) instead reaches 0.7973 at 7.60M. Tier 3's width is the sensitive
knob at these sizes: 16 → 24 → 32 → 48 moves 0.7869 → 0.7974 → 0.8030 →
0.8060, while (2, 4, 8) → (1, 2, 4) across the three tail tiers costs only
0.005–0.006 and saves 1.3M (0.8030 → 0.7973 at fixed tier 3 = 32; 0.7974 →
0.7923 at tier 3 = 24).

**2. K and width trade differently here than on wikikg2.** At ~5M, more width
with fewer subspaces wins: (2, 4, 8, 32, 96, 144) at K = 64 reads 0.7824 for
5.56M against 0.7681 for (2, 4, 4, 16, 48, 144) at K = 256 and 5.06M. On
wikikg2 the same comparison went the other way ("halving the tail width costs
0.008–0.016 while 256 → 1024 gains +0.015"). The mechanism is the row count,
not the dataset: a bank of K subspaces is fixed overhead amortised over the
tier's entities, and biokg's tiers hold thousands to tens of thousands of rows
where wikikg2's held hundreds of thousands.

**3. Post-hoc compression works on biokg where it failed on wikikg2.** CP1's
single-subspace-per-tier PCA collapsed to 0.4834 on wikikg2 and the conclusion
was that trained tail rows are near-isotropic, with no shared narrow subspace
to project onto. The same construction here reads 0.7180 — a real loss, but
not a collapse — and clustering recovers to within 0.01 of the wide model with
no training at all. biokg's rows are far more clusterable, which is consistent
with 93,773 entities in five type-disjoint ranges and type-matched negatives:
the model never has to separate a protein from a side-effect.

**4. Honest caveat on the "variance kept" column.** Once K > 1 the number
reported is the *within-cluster* variance kept, so it drops when clustering is
turned on (tier 0: 0.517 at K = 1 → 0.166 at K = 16) while MRR rises by 0.052.
The between-cluster structure has moved into the per-cluster offsets `mu`,
which that statistic does not count. It is comparable across configurations at
equal K, not across K.

## Status

A curve point, not a filing. The single-model row stays at 27.1M / 0.8158
test. What is not yet run, in the order it would matter:

* the refit (CP2/CP3 analogue): coefficients and projections trained with the
  wide model as a T = 2 teacher, operators frozen. On wikikg2 a full 200k-step
  refit added +0.012 over its init. That would need the tiered table wired into
  `train_biokg_comp.py` (a `--tiered-from` path), which this port does not add.
* the retrieval blend: whether the 9.56M row still gains the +0.031 that the
  Jaccard/analogy members give the 27.1M single (0.8158 → 0.8463). The members
  are computed from the graph, not the table, so most of that should survive —
  but it is unmeasured.
* other seeds. Everything above is seed 0 only.

# CP-B2: does the compact row survive the retrieval blend? (registered 2026-09-08 13:35, user approved)

CP-B1 left the 9.56M row at 0.8060 on validation against the wide model's
0.8155 (seed 0). The question that decides whether it is a curve point or an
entry is what the four label-free retrieval members do to it. Precedent from
wikikg2 (CK4): distillation's +0.022 on the MODEL became +0.0014 on the
BLENDED row — "the members carry most of what the wider or better-trained
model would add." If that transfers, the compact row blends to near row B's
number at 35% of the parameters.

Model: `checkpoints/cpb1_s0_1.pt`, widths (2, 4, 8, 48, 144, 144),
K = (256, 256, 256, 32, 1, 1), 9,555,497 parameters, init only (no refit),
expanded to the public dense checkpoint format by `tiered_to_dense.py` so the
released blend scripts run against it unchanged.

Members, identical to row B seed 0: the model's own scores, analogy max and
top-3 mean from its table, and the shared model-free Jaccard max and top-3
mean from the training graph. Blend: `ensemble_weights.py --norm z --seed 0
--min-rows 2000`, the per-relation SELECTION recipe (a choice among a fixed
candidate list per relation group). `learned_blend.py` is NOT used — its
Adam-fitted per-relation weights are gradient search on validation labels,
which OGB's rule forbids and which is why the wikikg2 F / C-F rows are not
filed.

Comparator, same seed and same members (`results/sparse/pure_single_s0.log`,
`committed_single_s0.json`): the 27.1M single blends to held-out valid
**0.8461** and committed test **0.84605**; model-only held-out is 0.8142.

**Bars, fixed now, before the blend is computed:**
* held-out valid ≥ **0.8261** (within 0.02 of 0.8461) → the compact row earns
  the single frozen test read;
* held-out valid ≥ **0.8361** (within 0.01) → it is a filing candidate as a
  compact biokg row alongside the 27.1M one;
* below 0.8261 → NO test read; recorded as a validation-only curve point.

**Test protocol, if the bar is met:** weights frozen on the FULL validation
split by `freeze_test.py`, applied ONCE to the test caches, official Evaluator,
one number. Building the test-split member caches uses the training graph only
and reads no test label; the single Evaluator call is the one test read for
this seed. No tuning of any kind after it.

## CP-B2 RESULT, validation stage (2026-09-08 14:30, jinx 1080 Ti): the members absorb two thirds of the compression loss

Held-out estimate on validation halves (`ensemble_weights.py --norm z --seed 0
--min-rows 2000`, fit on a random half of the triples, reported on the other),
five members, seed 0 throughout. Control = the same command with the 27.1M
model's own members, run here to check the pipeline against the committed
receipt.

**Pipeline control reproduces `results/sparse/pure_single_s0.log` exactly**
(`runs/pure_single_s0_repro.log`): uniform held-out 0.8388, model-only
held-out 0.8142, global best top3 0.8413, per-relation fit-half 0.8473,
HELD-OUT 0.8461, delta +0.0073, 5/102 local groups and 93 family weights —
every figure identical to the shipped log.

| row | table params | total params | model-only held-out | blended HELD-OUT |
|---|---|---|---|---|
| wide single (`sparse_s0`) | 27.01M | 27.12M | 0.8142 | **0.8461** |
| compact CP-B1 (`cpb1_s0_1`) | 9.44M | **9.56M** | 0.8049 | **0.8430** |
| difference | −65% | −65% | −0.0093 | **−0.0031** |

Member MRRs alone, from the compact table: analogy max 0.6482, analogy top-3
mean 0.7537 (positive has holders on 100% of rows). Uniform z-mean blend
0.8363 held-out; the per-relation selection adds +0.0066 (7/102 groups took
local weights, 90 family).

**Reading.** The 0.0093 the narrow table loses on its own scores becomes
0.0031 once the four label-free members are in — the members carry two thirds
of what the wider table was contributing. This is the biokg analogue of
wikikg2's CK4 finding (distillation's +0.022 on the model became +0.0014 on
the blended row) and it argues that the CP2/CP3-style refit, worth about
+0.012 on the model, would be worth a fraction of that here and is not the
next thing to run.

Both bars registered above are met: 0.8430 ≥ 0.8361 (within 0.01 of 0.8461),
so the compact row is a filing candidate and has earned the single frozen
test read. Test-side caches (model scores, analogy, shared Jaccard) are built
from the training graph with no test label read and no test number printed;
`freeze_test.py` follows with weights frozen on the FULL validation split.

## CP-B2 RESULT, the committed test read (2026-09-08 14:55, jinx 1080 Ti): 0.8433 at 9.56M

One test read, weights frozen on the full validation split, applied once,
official OGB Evaluator (`results/cpb2/committed_cpb1_w48_s0.log`, receipt
`committed_cpb1_w48_s0.json`, frozen weights `frozen_cpb1_w48_s0.npz`):

| row | params | valid (held-out) | **test MRR** | hits@1 | hits@3 | hits@10 |
|---|---|---|---|---|---|---|
| B. single + retrieval members, seed 0 (committed) | 27,124,129 | 0.8461 | 0.84605 | — | — | — |
| **CP-B1 compact + the same members, seed 0** | **9,555,497** | 0.8430 | **0.8433** | 0.7832 | 0.8857 | 0.9515 |
| difference | −64.8% | −0.0031 | **−0.0028** | | | |

The frozen fit chose uniform global weights (in-sample full-valid 0.8442) with
2/102 groups taking local weights and 91 family weights. Validation predicted
test to 0.0003, the same tight agreement the wide row shows (0.8461 → 0.84605).

**What this establishes.** A 65% parameter cut costs 0.0028 test MRR once the
label-free members are in, and the compact table was never trained — it is the
k-means + per-cluster PCA initialisation of the released single model's own
rows, with the operators copied and frozen. The compression is post-hoc and
takes about ten seconds on a 1080 Ti.

**Why the refit is now the wrong next step.** The narrow table loses 0.0093 on
its own scores and 0.0028 after blending: the members absorb roughly two thirds
of any model-quality difference. A CP2/CP3-style refit is worth about +0.012 on
the model, which by that exchange rate is worth ~+0.004 blended, and it costs a
200k-step training run plus a `--tiered-from` path in `train_biokg_comp.py`.
The cheaper and more informative work is seeds.

## Status after CP-B2

**Not a filing.** This is seed 0 alone; the biokg rows are ten-seed means with
one test read each, and a single seed cannot say whether −0.0028 is typical.
Seed 0's wide model is also slightly below its own ladder (valid 0.8155 against
the ten-seed 0.8164), so the compact row is measured against a marginally weak
reference.

To make it an entry: seeds 1–9 of the same construction (each needs its analogy
pair, ~20 min GPU per split, and one frozen test read; the Jaccard pair is built
and shared), then mean ± unbiased σ. Everything else is in place — the tiered
checkpoints are produced in ~10 s per configuration from an already-released
model, and `tiered_to_dense.py` makes them load in every existing script.

# CP-B3 (registered 2026-09-08 15:05): compress the DISTILLED model, not the plain one

CP-B1/B2 compressed `sparse_s0` (row A, 0.8158 alone / 0.8461 blended). The
submission-grade biokg model is row C's T=2 distilled student
(`checkpoints/dist_T2_s0.pt`, 0.8528 blended over ten seeds), which is the same
architecture and parameter count, so the identical tiered construction applies.
Init-only sweep, validation only, no test read, cutoffs and configurations from
CP-B1's frontier. Reported against the distilled model's own wide validation
MRR. If the compact distilled row holds the CP-B2 pattern it is the entry worth
a ten-seed campaign, and CP-B1/B2 become the method's development record.

## CP-B3 bars (registered 2026-09-08 15:20, before the blend was computed)

Comparator, seed 0, same members and settings
(`results/sparse/pure_distT2_s0.log`, `committed_distT2_s0.json`): the 27.1M
distilled model blends to held-out valid **0.8532** and committed test
**0.85306**; model-only held-out is 0.8308.

Compact distilled (`cpb3_dist_w48_s0`, 9,555,497 params) model-only reads
0.8176 fp32 / 0.8173 in the f16 cache, a gap of 0.0135 before blending.

* held-out valid >= **0.8332** (within 0.02) -> earns the single frozen test read;
* held-out valid >= **0.8432** (within 0.01) -> filing candidate as entry 2;
* below 0.8332 -> no test read; the compact distilled row is recorded on
  validation only and entry 2 is dropped in favour of the CP-B1/B2 chain.

Note the seed-0 test read for the `sparse_s0` chain (CP-B2, 0.8433) is already
spent and is NOT reusable here; this is a separate model and gets its own
single read under the protocol.

## CP-B3 RESULT, validation (2026-09-08 15:45): entry 2 clears the filing bar

`runs/pure_cpb3_s0.log`, seed 0, five members, same settings as row C's receipt:

| row | params | model-only held-out | blended HELD-OUT |
|---|---|---|---|
| C. distilled + retrieval (`dist_T2_s0`) | 27,124,129 | 0.8308 | **0.8532** |
| **entry 2: compact distilled (`cpb3_dist_w48_s0`)** | **9,555,497** | 0.8160 | **0.8478** |
| difference | −64.8% | −0.0148 | **−0.0054** |

Members alone from the compact distilled table: analogy max 0.6507, top-3 mean
0.7561. Uniform z-mean 0.8368; the per-relation selection adds +0.0110 (global
soft20, 5/102 local groups, 90 family).

Both bars met (0.8478 >= 0.8432). **The absorption ratio reproduces**: the
members carried 63.5% of the compression loss here against 66.7% on the
`sparse_s0` chain (CP-B2), from two different source models. The compact
distilled row is entry 2 and the ten-seed campaign is justified.

Configuration is FROZEN as registered — widths (2, 4, 8, 48, 144, 144),
K = (256, 256, 256, 32, 1, 1), cutoffs (5, 8, 32, 128, 1024). The three-point
check on `dist_T2_s0` confirmed the schedule transfers (9.56M 0.8176 >
8.92M 0.8151 > 11.75M 0.8142); it is not re-tuned after seeing the blend.

Campaign: `scripts/run_cpb3_campaign.sh`, one frozen test read per seed. All
ten `dist_T2_s*.pt` staged and sha256-verified against the release SHA256SUMS.

## CP-B3 portability note (2026-09-08 16:10): the compression is device-deterministic, NOT device-portable

Checked because the entry-2 pitch was "no new release assets: regenerate the
model from the released checkpoint in ten seconds". That claim is WRONG as
stated and is withdrawn.

* **Same device, re-run: bitwise identical.** Re-running `compress_biokg.py` on
  the same GPU reproduces `coef`, `proj`, `mu`, `cluster_of` and the
  reconstructed table exactly (max|diff| = 0). `torch.linalg.eigh` is
  deterministic run-to-run here, so even the basis matches — the expected sign
  flips do not occur.
* **CPU vs GPU: different model, same quality.** Cluster assignments differ and
  the reconstructed table differs by max|diff| 2.09 (49% relative). Cause:
  `init_from_wide` seeds k-means with `torch.randperm(..., device=dev)`, and the
  CPU and CUDA generators give different streams for the same seed; 15 k-means
  iterations amplify it. The resulting table scores valid **0.8197** against the
  GPU table's **0.8176** — equally good (here 0.002 better), just not the same
  artifact.

**Consequences.**
1. The compact checkpoints MUST ship as release assets. `compress_biokg.py`
   alone reproduces a statistically equivalent model, not the one whose test MRR
   was committed, so `verify.py`-style re-scoring needs the actual files
   (~40 MB each, ten seeds).
2. Two different CUDA devices may also diverge: the Philox stream matches across
   CUDA cards, but k-means iterations use floating-point reductions whose order
   is device-dependent. If the campaign is sharded over mixed GPUs, each seed's
   committed number belongs to the card that produced it. Recording the device
   per seed in the receipts, and preferring one architecture for all ten.
3. There is a clustering-initialisation component of variance (~0.002 on this
   evidence) on top of the model-seed variance. The k-means seed stays fixed at
   0 as registered; it is NOT selected on validation, which would inflate the
   reported number and would have to be disclosed as tuning if it were.

## CP-B3 RESULT, ten seeds (2026-09-08 17:35, jinx 1080 Ti): entry 2 is 0.8468 at 9.56M

Ten seeds, one frozen test read each, weights fixed on the full validation
split. Receipts in `results/cpb3/`; `scripts/summarize_cpb3.py` regenerates
every figure below from them.

| row | params | model alone (held-out) | valid held-out | **test MRR** |
|---|---|---|---|---|
| C. distilled + retrieval | 27,124,129 | 0.8305 ± 0.0003 | 0.8532 ± 0.0003 | **0.8528 ± 0.0002** |
| C'. compact + retrieval | **9,555,497** | 0.8162 ± 0.0007 | 0.8478 ± 0.0003 | **0.8468 ± 0.0003** |
| cost of 2.84x compression | −64.8% | +0.0143 | +0.0054 | **+0.0061** |

Per seed: 0.8466 0.8470 0.8466 0.8473 0.8466 0.8460 0.8467 0.8469 0.8471
0.8467. test hits@1/3/10 = 0.7888 / 0.8880 / 0.9513. Held-out minus test:
mean +0.0011, max 0.0015 — validation is mildly optimistic and consistently
so, in the same direction on all ten seeds.

**The two claims, as measured.** (1) The members absorb **57%** of the
compression loss: 0.0143 on the model's own scores becomes 0.0061 blended,
because the analogy and Jaccard features read the training graph rather than
the entity table. The `sparse_s0` chain gave 0.0093 -> 0.0028 (70%) on a
different source model, so the effect reproduces but the ratio is not a
constant. (2) The cost is steady across seeds: gaps span 0.0056–0.0067 with
one mild outlier (seed 5, 0.0067), and the compact row's spread (±0.0003) is
within a whisker of the uncompressed row's (±0.0002), so the compression adds
essentially no variance of its own. The mean gap did not move from 0.0060–0.0061
between n=4 and n=10.

**Board position** (fetched 2026-09-04): C' would sit 6th, 0.0024 below
ComplEx-RP (0.8492, 188M) and 0.0120 above TripleRE (0.8348, 470M), at 9.8x
fewer parameters than AutoSF (94M), the smallest entry currently listed, and
49x fewer than TripleRE which it beats.

**Written up in**: `paper/resonate.tex` §"Compressing the entity table after
training" (construction, the wikikg2 contrast, where the width belongs, the
result table, and the device-portability disclosure); `site/index.html` biokg
table row C' plus a finding block.

**Still open**: the cross-machine verification — re-scoring the ten committed
compact checkpoints on a second machine to establish the "<3e-4 across
machines" claim `VALIDATE.md` makes for the released rows. The compact
checkpoints must be released as files (see the portability note above); the
script alone does not reproduce the committed artefacts on a different device.

## CP-B3 release + cross-machine verification (2026-09-08 18:20)

Release staged as `checkpoints/tiered_T2_s{0..9}.pt` (39 MB each, 387 MB total,
the TIERED artefact at 9,555,497 parameters -- not the 108 MB dense expansion,
which `tiered_to_dense.py` regenerates), with `SHA256SUMS.compact`,
`scripts/fetch_checkpoints.sh compact` (tag `v2.1-compact`) and
`verify_compact.py`.

| machine | torch | max abs deviation vs logged |
|---|---|---|
| GTX 1080 Ti (built them) | 2.6.0+cu124 | 0.00004 |
| RTX 5090 (vast.ai 50261550) | 2.11.0+cu128 | 0.00004 |

All ten reproduce on both, and the per-seed deviations are IDENTICAL to five
decimals, so the two machines agree with each other to better than 1e-5 across
two GPU architectures and two CUDA builds. Checksums re-verified after the
387 MB transfer (10/10 OK).

This settles the distinction the portability note opened: **the released files
are device-independent** (reconstructing a row is `coef @ P + mu`, a matmul),
while **the construction is not** (k-means seeds from the device RNG). Shipping
files rather than a re-run script is therefore necessary, and now measured
rather than argued.

Boxes: three RTX 5090s are rented (vast CLI); 50209059 and 50270859 were busy
with the wikikg2 TR1 refits and were left alone. 50261550 was idle (0% GPU,
0.04% CPU) and was used for this. Setup there: torch 2.11+cu128 and ogb 1.3.6
were already in `/venv/main`; only the code, the checkpoints and the 2.9 GB
biokg download were added.
