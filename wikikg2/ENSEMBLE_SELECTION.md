# ES1: ensemble rows under the selection blend only (registered 2026-09-08 02:25, user: "stay within the rules")

The ensemble rows in the README (ten teachers + members + learned combiner,
0.7426 test) are not filed because of the combiner. This measures the same
ensembles with the selection blend (allowed tuning) and the reverse members.

Ensembles are score averages (`ens_cache.py`, no per-model caches):
* `ens7s`: the seven released T=2 students (s1, s4–s9), the distilled ensemble, 2.3B.
* `ens10t`: the ten released teachers (s0–s9), 3.29B.
* `ens17`: all seventeen, 5.6B.
For each: the ensemble's own score, its rev_raw and rev_nov (averaged over the
same models), plus analogy_d1_t3, analogy_s0_t3, holders, cn_aa, linked,
cn3_aa, typed. Selection blend on validation halves (guard 250, seed 0),
reported with the ensemble alone and with the members without reverse.
Bar for a filing candidate: held-out MRR ≥ the student row (0.7711) + 0.01.
Validation only; the test reads, if any, are a separate registration for the
user to approve in the morning.

## ES1 RESULT (2026-09-08 03:45): both ensembles clear the bar with allowed tuning

Ensembles alone on validation: ens7s 0.7331, ens10t 0.7373, ens17 0.7373
(their reverse members: see `results/es1/ens*.log`). Selection blend, held-out
half (`results/es1/selection.log`):

| ensemble | alone | + members | + members + reverse | head |
|---|---|---|---|---|
| ens7s, seven T=2 students, 2.3B | 0.7331 | 0.7640 | **0.7834** | 0.590 |
| ens10t, ten teachers, 3.29B | 0.7373 | 0.7682 | **0.7909** | 0.605 |
| ens17, all, 5.6B | 0.7373 | 0.7678 | 0.7903 | 0.603 |

Bar 0.7721 (student row + 0.01): ens7s and ens10t PASS; ens17 adds nothing
over ens10t and is dropped. Reference: the same ten-teacher ensemble with the
validation-fit learned combiner reads 0.7992 held-out and 0.7426 on test; the
allowed selection with the reverse members is 0.008 below it on validation.
Filing candidates, in the user's hands: ens10t + members + reverse (estimate
~0.735 on test, ±0.01; RelEns is 0.7392) and ens7s + members + reverse
(~0.727), each one test read, under a registration to be approved in the
morning. No test cache was built or read tonight.

## ES2 (registered 2026-09-08 14:40, user's idea): the ensemble built from compact models

Each of the ten released teachers is compressed as in CP3 (its own k-means
clusters, K = 4096/4096/16/1, widths 4/8/36/64, per-cluster PCA, operators
kept, no refit): ten models of ~50M each. `ens10c` = their score average, with
their averaged reverse members (`ens_cache.py`), plus the shared members and
the two analogy members already in use; selection blend on validation halves
(standard and rich), reported next to ES1's ens10t (0.7909 / 0.7923). Bars: a
filing candidate in place of C if it is within 0.01 of ens10t (the size drops
from 3.29B to ~0.5B); reported as a curve point otherwise. If it passes, the
per-model refit (10 × 30 min) is the optional follow-up. Validation only.

## ES2 RESULT (2026-09-08 15:08): the compact ensemble misses the bar; a curve point

Ten teachers compressed as in CP3 (41.5M tables each, no refit; individual
validation 0.6848–0.6901, the wide teachers ~0.704): `ens10c` alone 0.7190,
its reverse 0.6924 (wide ensemble: 0.7373 / 0.7125). Selection blend, held-out
half (`results/es2/selection.log`): ens10c 0.7188; + members 0.7549 (rich
0.7580); + members + reverse **0.7732** (rich **0.7755**), against 0.7909 /
0.7923 for the wide ensemble. The compression cost (~0.018) passes through the
whole stack; the members do not absorb it. Bar (within 0.01 of ens10t) not
met: entry C stays the wide ensemble. Note for the paper: ten compressed
teachers (0.5B) land where one wide student lands (0.7734, 329M). A per-model
refit (+0.005 each on the student) would put the compact ensemble near 0.78;
not run.
