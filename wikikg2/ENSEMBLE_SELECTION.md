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
