# TR1 proposal: test reads for the allowed rows (DRAFT for the user's approval; nothing here has run)

Written 2026-09-08 04:00 while the user slept. No test cache exists on the box
for any of these rows; building them is part of the read and waits for the go.

Rows, all "selection blend on full validation, frozen, applied once to test"
(`blend_wiki.py freeze --test`, guard 250, the same members as measured):

| row | validation held-out | test estimate | params | reads |
|---|---|---|---|---|
| A. T=2 student s1 + nine members (analogy, holders, cn_aa, linked, cn3_aa, typed, rev_raw, rev_nov) | 0.7711 | ~0.71 | 329M | 1 per released student (7) for a mean ± std, or 1 for s1 alone |
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
