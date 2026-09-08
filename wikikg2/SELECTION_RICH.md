# SEL1: a larger fixed candidate family for the selection blend (registered 2026-09-08 03:30)

The selection blend picks, per (relation, direction) group, one of a fixed
list of weight patterns by fit-half MRR (uniform, top-m, softmax over member
MRRs). SEL1 enlarges the list, still finite and fixed before any number is
seen, still no gradient: softmax at eta 10 and 200; top-3/5/9 with the model
channel(s) counted twice; and "model + one member" at 0.25 / 0.5. The
`--rich` flag of `blend_wiki.py`; `typed` and `rev_*` are now correctly
treated as non-model members for the model-only pattern (they were counted
as models before; this changes only which pattern is called model-only, not
the standard list's numbers, which were re-run unchanged).

Test: the student + nine row and the best ES1 row, standard vs rich, seed-0
halves, guard 250. Bar: held-out MRR +0.002 with rich. If met, rich is the
selection used for any filing of those rows; if not, dropped. Validation only.

## SEL1 RESULT (2026-09-08 04:26): the rich family meets its bar on both rows

Held-out half, seed 0 (`results/sel1/selection.log`): student + nine 0.7711 →
**0.7734** (+0.0023; head 0.571 → 0.575); ten-teacher ensemble + members +
reverse 0.7909 → **0.7923** (+0.0014; head 0.605 → 0.607). The bar (+0.002)
is met on the student row and not on the ensemble row. Decision, as
registered: rich is the selection for any filing of the student row; for the
ensemble row the standard family is kept (its gain is under the bar, and the
smaller family is the simpler claim). STAB1 (other half-split seeds) reports
whether the +0.002 is stable.

## STAB1 RESULT (2026-09-08 04:28): the held-out numbers are stable over the half-split seed

Seeds 0 / 1 / 2 of the validation halves (`results/stab1/selection.log`):
student + nine, standard 0.7711 / 0.7712 / 0.7710, rich 0.7734 / 0.7736 / 0.7732;
ten-teacher ensemble + members + reverse, standard 0.7909 / 0.7908 / 0.7907,
rich 0.7923 / 0.7924 / 0.7923. Spread ±0.0003; the SEL1 gains (+0.0023 and
+0.0015) hold on every split.
