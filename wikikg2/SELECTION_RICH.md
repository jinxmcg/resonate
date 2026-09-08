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
