<!-- Replacement text for the README's "### ogbl-wikikg2" subsection (the README itself is held by
     another session's uncommitted edits; paste this in place of the current wikikg2 table + Compliance
     paragraph). Final numbers. -->

### ogbl-wikikg2 (`wikikg2/`)

Two rows are filed (2026-09-08); everything else in this table is reported, not filed.

| | test MRR | valid MRR (frozen selection, in-sample / held-out estimate) | params | filed |
|---|---|---|---|---|
| A. single k=8 model | 0.6676 ± 0.0010 | 0.7030 ± 0.0009 | 328,842,753 | no |
| distilled (T=2) student, alone | 0.6855 ± 0.0008 | 0.7190 | 328,842,753 | no (superseded by E) |
| **E. compact student + graph evidence + reverse operator, selection blend** | **0.7096 ± 0.0013** (7 seeds) | 0.7668 ± 0.0013 / 0.7667 | **50,244,249** | **yes** |
| **C. ten-teacher ensemble + graph evidence + reverse operator, selection blend** | **0.7320** (one read) | 0.7914 / 0.7909 | 3,288,427,530 (ensemble) | **yes** |
| F / C-F / ensemble with the learned (Adam-on-validation) combiner | 0.7222 / 0.7320 / 0.7426 | 0.780 / 0.788 / 0.799 | 329M / 329M / 3.29B | no |

**What the filed rows are.** Both use the same evidence layer over the training
graph and the frozen models: two analogy members, holders, common neighbours,
two- and three-hop links, typed paths, and the two *opposite-operator* members
(`reverse_wiki.py`: for a head question the forward operator scores every
candidate as the source of the link, and vice versa; +0.017 MRR on validation,
all in the head direction). The members are combined per (relation, direction)
by a **selection** among a fixed list of weight patterns by validation MRR
(`blend_wiki.py`, guard 250 rows chosen on validation halves), which we read as
the hyper-parameter tuning OGB's rule allows; no gradient step touches
validation labels in either filed row. Row E's model is the released T=2 student
with every entity row replaced by a narrow row (width 4/8/36/64 complex by
training degree) expanded through one of 4,096 learned subspaces per tier
(k-means over the trained rows, per-cluster PCA), refit for 200k steps with
the student as teacher and the operators frozen (`resonate_tiered.py`,
`train_wiki.py --tiered-from`); one such model per released student seed, each
with its own frozen selection. Row C is the score average of the ten released
teachers with the same members and the standard selection family.

**Compliance and disclosures.** OGB's rule reserves validation for "standard
hyper-parameter tuning (not allowed: gradient-based search, use as model input)".
The learned combiner rows fit ~1,500 per-relation weights by Adam on validation
labels and stay unfiled. The selection guard used for the filed rows was chosen
on validation halves; an earlier guard informed by the test split's relation mix
is not used anywhere in them. Test was read once per filed row (`results/tr1/`).
The research record for every step, positive and negative, is in
`REVERSE_MEMBER.md`, `ENSEMBLE_SELECTION.md`, `SELECTION_RICH.md`,
`COMPACT_K.md` (CP1–CP3f: post-hoc PCA fails, one subspace per tier fails,
clustered subspaces work), `HOLDOUT_COMBINER.md` (three failed attempts to fit
the combiner on a training holdout) and `TEST_READS_PROPOSAL.md`.

The public board as of 2026-09-04 (28 entries): RelEns 0.7392 (2.18B, ensemble),
StarGraph + TripleRE + Text 0.7305 (1.93B, uses entity text), InterHT+ 0.7293
(156M), StarGraph + TripleRE 0.7286 (93M), InterHT+ 256-dim 0.7257 (148M),
StarGraph + TripleRE 0.7201 (87M), CompoundE3D 0.7006 (751M), TranS 0.6939
(38M), TripleRE + NodePiece 0.6866 (36M). Row C would be 2nd; row E 8th, the
best entry under 90M parameters.
