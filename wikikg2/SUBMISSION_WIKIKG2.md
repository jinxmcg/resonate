# ogbl-wikikg2 filing (2026-09-09: entry E only; all reads done)

**One entry is filed: E, the compact single model.** Both rows below were
prepared by allowed means (validation used only to select among fixed
per-relation weight patterns; no gradient touches validation labels; one test
read per row through the official Evaluator), but the decision of 9 September is
to file E alone. Entry C stays here as the documented result, not as a
submission. Team: Cristian Malaia. Code and checkpoints:
https://github.com/jinxmcg/resonate (release v2.0-two-boards plus the compact
checkpoints and frozen selection weights).

Against the 2026-09-04 board, E at 0.7096 would rank 7th; C at 0.7320 would have
ranked 2nd. C is a ten-model ensemble of 3,288,427,530 parameters — the largest
thing this project has built — and filing it would put the project's headline
wikikg2 number on the one row that argues against its own thesis. E, at
50,244,249 parameters, is the row that carries the claim.

## Entry C — ResonatE ensemble + graph evidence (ensemble) — **NOT FILED**

| field | value |
|---|---|
| method name | ResonatE ×10 + retrieval + reverse (selection blend) |
| test MRR | **0.7320** (one read, 2026-09-08; hits@1 0.6656, hits@3 0.7641, hits@10 0.8643; tail 0.9691 / head 0.4949) |
| validation MRR | 0.7914 (the frozen selection, in-sample); 0.7909 held-out estimate |
| parameters | 3,288,427,530 (ten released k=8 teachers of 328,842,753 each; the members add none) |
| ensemble | yes |
| external data | none |
| hardware | RTX 5090, 40 min per teacher; members and blend minutes |

What it is: the score average of the ten released teachers, plus nine evidence
members computed from the training graph and the frozen models (two analogy
members, holders, common-neighbour, two-hop link, three-hop, typed paths, and
the two opposite-operator members rev_raw / rev_nov), combined per (relation,
direction) by a selection among nine fixed weight patterns chosen by validation
MRR (guard 250 rows, chosen on validation halves). Validation held-out estimate
of the same procedure: 0.7909.

## Entry E — ResonatE compact single model + graph evidence — **THE FILED ENTRY**

| field | value |
|---|---|
| method name | ResonatE compact (clustered narrow rows) + retrieval + reverse |
| test MRR | **0.7096 ± 0.0013** over seven seeds (s1, s4–s9); hits@1 0.6443 ± 0.0010, hits@10 0.8383 ± 0.0024 |
| validation MRR | 0.7668 ± 0.0013 (frozen selection, in-sample); held-out estimate 0.7667 |
| parameters | 50,244,249 (27,681,304 coefficients + 12,746,752 projections + 1,050,752 offsets + 8,765,440 operators + temperature) |
| ensemble | no |
| external data | none |
| hardware | RTX 5090; compression 1 min, refit 30 min per seed |

What it is: the released T=2 student (k=8, 329M) with every entity row
replaced by a narrow row of width 4/8/36/64 complex by training degree,
expanded through one of 4,096 learned subspaces per tier (k-means over the
trained rows, per-cluster PCA), refit for 200k steps with the student as
teacher and the operators frozen. Alone it reads 0.7074 on validation
(student 0.7190); with the nine members and the richer selection family
(SEL1) its held-out estimate is 0.7667 (student row 0.7734).

## Disclosures carried into README and paper

* The learned per-relation combiner (Adam on validation labels; rows F / C-F /
  ensemble, up to 0.7426 test) remains reported and not filed.
* The selection guard (250) is chosen on validation halves; the earlier
  test-mix-informed guard is not used in any filed row.
* The reverse members are functions of the frozen models; the graph members
  read the training graph only; the compact rows are built from the released
  student and refit on training edges.
* Records: REVERSE_MEMBER.md, ENSEMBLE_SELECTION.md, SELECTION_RICH.md,
  COMPACT_K.md (CP1–CP3f), HOLDOUT_COMBINER.md (HC1–HC3), TEST_READS_PROPOSAL.md.
