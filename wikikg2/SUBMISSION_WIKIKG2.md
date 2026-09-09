# ogbl-wikikg2 filing (2026-09-09: entry E only; all reads done)

**One entry is filed: E, the compact single model.** Both rows below were
prepared by allowed means (validation used only to select among fixed
per-relation weight patterns; no gradient touches validation labels; one test
read per row through the official Evaluator), but the decision of 9 September is
to file E alone. Entry C stays here as the documented result, not as a
submission. Team: Cristian Malaia. Code and checkpoints:
https://github.com/jinxmcg/resonate (release v2.0-two-boards plus the compact
checkpoints and frozen selection weights).

Against the 2026-09-04 board, E at 0.7100 would rank 7th; C at 0.7320 would have
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
| test MRR | **0.7100 ± 0.0014** over ten seeds (s0–s9), one read each |
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

## RESOLVED 2026-09-09: the ten-seed requirement is met (see the TR2 completion note at the end)

The section below records why E was blocked; TR2 closed it and E is now form-ready.

### (historical) NOT READY TO FILE: seven seeds, and OGB requires ten

OGB's submission rules state that the "average (`torch.mean`) and unbiased
standard deviation (`torch.std`) must be taken over 10 different random seeds".
**Entry E currently has seven** (s1, s4-s9). Students 0, 2 and 3 were deleted by
the campaign script after their TR1 reads and cannot be recovered from a
checkpoint, so the three have to be retrained before this entry can go in.

That is exactly what TR2 was registered for (`TEST_READS_PROPOSAL.md`) and what
is running now: `scripts/tr2_students.sh` retrains seeds 0, 2 and 3 with the
row-C recipe and puts each through the full E procedure -- compress with the
student's own clusters, refit 200k, nine members, rich selection frozen on full
validation, one read each -- to be reported with the seven as a mean over ten.

**Consequences for the form.** Do not submit E until those three land: the seed
count is a stated requirement, not a preference, and the mean and standard
deviation above will both move when n goes from seven to ten. The biokg entries
are unaffected -- C and C' each have their full ten seeds with one read apiece
-- and can be filed now.

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

## TR2 completion note (2026-09-09)

Entry E was filed-blocked at seven seeds: OGB requires the mean and unbiased
standard deviation over ten random seeds, and the students for seeds 0, 2 and 3
had been deleted after their original reads. TR2 part 1 retrained those three
(`train_wiki.py --distill <ten teachers> --distill-T 2.0`, 400k steps) and ran
the standard E procedure on each -- compress with the student's own clusters,
refit 200k, nine members, rich selection frozen on full validation, one test
read. Nothing else changed; the seven existing seeds were not re-run or
re-read.

| seed | test MRR | | seed | test MRR |
|---|---|---|---|---|
| 0 (new) | 0.7109 | | 5 | 0.7105 |
| 1 | 0.7116 | | 6 | 0.7085 |
| 2 (new) | 0.7097 | | 7 | 0.7083 |
| 3 (new) | 0.7122 | | 8 | 0.7087 |
| 4 | 0.7090 | | 9 | 0.7107 |

**Ten seeds: 0.7100 ± 0.0014** (the seven-seed figure was 0.7096 ± 0.0013). The
three new seeds fall inside the existing spread, so the retrained students
reproduce the recipe rather than drifting from it; seed 0's student read valid
MRR 0.7190, matching the released students exactly.

Thirteen test reads were registered for TR2; ten were used (three here, and the
ten leave-one-out reads for entry C were NOT taken -- entry C is documented and
not filed, so its spread is not needed). Run record: seeds 2 and 3 completed on
vast.ai 50209059; seed 0's student and refit completed on 50270859, whose disk
filled during member building and truncated `holders.test.npz` to zero bytes,
failing the blend with `EOFError`. The student and refitted model were recovered
and the members and read redone on 50209059 -- no retraining was repeated.
