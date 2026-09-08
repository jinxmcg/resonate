# H34 — relation-aware analogy screen

Completed 2026-09-05 on the local GTX 1080 Ti. **Did not pass the preregistered
advance gate. No promotion and no full-validation confirmation launched.**
This is a negative/inconclusive screen of one specific zero-training metric
replacement, not a rejection of all relation-aware retrieval.

## Matched validation results

5,000 fixed validation triples, both directions, all 500 official typed
negatives: 10,000 queries. The tuning half contains 2,487 triples and the
report half 2,513 triples (5,026 queries). Both arms use the same frozen
27,124,129-parameter T=2 student, same Jaccard features, same finite mixture
grid, and same paired split. Only the analogy metric differs.

| Report-half metric | Global-cosine control | Relation-cosine replacement |
| --- | ---: | ---: |
| MRR | 0.8410782893 | 0.8398110583 |
| Hits@1 | 0.7799442897 | 0.7785515320 |
| Hits@10 | 0.9500596896 | 0.9506565858 |

Paired MRR change: **−0.0012672311**. Paired-triple bootstrap 95% interval:
**[−0.0033761355, +0.0009693595]**, 1,000 replicates. This interval crosses
zero: do not claim a statistically established regression. It does not
include training-seed uncertainty or correct for repeated development-set use.

Both arms selected the same weights on the tuning half:
`[student=.5, analogy-max=.125, analogy-top3=.125, Jaccard-max=.125, Jaccard-top3=.125]`.
With fixed uniform weights, the report-half MRR also decreased:
0.8328140103 → 0.8314302112. The frozen student alone scored 0.8283475199 over
the complete 10,000-query screen.

These scores are **not** directly comparable with the historical submission
pipeline's full-validation / held-out scores: this is a different subset and
uses global finite mixture selection, not the historical per-relation recipe.

## What we learned

- The learned operators genuinely change cosine geometry; this was not an
  ineffective rotation. Per-direction/relation metric audits are in the JSON.
- The treatment improved some small groups but lost on drug–drug in both
  directions (approximately −0.00992 and −0.00572 MRR in this report half).
  Those are exploratory slices, not a reason to retrofit relation-specific
  switches on the same report labels.
- The hypothesis that an existing relation operator automatically supplies a
  better analogy metric is not supported by this screen. Next in the queue:
  change the single-model scorer's expressivity (H35), instead of promoting
  this feature replacement.

## Integrity / reproducibility

- No model training, validation gradients, TEST split/cache loading, or rented
  GPU use. LR is not applicable to a frozen inference experiment.
- TRAIN-only unique-edge graph; exclude each query source from candidate
  holders. Deduplication removed one duplicate edge (counted once in each
  direction). Both arms use exactly the same deduplicated graph.
- FP32 feature generation, normalization and mixture scores, TF32 disabled.
  Official average-tie ranks; all candidates scored symmetrically. A separate
  post-run check against the installed OGB Evaluator verified every query for
  all seven feature channels and both selected mixtures. Maximum reciprocal
  rank difference was 1.99e-8, from float32 vs float64 reciprocals.
- Seven new synthetic tests passed, including end-to-end brute-force feature
  parity, missing/self-holder handling, candidate permutation, unitary-metric
  invariance, selective loader invocation, no checkpoint mutation, and tuning
  isolation. Entire BioKG test suite: **43 tests passed**.
- Measured feature generation + saving + mixture selection/bootstrap: **26.86
  seconds**. Excludes checkpoint/data loading and dependency setup. Snapshot
  during execution: GPU memory 2,495 MiB total including the existing desktop,
  temperature 42°C. GPU compute process was gone after completion.
- Checkpoint: released `dist_T2_s0.pt`, SHA256
  `b463a8bcc431ada38f4834e235677464f9cd3a86346f88aecd2fd681a01260ef`.
  Checkpoint configuration records seed 0, T=2, 50k steps, k=12, block size 4,
  `eval=valid`; teacher paths reference ten sparse BioKG models. This verifies
  the recorded recipe, not an independent audit of every historical stage.

The immutable [launch protocol](protocol_at_launch.md) matches the
`protocol_sha256` in [prerun.json](screen_s0/prerun.json). The live
[optimization queue](../../../optimizations.md) can evolve without changing
that record. [summary.json](screen_s0/summary.json) has all metrics and geometry
audits; `valid_indices.npy`, `features.npy`, and `queries.npz` preserve the
sample, seven raw feature matrices, and paired ranks (about 134 MiB total).

From the repository root, use a **new** output directory:

```sh
/mnt/geocore/geocore/.venv/bin/python -m biokg.relation_analogy \
  --model biokg/checkpoints/dist_T2_s0.pt \
  --data-root /mnt/geocore/geocore/data_ogb \
  --device cuda --triples 5000 --chunk 32 --threads 4 \
  --out biokg/results/h34/reproduction_s0
```

Do not run this merely to shop for another screen seed or change the result.
No H35 training has started yet.
