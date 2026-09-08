# TRAIN-only loss-gradient diagnostic

Fixed before diagnostic execution, 2026-09-06. User authorized local GTX
1080 Ti. This measures gradients, not generalization or an architecture gain.

- Read our released `dist_T2_s0.pt` and H35F `single.pt` only, plus the same
  ten frozen, own H24 teachers used for their distillation. Verify hashes
  against prior receipts before loading. No external trained artifacts.
- Do not construct an optimizer, call backward into parameter `.grad`, alter
  any parameter, save a new checkpoint, or launch an ablation automatically.
  Learning rate is explicitly 0 (no optimizer); MRR is not evaluated.
- Open only official TRAIN and raw node-count metadata. Use an audit hook to
  reject other dataset files, including VALID, TEST and processed caches.
  No OGB dataset constructor, validation error masks or evaluation code calls.
- Both A snapshots get the same 512 batches from `TrainStream`, seed 3601,
  2,048 positives and 4,096 shared typed negative draws per batch. Preserve
  existing relation-frequency sampling and uniform head/tail direction;
  do not filter sampled positives or duplicates or change the objective.
- Compute gradients of CE, T=2 KD weighted by T squared, and trajectory
  distance weighted by 0.1, exactly as in `retained_loss`. Teachers produce
  detached mean logits once per batch. Complex64/fp32, no AMP/TF32.
- Measure real-coordinate Euclidean gradient norms and pairwise cosines for
  entity table, operator bank and log temperature; also representation-only
  (entities + operators) and all-parameter aggregates. Undefined cosines
  from zero norms are absent, not evidence of agreement or disagreement.
- Primary diagnostic: cosine of `g_CE + g_KD` with `0.1*g_trajectory`, its
  norm ratio and `1 + dot(g_CE+KD, g_trajectory_weighted)/||g_CE+KD||^2`.
  The latter measures the combined gradient's projection on CE+KD relative
  to CE+KD alone under infinitesimal vanilla descent. It is **not** a measured
  MRR change or an Adam-update prediction; no optimizer moments are loaded.
- Report batch-distribution mean/median/p10/p90 and negative fraction,
  plus head/tail and TRAIN entity-type-family slices. These are batch-wise
  geometries, not per-example conflict rates or cosines of average gradients.
  Report observed relation coverage; rare slices may have few/no batches.
- On the first batch verify separate gradients sum to the retained total.
  After all batches verify exact before/after tensor hashes for both A
  snapshots and all teachers; no `.grad` populated; all input/source files
  unchanged. Record batch-stream checksum and all dataset paths opened.
- Negative cosine alone is normal in multitask/regularized learning and
  does not justify removing an objective. Interpret its magnitude and
  alignment, compare snapshots, and propose a separate matched training
  ablation only if warranted. No automatic promotion or submission change.

Synthetic tests:

```sh
/mnt/geocore/geocore/.venv/bin/python -m unittest biokg.test_gradient_diagnostic biokg.test_joint_operator
```

Requested local-GPU run:

```sh
/mnt/geocore/geocore/.venv/bin/python -m biokg.gradient_diagnostic --out biokg/results/gradient_diagnostic/s0
```
