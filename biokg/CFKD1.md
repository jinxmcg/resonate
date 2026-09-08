# CFKD1: fixed candidate-focused distillation comparison

Registered 2026-09-06 before real training. User approved the proposed
two-arm experiment. Local GTX 1080 Ti only. No external trained artifact,
new teacher, B branch, inference ensemble, architecture change or TEST use.

## Model and intervention

Both independent copies start at H35F `single.pt`, SHA256
`df7c783e7b8a6110f126bca1c60d89209735accbaa5ffc0b362f2cab52d9d2f1`.
This is improved A: k=12, 144 complex coordinates, free 4x4 blocks,
27,124,129 real parameters. Keep its existing standalone checkpoint format.
The same ten own frozen H24 teachers provide detached mean raw logits.

- Control: unchanged `CE + T²*KL_full + 0.1*trajectory`, T=2.
- Focused: control loss plus `0.25*T²*KL_conditional`, T=2.
- Conditional support is the union of each TRAIN query's top 32 teacher
  and top 32 student candidate columns, including all cutoff ties and the
  explicitly known TRAIN-positive column. Selection is detached. Recompute
  conditional student and teacher softmax on that support, not a truncation
  of probabilities without renormalizing.
- Keep full-pool KD, CE and trajectory exactly unchanged. No score rescaling,
  teacher-rank fusion, false-negative filtering, auxiliary labels or features.
  Sampled duplicate IDs and sampled positives remain as in the existing loss.
- Masking never relies on a presumed positive column at evaluation: the
  positive index is a known TRAIN label only. All evaluation candidates use
  the identical standalone scorer. Cutoff ties are included, not broken by
  column order. No candidate selection or teacher exists at inference.

## Matched budget and logging

- Exactly 5,000 paired TRAIN updates, `TrainStream` seed 3603, batch 2,048,
  4,096 shared typed negatives. Natural relation-frequency sampling and
  uniform head/tail direction. Each batch/target shared by both arms.
- Fresh dense Adam, default betas/epsilon, no weight decay, LR 1e-4 for
  every A parameter; common cosine to zero over 5,000 updates, global clip
  1.0. Same architecture, initialization, precision and optimizer schedule.
  Fixed endpoint, no early stopping or best-probe checkpoint selection.
- Complex64/fp32, no AMP or TF32. Record per-arm loss components, selected
  candidate count and teacher mass; cumulative sample/optimizer counts,
  batch-stream SHA256, timing and peak allocation.
- At least every 100 steps or 40 seconds, report actual LR used for that
  update, next-step LR, and last probe MRR with its step explicitly shown.
  At initialization and every 500 updates, use the existing H35 fixed
  1,000-VALID-triple probe (2,000 directed queries, 500 official negatives).
  Probes are read-only and do not choose candidates, targets or any setting.

## Data boundary and evaluation

- Open official TRAIN, VALID and raw node counts only. A file-open guard
  rejects TEST, processed dataset caches and every other dataset path.
  Load splits selectively; no OGB dataset constructor. Verify all checkpoint,
  data and source hashes before and after. No VALID gradient or graph input.
- Verify the initial probe exactly reproduces the corresponding archived
  H35F single-model VALID ranks. That endpoint's full VALID ranks are the
  fixed pre-training reference; never used for learning or sampling.
- After 5,000 updates, save both endpoints and restore each without teachers.
  Evaluate full VALID once per endpoint with the official OGB average-tie
  ranks, both directions, fp32 scores and all 500 supplied negatives.
- Primary: focused minus matched control full-VALID MRR. Two-sided 95%
  whole-triple paired bootstrap, 1,000 replicates, seed 3604. Advance for
  seed confirmation only if delta >= +0.001 and lower bound > 0; additionally
  require focused MRR >= starting A MRR. No automatic promotion or next run.
- Descriptive: each endpoint versus starting A, top-1 recoveries/losses,
  and TRAIN-type-family/head-tail slices. These are evaluation-only and do
  not define examples, labels or weights for further learning.
- This is an adaptive single-seed development screen. Intervals describe
  query variability, not seed uncertainty or pristine-holdout generalization.
  The intervention increases effective KD weight as well as concentrating
  support; it cannot isolate those two mechanisms without another control.
  Feature-pipeline transfer and multi-seed confirmation remain separate.

## Verification

Before real data: synthetic loss/gradient equivalence to sliced conditional
KL, detached teacher/selection, all-tie/full-support behavior, exact control
parity, candidate permutation with remapped positive labels, finite masked
gradients, dataset access restrictions and standalone scorer/restore tests.
Then a full-size 20-update-per-arm CUDA smoke with synthetic entities and
teachers, immutable teacher hashes, independent student storage, all A groups
updated, candidate symmetry and realistic batch shape. Abort on OOM/nonfinite;
do not silently change precision, support size or batch budget.

Audit teacher states before/after, exact per-parameter optimizer counts,
checkpoint round trips, changed student groups, frozen evaluation state,
saved ranks, source/input hashes, no TEST access and the complete schedule.

```sh
/mnt/geocore/geocore/.venv/bin/python -m unittest biokg.test_candidate_focus biokg.test_joint_operator biokg.test_mixed_operator
/mnt/geocore/geocore/.venv/bin/python -m biokg.train_candidate_focus --smoke
/mnt/geocore/geocore/.venv/bin/python -m biokg.train_candidate_focus --out biokg/results/candidate_focus/s0
```
