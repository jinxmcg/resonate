# H33 execution and observability addendum

Written 2026-09-05 before any H33 training, after the user rented Vast instance
49991229 and requested the fixed experiment plus minute-by-minute LR and MRR.
That initial host's setup was cancelled before training because of severe
network packet loss. The user replaced it with instance 49992742 in Sweden
(212.85.84.41); all six runs will use this replacement. No model-training or
validation result was produced on the first host. Its setup processes were stopped.
The original H33.md is preserved unchanged. This addendum supersedes only its
local GTX 1080 Ti execution location and no-progress-probes setting, not the
six-run optimization recipe or decision gate. The user subsequently requested
a self-contained uv script for dependency installation and execution.

All six fresh runs use the same rented RTX 5090, PyTorch 2.11.0+cu128 and CUDA
12.8 build in a uv-managed environment. Record the resolved environment before training. Do not copy
the local Pascal CUDA environment. Transfer only the existing official OGB
processed dataset, train.pt, valid.pt, release metadata and name mappings;
do not transfer or open test.pt. No checkpoint from previous experiments is
transferred. The model, optimizer, objective, sampling and schedule are unchanged.

Progress output adds the two optimizer learning rates (Adam, RowAdagrad), for
the next update after the logged step, at the existing 1000-step log interval.
MRR progress uses the existing read-only validation probe every 5000 steps,
on the same 5000 validation triples (10000 directed queries), selected by its
unchanged private NumPy seed 0, with the official 500 supplied negatives.
Label these subset scores as progress probes, not full-validation results.
The probe runs under no_grad and restores training mode. It does not feed
gradients, training examples, weights or candidates into training. The same
fixed probes apply to every seed/arm. A paired toy-training test compares
probed and unprobed runs and checks exact final weights and Torch RNG state.

The primary comparison still uses full validation on final frozen checkpoints.
No probe-driven optimization changes, adaptive stopping, intermediate checkpoint
selection, or test evaluation are permitted.
All six seeds/arms and the original follow-up gate remain fixed. Technical
failures are recorded rather than silently discarded or reinterpreted.
