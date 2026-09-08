# NL1: same 4x4 storage, nonlinear function

2026-09-07. [Fixed protocol](../../NONLINEAR_OPERATOR.md).
Status: completed; failed the fixed advance gate. No follow-on sweep or
submission change. The finite GPU job exited successfully.

Compare fresh linear and nonlinear single models, identical seed-0 initial
weights, batches, loss and optimizer schedule, 50,000 updates each on TH1 fit.
Nonlinear f(z)=z/(1+|z|), coordinatewise before global hop normalization,
retained at inference. Same k12, 36 4x4 complex blocks per directed relation,
same 27,124,129 real parameters. No extra model, teacher, graph context or KD.

Fixed all-holdout MRR gate: nonlinear minus linear >= .001 and positive lower
pair-cluster 95% bound. No interim holdout scores or checkpoint selection,
no alpha sweep, official VALID/TEST or submission changes. This previously
observed internal TRAIN holdout is development data, not pristine confirmation.

35 local tests passed, including 12 new activation/model tests and inherited
MN1/TH1 tests. Zero-alpha original loss/gradient parity, explicit real-arithmetic
complex formula, complex128 numerical gradient check, phase retention, finite
zero/large inputs, independent storage, sparse updates, candidate symmetry,
checkpoint activation preservation, private evaluation RNG and summary checks.
All 27 deployed NL1/MN1 tests passed in the remote uv environment. Full-size
synthetic GPU smoke passed: 20 paired updates in 0.11978 seconds, peak PyTorch
allocation 448,818,176 bytes, no real data. This is not a real-data ETA.

Authorized Vast RTX 5090 instance 50109054, direct SSH 79.19.114.194:50034.
Isolated /workspace/biokg_nl1; finite supervisor program biokg_nl1, no automatic
restart. uv-pinned environment reuses the existing private dependency cache.
No other instance/job was touched. Only TH1 derived arrays/manifest were inputs;
no old checkpoints were transferred. Both standalone checkpoints, all receipts,
rank arrays, progress and the supervisor log are now backed up locally. Source,
data and artifact hashes were verified against the remote receipts. The GPU
is free; the rented instance itself has not been stopped or destroyed.

## Fixed endpoint result

Internal TRAIN pair holdout, 236,285 rows in both directions, 472,570 queries.
These are not official BioKG validation/leaderboard scores. The same fixed
500-negative candidate stream and known-fit-answer filtering were used in both
arms; other held-out answers were not consulted for filtering. Unknown false
negatives can remain. No held-out labels entered either model's training.

| Model, 50,000 updates | Internal holdout MRR | Hits@1 | Hits@10 |
| --- | ---: | ---: | ---: |
| Linear, original 4x4 function | 0.6204363194 | 0.4948579046 | 0.8896967645 |
| Nonlinear, same 4x4 parameters | 0.6194131842 | 0.4939501026 | 0.8886112110 |

Primary nonlinear minus linear: **-0.0010231352 MRR**, pair-cluster 95%
interval **[-0.0015737196, -0.0005040952]**. Gate failed. This is one training
seed; pair resampling does not measure seed uncertainty.

Drug–drug (110,520 queries): linear 0.2758951508, nonlinear 0.2721240596,
delta -0.0037710912. Cold-endpoint slice (2,224 queries): 0.0921839483 versus
0.0940664498; descriptive only, not a reason to override the failed primary.
Seen endpoints (470,346 queries): 0.6229341259 versus 0.6218972515.

Final in-sample fit-probe MRR was 0.8825787593 linear versus 0.8812469370
nonlinear. Earlier probe advantages varied and did not persist. No probe
selected an endpoint or altered the schedule.

## Did the function actually change?

Yes. Synthetic tests verify different operator gradients with identical
initial weights. At the final nonlinear fit probe, preactivation magnitude
10/50/90 percentiles were 0.04561 / 0.13802 / 0.32496; attenuation percentiles
were 0.75474 / 0.87872 / 0.95638. Mean cosine between the nonlinear query and
the hypothetical linear query from the SAME nonlinear weights was 0.99019.
This is not a comparison with the separately trained linear model. Mean block
Frobenius norm was 3.29118 nonlinear versus 3.35768 linear: saturation did not
vanish by scaling every block close to zero. These diagnostics do not explain
causally why MRR fell, nor identify the semantics of individual coordinates.

The specific phase-preserving saturation function changed learning but did
not improve held-out ranking. This rejects this fixed version, not every
possible nonlinear function or the same-storage research direction. Do not
launch an alpha/initialization/objective sweep from this negative result.

## Verification and reproducibility

35 local NL1/MN1/TH1 tests plus 29 broader legacy regression tests passed
(64 total); the latter required their established `biokg` import path. The
first broad invocation had three import-path errors, not numerical failures;
rerunning with that path resolved them without source edits. All 27 deployed
tests and the full-size synthetic GPU smoke passed before training.

The endpoint audit regenerated the common random initial weights and optimizer
state, verified parameter counts, alpha configuration, optimizer counters and
accumulators, and replayed the recorded LR schedule. It regenerated every
held-out candidate stream, rescored both frozen models, matched all ranks and
reproduced the summary/bootstrap. Model hashes were unchanged during scoring;
training opened only fit_train.npz and manifest.json. This is not a replay of
every optimizer step.

Additional local cross-experiment check: NL1's linear endpoint model AND
optimizer digests exactly match the previously frozen MN1 base. Training batch
and evaluation candidate stream hashes also match. This confirms control
reproduction, not a second independent seed. No MN1 checkpoint entered NL1
training; the comparison was made only after completion.

Training/loading/probes/checkpoints: 330.75 seconds. Evaluation: 26.08 seconds.
Endpoint replay audit: 25.52 seconds. About 6 minutes 22 seconds combined,
excluding setup, process-start and backup overhead. Peak PyTorch allocated
training memory: 427,863,040 bytes (about 0.3985 GiB), not total GPU usage.
Both optimizer LRs ended at zero. Existing official-VALID development pipeline
0.8582166512 remains unchanged and is not comparable to this internal metric.

Receipts: [summary](s0/summary.json), [training audit](s0/training_audit.json),
[evaluation audit](s0/evaluation_audit.json), [endpoint audit](s0/endpoint_audit.json).
