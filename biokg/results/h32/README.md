# H32 result: random auxiliary helps; head/tail skew adds almost nothing

Completed 2026-09-05. Four fresh runs, one fixed seed (0), 12,500 steps
each. These are early, undistilled single-model screens, not full-length
or submission scores. The pre-run design remains unchanged in
[`../../H32.md`](../../H32.md); its original hash is in `prerun.json`.

## Primary result

| Arm | Validation MRR | Delta vs baseline | Hits@1 | Hits@10 | Training time |
|---|---:|---:|---:|---:|---:|
| baseline | 0.7825889281 | — | 0.7044773645 | 0.9253005169 | 160 s |
| random auxiliary | 0.7860282071 | +0.0034392790 | 0.7088730769 | 0.9267094778 | 225 s |
| head/tail skew | 0.7827879044 | +0.0001989763 | 0.7046154980 | 0.9262183368 | 168 s |
| random auxiliary + skew | 0.7860861634 | +0.0034972353 | 0.7086888990 | 0.9277071080 | 226 s |

Adding skew to random auxiliary improves MRR by only **0.0000579563**,
while Hits@1 falls by 0.0001841779. Skew alone improves baseline by
0.0001989763. Both fail their pre-stated +0.003 follow-up margins. The
factorial interaction is -0.0001410200 MRR: no positive interaction was
observed in this screen. This is not a statistical test of zero effect.

Conclusion: random auxiliary accounts for almost all of the observed
improvement. Do not adopt this skew or launch a cap/direction sweep on
these results. Keep uniform direction sampling for the proposed
50,000-step, seeds 0/1/2 confirmation of random auxiliary versus baseline.
That longer comparison has **not** been run. No longer campaign or
submission change was made by this experiment.

Both uniform controls are bitwise identical to their H31 model tensors.
This verifies backward compatibility and repeatability with the same
seed/environment; it does not supply independent seed evidence for the
random auxiliary gain. Its full-training/distilled-model value remains
unknown. Both new features remain off by default.

## Where the skew helps and hurts

Overall direction averages:

| Prediction direction | Baseline | Random | Skew | Random + skew |
|---|---:|---:|---:|---:|
| head | 0.7728638883 | 0.7776759534 | 0.7744937628 | 0.7788691968 |
| tail | 0.7923139678 | 0.7943804608 | 0.7910820460 | 0.7933031300 |

The skew raises head-prediction MRR but lowers tail-prediction MRR on
average. On top of random auxiliary, these shifts are +0.0011932434
and -0.0010773309, respectively. They nearly cancel in the equally
weighted official two-direction metric.

Pre-specified bottleneck directions:

| Relation and predicted endpoint | Baseline | Random | Skew | Random + skew |
|---|---:|---:|---:|---:|
| protein-function, head (protein) | 0.6361665714 | 0.6511190792 | 0.6518167437 | 0.6651183530 |
| protein-function, tail (function) | 0.8727745400 | 0.8738384205 | 0.8687810037 | 0.8695791113 |
| drug-sideeffect, head (drug) | 0.3528666425 | 0.3528468963 | 0.3473476763 | 0.3478945970 |
| drug-sideeffect, tail (sideeffect) | 0.1268397285 | 0.1283601287 | 0.1325314138 | 0.1325829729 |

Skew helps the two oversampled bottleneck directions, but this is not a
reason to promote it using only those slices. The complete family table
shows losses elsewhere, including families whose sampling stays 50/50
but which share model parameters and entity embeddings:

| Family (both directions) | Baseline | Random | Skew | Random + skew |
|---|---:|---:|---:|---:|
| drug-drug (38 relations) | 0.6444463797 | 0.6470922950 | 0.6424538213 | 0.6449098244 |
| protein-function | 0.7544705557 | 0.7624787499 | 0.7602988737 | 0.7673487322 |
| drug-sideeffect | 0.2398531855 | 0.2406035125 | 0.2399395451 | 0.2402387850 |
| function-function | 0.9373279252 | 0.9394203893 | 0.9370550226 | 0.9395036366 |
| disease-protein | 0.5455043979 | 0.5487220563 | 0.5332653322 | 0.5360641658 |
| drug-protein | 0.7872374259 | 0.7904147054 | 0.7745318337 | 0.7803732757 |
| protein-protein (8 relations) | 0.9530056110 | 0.9532683302 | 0.9524587952 | 0.9526929169 |
| drug-disease | 0.2613854694 | 0.2672406262 | 0.2552709031 | 0.2575307391 |

## Sampling and verification

The policy altered only five asymmetric relations; 46 symmetric relations
remained at 50/50. Expected overall tail probability was 0.4851638 and
observed tail fraction was 0.48496. Both skew arms had exactly the same
policy and per-relation direction counts, totaling 12,500 batches each.

| Relation | Expected tail probability | Actual tail batches | Actual head batches |
|---|---:|---:|---:|
| disease-protein | 0.666667 | 148 | 78 |
| drug-disease | 0.650897 | 5 | 7 |
| drug-protein | 0.666667 | 199 | 104 |
| drug-sideeffect | 0.666667 | 269 | 152 |
| protein-function | 0.333333 | 677 | 1372 |

The rare drug-disease relation received only 12 sampled batches. A fixed
sampling probability does not guarantee that every relation realizes that
fraction in one finite run. No per-relation minimum or extra exposure
was introduced after observing this.

All 32 unit/integration tests passed, including the GPU-enabled run.
Checks covered direction conventions, exact edge-weighted fanouts,
deduplication, caps, symmetry/reversal, RNG isolation, train-only policy
inputs, all four training combinations, finite sparse gradients and
unmodified evaluation. All four checkpoints have 27,124,129 real
parameters, matching model keys and finite tensors. Saved common arguments
match; only output path, auxiliary mode and direction mode differ.

Both auxiliary runs filled all 1,310,720,000 requested candidate slots.
The auxiliary index and direction policy share the TRAIN digest
`ac4d083d3c5104e605eb2b38563778dc4e57bf799573b39b028552606a4a038b`.
The 697.4 MiB training index and tiny CPU direction policy add no model
parameters or inference operations. Random auxiliary costs about 41%
more training time than baseline here; the comparison fixes steps, not
wall-clock compute. The whole campaign, including loading, diagnostics
and receipt generation, took approximately 16 minutes 15 seconds.

The source/preregistration hashes were unchanged throughout the run.
No test split was loaded. Neither validation edges nor error diagnostics
entered gradients, the direction policy or auxiliary candidate selection.
Evaluation scored all 325,772 directed validation queries against the
original 500 typed candidates, without training masks or reweighting.
OGB's tie convention was verified per query; random had one tied query,
the other arms had none. Prior experimental history remains a separate
disclosure issue; this screen is not a certification of a submission.

## Artifacts

* `prerun.json`: frozen code/document hashes, paths, environment and start time.
* `screen_s0.json`: all exact metrics, effects, gates, model/graph hashes,
  complete direction counts and grouped diagnostics.
* `{baseline,random,skew,random_skew}_s0.log`: copied original training logs.
* `../../runs/h32/{baseline,random,skew,random_skew}/model.pt`: local ignored checkpoints.
* `../error_analysis/h32_{baseline,random,skew,random_skew}_s0/summary.json`:
  independent exact summaries, with local ignored per-query CSV/NPZ files.
