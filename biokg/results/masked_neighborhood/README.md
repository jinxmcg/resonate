# MN1: masked-neighborhood single-model pilot

2026-09-07. [Fixed protocol](../../MASKED_NEIGHBORHOOD.md).
Status: completed; failed the preregistered advance gate. Ran on the
user-provided Vast RTX 5090, instance 50109054,
direct SSH `root@79.19.114.194 -p 50034`, isolated `/workspace/biokg_mn1`.
Finite supervisor program `biokg_mn1` exited successfully, automatic restart
disabled. No follow-on training was launched; the GPU job is finished.

Single fresh k12/4x4 model, 50,000 fit-only steps. Fork independent copies of
its weights AND optimizer states; 12,500 matched ordinary versus masked-context
continuation updates. No teachers, distillation, extra model parameters,
inference-time context, external models or official VALID/TEST data.

Only the TH1 fit/holdout derived arrays and metadata were transferred. Training
has a file-open guard that denies holdout; only the separate frozen endpoint
evaluation may read it. Fit-only TRAIN probes are progress diagnostics, not
held-out generalization scores or promotion criteria.

The first masked context is eight other same-relation drug–drug answers, with
the target removed before sampling. A training-only assisted query mixes 25%
normalized neighbor context with the normal query; its CE has 25% weight,
with 75% ordinary CE and unchanged trajectory. No-context rows and all other
relation families keep the original loss. Existing unfiltered negative sampling
is retained in both arms. This tests transfer into the plain scorer, not a
general claim about every masked-graph architecture or false-negative solution.

Prelaunch verification: 151 local synthetic/regression tests, 15 new remote
tests and full-size synthetic GPU smoke passed. The smoke ran 20 paired updates
in 0.20235 seconds, peak allocated memory 521,960,448 bytes, with 27,124,129
real model parameters and no real data loaded. This synthetic speed is not
a promise for the real graph, data sampling or endpoint evaluation.

uv pins Python 3.12, torch 2.11.0+cu128, numpy 2.5.2 and scipy 1.18.1. The image
defaults to `UV_NO_CACHE=1`; only this job overrides that with a private uv cache
so its phases reuse the environment. No system driver changes or other service
changes. Remote storage is not volume-backed. All three checkpoints, progress,
rank arrays, receipts and audits have been copied into local `s0/`; source,
data and artifact hashes were verified against the receipts. The supervisor
log is also backed up here. The rented instance itself was not stopped or
destroyed by this experiment.

## Endpoint result

All 236,285 reserved rows, both directions: 472,570 queries. These are internal
unordered-pair-holdout scores, not official BioKG validation or leaderboard MRR.
Each query has 500 identical typed sampled negatives across the three models;
known fit answers, source and designated positive are excluded from negatives.
Other held-out labels are not used to filter candidates, so unknown false
negatives can remain. Every candidate column uses the same plain scorer.

| Frozen endpoint | Internal holdout MRR | Hits@1 | Hits@10 |
| --- | ---: | ---: | ---: |
| Fresh base, 50,000 steps | 0.6204363194 | 0.4948579046 | 0.8896967645 |
| Ordinary continuation, +12,500 steps | 0.6182218144 | 0.4927989504 | 0.8888757221 |
| Masked continuation, +12,500 steps | 0.6173322471 | 0.4924434475 | 0.8878875087 |

Primary masked minus continued-control MRR: **-0.0008895674**, pair-cluster
bootstrap 95% interval **[-0.0009996977, -0.0007779612]**. The advance gate fails.
Control minus base is -0.0022145050; masked minus base is -0.0031040723.
This interval describes held-out pair resampling, not training-seed uncertainty.

Drug–drug (110,520 queries) MRR: base 0.2758951508, control 0.2703584013,
masked 0.2661412842. Masked minus control is -0.0042171171 on the intervention's
target family. The 2,224 cold-endpoint queries score 0.0921839483,
0.0911695298 and 0.0893014615 respectively; they remain in the primary metric.

This exact training-only neighborhood mixture did not transfer a benefit to
the ordinary scorer. Both continuations lost to the base, so the experiment
does not establish that more training or larger relation blocks are needed.
It does not reject all masked-neighborhood architectures or resolve TF3's
known-positive negative-label issue. No sweep or submission change follows.

## Timing and completed audit

Training, loading, probes and checkpoints: 259.06 seconds. Frozen evaluation:
27.07 seconds. Full candidate/rank replay audit: 26.45 seconds. About 5 minutes
13 seconds combined, excluding dependency setup and process-start overhead.
Peak PyTorch allocated training memory: 0.5773 GiB (not total device memory).

Training receipts confirm identical initial fork weights and optimizer state,
independent storage, retained accumulators, matched batch/negative streams,
finite state and the recorded cosine LR schedules. Target exclusion was
checked on every context batch; the training process did not open holdout.
There were 6,151,934 valid context row occurrences and 49,215,472 sampled
context slots. Fit probes never selected checkpoints.

The read-only endpoint auditor regenerated every evaluation candidate stream,
rescored all three frozen checkpoints, matched ranks/hashes and reproduced
the summary and pair-cluster bootstrap. Evaluation changed no model and used
no gradients. This is an endpoint replay, not a replay of every optimizer step.
See [summary](s0/summary.json), [training audit](s0/training_audit.json),
[evaluation audit](s0/evaluation_audit.json) and [endpoint audit](s0/endpoint_audit.json).

Previous official-VALID development pipeline value 0.8582166512 is unchanged
and is not the reference score for this internal split. No teachers, external
weights, official VALID or TEST were used.
