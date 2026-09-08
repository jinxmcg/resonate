# H33: stopped after a negative first full-training pair

The user requested cancellation on 2026-09-05 after the first completed pair
and asked to free the GPU for other work. Supervisor stopped the entire H33
process group at approximately 20:10:56 UTC. A subsequent check found 0% GPU
utilization, 1 MiB allocated and no compute processes. The Vast instance was
left running for the user's other work; no new training was launched.

## Completed full-validation results

| Seed | Baseline MRR | Random auxiliary MRR | Paired difference |
|---|---:|---:|---:|
| 0 | 0.8154589272510676 | 0.8130751635796661 | -0.002383763671401473 |
| 1 | 0.8162369190298121 | Not completed | Not available |
| 2 | Not run | Not run | Not available |

All completed runs trained for 50,000 steps and were evaluated on 325,772
directed validation queries with the official 500 typed negatives and OGB
tie convention. Baseline seed 0 took 227 seconds and random auxiliary seed 0
265 seconds in the training-loop log (including fixed progress probes).
Baseline seed 1 took 211 seconds. Training wall times are not an isolated
hardware benchmark.

Random seed 1 was interrupted after the last logged step 6,000; its last
subset probe was 0.6451 at step 5,000. Those are not final-validation results.
Its partial checkpoint and log are retained, not resumed or substituted for
an endpoint. Seed 2 was never started.

## Interpretation and protocol deviation

The first paired full-training result is negative. It already fails the
pre-stated condition that every paired seed improve. The experiment therefore
does not support promoting this unchanged random-auxiliary recipe.

The planned six-run confirmation was NOT completed. Cancellation was a
user-requested compute-allocation decision after seeing the first pair, not
the pre-stated complete protocol. There is only one completed pair: do not
report a three-seed mean, standard deviation, significance claim or conclusion
that every possible random-auxiliary schedule is ineffective. No aggregate
`confirmation.json` was generated.

The early H32 gain and early positive H33 probes did not persist through the
first full run. This is not a causal explanation of why they disappeared.
No intermediate checkpoint was selected as a winning model.

The original H33.md and pre-launch H33_VAST.md addendum remain unchanged after
launch. The addendum documents the user-requested fixed validation probes:
5,000 validation triples every 5,000 steps, identical across arms, no gradients
or changes to training based on those probes. Test data were not transferred
or opened. All 36 tests passed locally and in the uv-managed remote environment.

## Preserved artifacts

- `seed_0.json`: exact paired metrics, arguments, diagnostics and hashes.
- `baseline_s0.log`, `random_s0.log`: completed pair's training logs.
- `prerun.json`, `h33_environment.json`: frozen source, environment and dataset provenance.
- `../../runs/h33/`: three completed model files plus the interrupted fourth run.
- `../error_analysis/h33_{baseline_s0,random_s0,baseline_s1}/`: all three complete diagnostics.

Completed checkpoint SHA256 hashes were verified after download:

```
55d615403eabc4ce9ada44738cc94c2017553907062df1882680df2b5e2f34e1  baseline_s0/model.pt
3c1892fb944b0e3db810dd177cfa4b2ef87f3b348d437b244f03a383f058f222  random_s0/model.pt
7742b8da0a4729f33fc0b9dbaf0818b1a8082e377ff3b551e5b014878a25813b  baseline_s1/model.pt
```

The old Germany host's setup was cancelled before training because of network
packet loss. All results above were produced on the replacement Sweden host,
Vast instance 49992742, RTX 5090, torch 2.11.0+cu128. File transfers and backups
did not alter the training recipe or inference scores. All earlier artifacts
were preserved; no instance was rented, stopped or destroyed by this agent.
