# RS1: reverse scoring of drug–drug candidates

Status: completed and audited; small positive screen, below continuation gate.
See `biokg/REVERSE_SCORING.md` for the frozen protocol.

One own frozen H35F k12 checkpoint. Add its learned reverse-direction score
to the complete audited CS1 pipeline on drug–drug only. LR=0, CPU only,
no new model parameters, gradients, graph construction or TEST access.
Five fixed alpha choices, selected on the existing fit half only.

First seed0/fold0 report control: 0.8578497365986548 MRR. Stop unless the
complete-pipeline report gain is at least +0.0005 and the descriptive
paired-triple bootstrap lower bound is positive. If it passes, complete
the remaining five folds. Repeated VALID use and this screening decision
limit inference; no automatic submission change or untouched-holdout claim.

The original full three-partition pipeline aggregate is 0.8582166512116752.
Do not compare a single report-half result directly against that aggregate.

## Result (seed0/fold0 only)

Complete pipeline 0.8578497366 -> 0.8581763191; delta +0.0003265825,
descriptive paired-triple 95% interval [+0.0001030824, +0.0005505058].
Drug–drug 0.7004866965 -> 0.7021855994. Finite TRAIN-independent VALID fit
selection chose alpha .05 for all 38 tail groups and zero for all head groups,
via family/direction fallback. Global alpha was zero.

Recovered 591 top-one answers (all formerly ranks >1 through 10), lost 513,
net +78 on 162,398 report queries. Other families and head ranks unchanged.
Reverse-only drug–drug report MRR .6463750361 is descriptive, not the gate.

Stopped after this first fold: +.0003266 is below the fixed +.0005 minimum,
despite the positive descriptive interval. No remaining folds evaluated,
aggregate fabricated, new recipe promoted or submission changed. The result
suggests limited complementarity, not proof that reverse scoring is useless.

84 synthetic tests passed. Main run 79.36 seconds on CPU. Endpoint audit
passed: 76 direct reversed-model queries ×501 candidates, maximum absolute
fp32 difference 1.52587890625e-5 within combined atol/rtol=1e-5; all normalized
features, fit-only recipe, baseline/treatment ranks, coverage and statistics
reproduced. Model/source/input/artifact hashes verified; gradients absent.
TRAIN bytes were only hashed for provenance; VALID and own model deserialized;
no TEST or external trained artifacts. GPU unused, LR=0.

User subsequently clarified that the intended next step was a forensic
investigation of one TRAIN error; that is tracked separately as TF1.
