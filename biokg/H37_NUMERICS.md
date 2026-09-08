# H37N: repair normalization underflow, retain the H37 experiment

2026-09-07. H37 passed its algebra/gradient/cost pilot, then the real-data
fail-fast guard stopped before any held-out evaluation. A bounded identical
TRAIN-only replay reproduced nonfinite entity gradients at step1426 in the
squared arm: 288 NaNs in one entity row; H/tau gradients and all weights were
finite. The source row's squared norm rounded to zero in fp32. Scores stayed
finite, but differentiating sqrt(sum(abs(x)^2)) at this zero produced invalid
backward values. No MRR probe or endpoint was reached.

This is a numerical repair and fresh restart, not a hyperparameter sweep.
Preserve all H37 sources, protocol, pilot and failed-run receipts unchanged.
Replace normalization in BOTH arms, only, by:
`x / (sqrt(clamp_min(sum(abs(x)^2), finfo(real_dtype).tiny)) + 1e-8)`.
For fp32 sqrt(tiny) is about1.08e-19, far below an ULP of the existing1e-8
denominator epsilon. Test bit-identical fp32 forward outputs on zeros, very
small rows and normal rows; finite backward where the old expression fails;
ordinary-row gradient parity and explicit zero-row derivative1/eps. Confirm
the recorded failing batch's scores/loss are unchanged with finite gradients.
No change to squared epsilon1e-6, learning rates, trajectory, negative policy,
typed exact denominator, parameter counts, initialization or update budget.

After the reproducer/tests pass, repeat the mandatory H37 GPU cost pilot, then
train a fresh matched 50k pair on the same fit split, using the same fixed
H37 primary gate. Do not resume the failed state, select a partial checkpoint
or use any held-out information to choose this repair. Same rules/provenance
and development-holdout limitations as H37.md. No official VALID/TEST, external
weights, distillation, ensemble or submission change.

The exact loss remains the original H37 implementation. A documented runner
adapter changes ONLY model factory/restore/snapshot, source receipt list and
experiment identity; it reuses the original finite training/evaluation/audit
functions without editing their source. Every original and adapter file is
hashed into the new receipt. Run in a fresh process; separate outputs prevent
overwriting the failed attempt. Unit tests cover adapter wiring and checkpoint
identity. The old spec's numerical hyperparameters and the new spec agree.

Remote root stays /workspace/biokg_h37; new finite supervisor job
biokg_h37_stable, autorestart false; new output biokg/results/h37_stable/s0,
pilot biokg/results/h37_stable/pilot.json. Same pinned uv environment. Preserve
and back up original failure logs/diagnostic state separately. If the repaired
recipe fails, stop and report; no additional numerical/objective sweep.
