# H37: exact typed squared compatibility

2026-09-07. [Fixed protocol](../../H37.md). Status: original attempt stopped
on nonfinite gradients before its first MRR probe; no held-out evaluation.

Same free 4x4 operators and k12 entity table, 27,124,129 real parameters each.
Fresh matched seed-0 pair, 50k updates each, source/positive batches shared.
Control: sampled CE + .1 trajectory. Squared: exact eligible-type conditional
likelihood of epsilon+(tau*dot)^2 + .1 trajectory, epsilon fixed 1e-6. Both use
Adam .005 / RowAdagrad .3, cosine to zero. This changes scorer, objective and
candidate coverage together, not an activation-only ablation or ComplEx2
reproduction. The Gram matrix is a fresh differentiable computed statistic,
not a new learned matrix. Full eligible-table gradients are intentionally
dense, including cold-in-fit entity IDs; no held-out labels are accessed.

52 local tests passed (17 H37 and 35 NL1/MN1/TH1). One initial fp32 loss
comparison exceeded an overly strict tolerance by 8.6e-6: investigation found
GEMM versus paired positive-dot reduction order, amplified near a zero score.
With the positive numerator shared the loss difference was zero; denominator
relative errors were about 2.5e-7. The independent float64 probability/gradient
checks passed at tight tolerance. The full fp32 independent comparison now
uses a documented 2e-5 absolute / 2e-6 relative loss tolerance and still checks
all gradients. No model formula or optimizer was altered to address this.

Remote /workspace/biokg_h37, finite supervisor program biokg_h37 on authorized
Vast instance 50109054, RTX5090, 79.19.114.194:50034. uv-pinned environment,
private dependency cache reuse, no auto-restart. Only TH1 derived fit/holdout
arrays and metadata deployed; no old models or official VALID/TEST.

Fixed pilot gate: worst of two 20-pair timing blocks <=20ms/pair and peak
allocated memory <=8GiB, finite state and direct/Gram GPU gradient agreement.
If viable, fixed training endpoints only; MRR on the fit probe is progress,
not selection. Frozen all-pair internal holdout gate is squared minus linear
>=.001 with a positive lower paired 95% bound. No automatic sweep or promotion.
Pilot passed: 32 remote tests, independent direct/Gram GPU probability and
gradient checks; timing blocks 7.05535 and 7.19272 ms/pair, peak allocation
595,214,336 bytes. Synthetic largest eligible type 45,085, no real data loaded.
The original run stopped before step2000; a bounded unchanged TRAIN-only replay
identified the failure at step1426 in the squared arm. One source row's squared
norm underflowed to zero in fp32. All weights, queries, loss and H/tau gradients
were finite, but its entity gradient contained288NaNs. Recorded loss14.31043,
tau0.01077084, AdamLR0.004989986, tableLR0.29939916. The replay opened fit and
manifest only; no recipe change or held-out evaluation. Failure-state checkpoint
is diagnostic only, not an endpoint or restart source.

The normalization implementation has a numerical edge case at tiny norms;
this attempt does not supply an MRR comparison or show squared likelihood is
inferior. A separately versioned H37N normalization repair and fresh restart
is gated on identical failing-batch forward scores/loss, finite gradients and
the same tests/cost checks. All original H37 sources/receipts remain unchanged.
[Repair protocol](../../H37_NUMERICS.md). The separately recorded repair retry
has now completed: linear internal-holdout MRR0.6204363194 versus squared
0.5741653145, delta−0.0462710049; overall gate failed. Drug–drug improved, but
the other relations lost more in aggregate. See the [completed H37N result
and audit](../h37_stable/README.md) for the full comparison and caveats. Both
attempts are backed up locally. Vast instance50109054 was stopped at the
user's request, with disk retained. Existing submission remains unchanged.
