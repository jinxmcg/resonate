# BC1: fixed blend-selection confirmation

Status: six folds completed 2026-09-06; 50 synthetic tests and independent
saved-record audit passed. No next experiment or submission change launched.
The user approved confirming RF1's half-strength-plus-reselection control.
No new mixture settings or model training. GPU is not used.

[Fixed protocol](../../BLEND_CONFIRMATION.md),
[runner](../../confirm_feature_blend.py), [tests](../../test_blend_confirmation.py),
[auditor](../../audit_blend_confirmation.py).

Three fixed partition seeds (0,1,2), both fit/report directions each. Fit
baseline and exact half-strength treatment independently on identical fit
rows in each fold. Original seed0/fold0 is a reproduction, not fresh evidence.
All triples get one out-of-fit report per partition. Average reciprocal-rank
deltas across directions and partitions within each original triple before
paired bootstrapping. Do not treat overlapping partitions as independent
samples, average scores into an ensemble, or select the best split.

All validation is adaptively reused. This checks partition stability on one
fixed model, not model-seed robustness or untouched-data generalization.
No checkpoint/dataset deserialization, gradients, TEST, full-VALID refit or
submission change. Audited cached features derive from TRAIN only.

## Results

The unchanged method improves in all six fit/report comparisons and all
three complete partitions. It is split-stable on these reused validation
data, but the mean gain falls below the registered +0.001 practical gate.
Keep it as a promising candidate, not an automatically promoted submission.

| Complete two-way partition | Baseline MRR | Half-strength + reselection MRR | Delta |
| --- | ---: | ---: | ---: |
| Seed 0 | 0.8544422552 | 0.8555228741 | +0.0010806189 |
| Seed 1 | 0.8546442182 | 0.8554690595 | +0.0008248413 |
| Seed 2 | 0.8546416213 | 0.8555609968 | +0.0009193754 |
| Mean across all three | **0.8545760316** | **0.8555176435** | **+0.0009416119** |

These pipeline numbers **already include retrieval**. The same frozen A
alone has full-VALID MRR 0.8340991235 in the verified
[FC1 record](../feature_compare/s0_retry1/summary.json). Thus the combined
0.8555176435 is about +0.02141852 above model-only, not a model-only score
to which retrieval's gain can be added again. BC1 reuses the same full
query population through out-of-fit evaluation; it does not rerun A.

Within-original-triple paired bootstrap, 2,000 replicates, seed 3631:
descriptive 95% interval **[+0.0006704451, +0.0012121762]**. The resampling
unit is one of 162,886 unique triples, with heads/tails and all repeated
partition results kept together. There are 325,772 unique directed queries,
not three times as many independent observations. These means average
evaluation metrics, not candidate scores into an inference ensemble.

`consistent_positive=true`; `useful_gain_flag=false`. The lower interval
bound is positive and every partition improves, but +0.0009416 is below
+0.001. Do not move the threshold after seeing results or call this a
failure to improve at all. No new settings were tried during confirmation.

| Seed / fold | Fit triples | Report triples | Baseline MRR | Treatment MRR | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 / 0, original reproduction | 81,687 | 81,199 | 0.8540538580 | 0.8550978918 | +0.0010440338 |
| 0 / 1, reversed | 81,199 | 81,687 | 0.8548283322 | 0.8559453176 | +0.0011169855 |
| 1 / 0 | 81,452 | 81,434 | 0.8539560490 | 0.8550656544 | +0.0011096055 |
| 1 / 1 | 81,434 | 81,452 | 0.8553322354 | 0.8558723755 | +0.0005401402 |
| 2 / 0 | 81,302 | 81,584 | 0.8541671057 | 0.8549642800 | +0.0007971743 |
| 2 / 1 | 81,584 | 81,302 | 0.8551177828 | 0.8561597833 | +0.0010420004 |

Each reported score excludes that row's label from its own recipe selection.
Seed0/fold0 exactly reproduces the original FC1/RF1 recipes and report ranks;
it is not additional independent evidence. Other folds also reuse previously
exposed validation. The bootstrap is conditional on the fitted predictions;
it does not account for adaptive research selection, model-seed variance or
all dependence induced by shared fit data. No claim of pristine holdout,
leaderboard placement or a final single full-VALID-selected recipe.

## What improved

This remains one frozen A model plus the same four deterministic TRAIN
retrieval features. The 0.5 transformation changes which weighted blends the
finite candidate generator proposes. It adds no new model information and
does not imply that simply halving retrieval with old weights helps; RF1
already showed that fixed-old-weights variant loses.

Descriptive direction means across the three partitions:

| Prediction direction | Baseline MRR | Treatment MRR | Delta |
| --- | ---: | ---: | ---: |
| Tail | 0.8483098905 | 0.8498849188 | +0.0015750283 |
| Head | 0.8608421726 | 0.8611503682 | +0.0003081955 |

Both directions improve, predominantly tails. Do not retrofit direction
switches from this breakdown. Mean Hits@1 increases from 0.7988439768 to
0.7998385374. Mean Hits@10 is nearly unchanged, 0.9551588227 to 0.9551393817
(a small decrease, not a universal metric improvement). Per-partition top-one
recoveries/losses: 3,826/3,455; 3,505/3,226; 3,647/3,325.

## Verification, compute and next decision

50 synthetic tests passed: nine BC1 plus the unchanged 41 RF1/FC1/H34/model
tests. Coverage is exactly once per partition/query. Source, cached input
and own checkpoint hashes are unchanged. The checkpoint was hashed as bytes,
never deserialized; no optimizer, gradients, GPU calls or dataset opens.

Through selection, report and bootstrap: 451.91 seconds (~7.53 minutes).
Through final input/artifact hashing: 467.12 seconds (~7.79 minutes).
Both exclude the separate saved-record audit. No GPU use.

Independent audit passed: all twelve recipes were refitted exactly, and the
prior full-row router reproduced all report ranks independently of the new
report-only router. Coverage, original-fold parity, all metrics, clustered
bootstrap and source/input/artifact hashes also passed. No dataset splits
or models were deserialized by the audit.

The fixed confirmation is complete; neither a larger candidate search nor
automatic adoption follows. The useful next research lead is selection of
complementary blends among existing features, with a separately registered
finite comparison. A broader search, model-seed confirmation, full-VALID
refit, TEST or submission update would require a new decision.

[Prereceipt](s0/prerun.json), [fold results](s0/folds.json),
[summary](s0/summary.json), [ranks](s0/ranks.npz),
[runtime audit](s0/audit.json), [independent endpoint audit](s0/endpoint_audit.json).

```sh
/mnt/geocore/geocore/.venv/bin/python -m biokg.audit_blend_confirmation biokg/results/blend_confirmation/s0
```
