# TH1: fixed internal TRAIN holdout

Created 2026-09-07. [Protocol](../../TRAIN_HOLDOUT.md),
[builder/fit-only loader](../../train_holdout.py),
[tests](../../test_train_holdout.py).

Seed 36840, existing SplitMix64 pair buckets: 0–94 fit, 95–99 reserved.
Unordered **global entity pairs** stay together across every relation, reverse
copy and duplicate row. No seed search, row removal or coverage rebalancing.
Official TRAIN is unchanged; these are separate derived files.

| Part | Original TRAIN rows | Row percentage | Distinct unordered pairs |
| --- | ---: | ---: | ---: |
| Fit | 4,526,393 | 95.0388% | 2,114,965 |
| Reserved holdout | 236,285 | 4.9612% | 111,276 |
| Total | 4,762,678 | 100% | 2,226,241 |

Pair holdout fraction is 4.9984%. All 51 relations occur in both parts. No
pair overlaps between parts, and every original row occurs exactly once.
Rare-relation coverage is reported without backfilling: relation 9, for
example, has only four held-out rows. Preserve this limitation when reporting
future per-relation results.

There are 781 entities present in holdout but absent from fit edges. They occur
in 1,112 held-out rows (0.4706% of holdout rows). These rows are retained;
do not silently drop them or regenerate the split. Node IDs/counts remain
available, but a plain learned embedding has no positive training exposure for
these nodes. Two original self-link rows are both in holdout and are retained.
This conservative pair split is not the official OGB validation distribution.

## How to use it

```python
from biokg.train_holdout import load_fit

train, offsets, counts = load_fit("biokg/results/train_holdout/s0")
```

This returns the familiar local head/tail, relation and type fields from
**fit_train.npz only**. It verifies its hash and does not open official TRAIN
or the holdout artifact. It was tested without any holdout file present.
Future trainers need an explicit fit-only entry point and file-access guard;
legacy full-TRAIN runners have not been silently changed by this reservation.

- Build students, teachers, graph features, indexes, training-negative filters
  and learned preprocessing only from fit. Do not warm-start from the current
  all-TRAIN models or distill from the old all-TRAIN teachers for this experiment.
- Mask-and-reconstruct training targets come from **fit**, with their own
  target links/reverse/duplicate copies excluded from graph inputs as specified
  by the future experiment. The reserved part is never a learning target.
- Freeze a training recipe, endpoint and evaluation candidate procedure before
  measuring holdout predictions. No candidates or holdout metrics exist yet.
- If its results later guide repeated development, call it an internal
  development set, not untouched confirmation. Historical exploratory work
  already used official TRAIN; this reservation cannot undo that history.

## Verification and status

136 synthetic/regression tests passed. Build completed in 15.46 seconds on CPU.
No model loaded, training started, GPU used or VALID/TEST accessed. Only TRAIN
and node counts opened from the dataset; input and source hashes were checked
before/after. The separate-process endpoint audit passed: both saved parts,
pair assignments, coverage and fit-loader output exactly reproduced from
official TRAIN, with all hashes verified. See [endpoint_audit.json](s0/endpoint_audit.json).

Prior single model and pipeline are untouched. No new MRR; previous pipeline
VALID MRR remains 0.8582166512. This step reserves data only, not a trained
baseline or an implemented masked-neighborhood learning objective.

Artifacts: [manifest and complete coverage](s0/manifest.json),
[fit arrays](s0/fit_train.npz), [reserved arrays](s0/holdout_train.npz),
[row-to-pair assignment](s0/assignment.npz), [prerun receipt](s0/prerun.json),
[builder audit](s0/audit.json), [endpoint audit](s0/endpoint_audit.json).
