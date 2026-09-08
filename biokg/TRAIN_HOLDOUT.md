# TH1: reserve a small internal TRAIN holdout

2026-09-07. User approved reserving a small TRAIN part for the proposed
hide-a-link/predict-it experiment. This step only creates and audits the split;
it does not train, score a model, select an architecture or change a submission.

## Fixed split, before reading outcomes

Reserve approximately 5% of unordered **global entity pairs**, across all
entity types and all relations. Global IDs use the official node counts and
the existing sorted-type offsets. Canonical key is `min(h,t)*N + max(h,t)`.
Use the existing TP1 SplitMix64 bucket implementation with fixed seed 36840;
buckets 0–94 are fit-TRAIN, 95–99 are reserved holdout-TRAIN. Do not search
seeds, backfill small relations, rebalance by performance or drop cold nodes.

Every original TRAIN row is retained exactly once in its assigned part.
Duplicate rows, reverse copies, alternate relations and self-links on the
same pair receive the same assignment. This conservative pair holdout is
stricter than the official edge split. Multi-hop evidence through other pairs
remains available. Actual row percentages may differ from 95/5 because pairs
have different numbers of rows. Record row/pair/relation/type coverage and
entities present only in holdout, without changing the assignment.

## Files and future use

Do not edit official TRAIN. Write a derived `fit_train.npz` and separate
`holdout_train.npz`, containing local IDs, relation IDs, compact type IDs and
original TRAIN row indices. Node/type metadata is in `manifest.json`.
`assignment.npz` and receipts support an independent-process replay audit.
No pickle is needed for derived arrays. A `load_fit()` helper reads and
hash-verifies only the fit artifact and manifest, not held-out labels or
the full official TRAIN file. Existing full-TRAIN runners are NOT silently
converted; future experiments must explicitly use and enforce this boundary.

All future student parameters, teachers, graph inputs, learned features,
normalizers, candidate indexes and sampling/positive-filter indexes for this
experiment must derive only from fit-TRAIN. Existing students/teachers and
caches trained/built on all TRAIN are not clean baselines, warm starts or
distillation targets for this holdout. Start models from scratch, or use only
artifacts with verified fit-only provenance.

The reserved part is evaluation-only: no training targets, mining, negative
filtering for training, gradient updates, graph features or distillation.
Mask-and-reconstruct learning targets must come from fit-TRAIN, with their
target pair/reverse/duplicate copies hidden from graph inputs for that
prediction. This is separate from withholding evaluation pairs entirely.

Freeze the first experimental recipe, evaluation candidate construction and
training endpoint before looking at holdout prediction metrics. No evaluation
candidates or metrics are generated in TH1. If repeated holdout feedback later
guides decisions, call it internal development validation, not an untouched
confirmation set. This is newly excluded data for future models, not a claim
that these TRAIN edges have never appeared in historical exploratory work.

## Data boundary and verification

Only official TRAIN and node counts may be opened from the dataset. No VALID,
TEST, full-graph raw/processed files, model checkpoints or old predictions.
Pin input and source hashes before deserialization; verify afterward. CPU
only, no optimizer/GPU/model. Keep previous pipeline and score unchanged.

Synthetic checks: reverse/duplicate/cross-relation grouping, typed-ID isolation,
row-order independence, deterministic hash vectors, exhaustive row partition,
invalid schema/ID rejection, holdout exclusion, fit loader not opening holdout,
tampered fit rejection, file-open guard boundaries. Real replay reconstructs
both artifacts and assignments exactly, verifies no overlapping pair keys,
counts/coverage/manifest and source/input/artifact hashes. Stop after audit.
