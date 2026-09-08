# TF3: protected shared hard competitor, TRAIN-only diagnostic

2026-09-07. User approved the next protected shared-competitor diagnostic,
not training or a new architecture. Reuse TF2's16 cases, original2,048-row
TRAIN batches and strongest model competitors. Same own frozen H35F k12/4x4
student and ten original own teachers; verify provenance/state hashes.
Local1080Ti, fp32/complex64, CPU threads4, LR0, autograd only, no optimizer,
parameter edits, finite update, VALID/TEST access, external model or promotion.

## Fixed five arms; equal pool size and matched hard/random multiplicity

- baseline: original TF2 random4,096-draw pool, original loss.
- hard_raw: TF2 forced pool (replace first random draw only if hard absent).
- hard_protected: same hard_raw pool, with candidate-specific CE protection.
- random_raw: replace EVERY hard-ID occurrence in hard_raw with one uniform
  random drug, selected with RNG36830+case_index from typed drugs absent from
  hard_raw and not known positive for the focal query. This preserves pool
  size, positions and duplicate multiplicity exactly.
- random_protected: same random_raw pool, with identical CE-protection rule.

Random control is typed-uniform over eligible drugs, not degree/activity
matched. Record candidate degree and known-positive coverage; differences
cannot be attributed solely to a scalar measure of hardness. No selection
or retry based on gradients. All16 cases and five arms are reported.

For each batch row, use only same-relation TRAIN positives, including reverse
copies of these symmetric drug–drug relations. Mask all occurrences of the
chosen candidate in negative CE columns exactly where it is a known TRAIN
answer. Never mask the designated positive column. Do not mask other existing
known-positive negatives: isolate protection of this candidate, not a revival
of broad H30 positive masking. Original T=2,T² KD and0.1 trajectory remain
unmasked and unchanged for a fixed candidate pool. Thus protection removes
false-negative CE pressure; it cannot guarantee all combined updates preserve
every correct answer. Teacher logits are computed by the original function.

## Fixed probes for target and collateral effects

Probe groups, identical across arms within each case:

1. focal_model: original positive versus TF2 model competitor.
2. focal_pipeline: original positive versus saved TF2 pipeline winner; repeats
   are labeled, not independent observations.
3. background: first16 distinct non-focal sources in original batch order,
   using their sampled TRAIN target and strongest TRAIN-unknown catalog drug.
4. hard_known: first8 distinct non-focal batch sources for which hard candidate
   is a known TRAIN answer. Treat hard candidate as positive and compare with
   that source's strongest TRAIN-unknown catalog drug.
5. random_known: analogous first8 sources for the random control. Empty groups
   remain empty, never padded/fabricated.

All competitor lookup uses the frozen student and full typed drug catalog,
filtering other known TRAIN answers but not VALID/TEST. Unobserved is not
proven biologically false. Save probes before arm gradients; do not choose
them based on treatment outcomes. These probes measure within-relation TRAIN
effects, not all relations or generalization.

For total original/protected loss, calculate negative-gradient directional
derivatives of each probe's temperature-independent positive score, competitor
score, positive-minus-competitor margin, and target norms. Use functional JVP
on entity/operator tensors, with no model edits; separate entity and operator
contributions and sum. Positive margin derivative is locally corrective.
Score/norm derivatives help distinguish discrimination from suppressing an
entity generally. This is Euclidean first-order behavior, not Adam or MRR gain.

Primary diagnostic comparisons: hard_protected versus random_protected for
focal correction and background effects; hard_protected versus hard_raw for
preserving hard_known answers. Also report all baseline/raw controls. Report
per-case group means and equal-case aggregate means rather than treating
duplicated probes across cases as independent. No statistical significance,
confirmation, automatic training trigger or post-hoc subgroup promotion.

## Verification and stopping

Synthetic CPU/GPU tests of duplicate protection, symmetric TRAIN index,
designated-positive retention, same-size random substitution, deterministic
probes, unchanged KD/trajectory, zero masked CE derivatives, functional JVP
versus reverse-mode contractions/finite differences and unchanged tensors.
Reproduce TF2 baseline/hard_raw focal total projections within fp32 tolerance;
independently contract real focal margin gradients in every real arm. Check
KD/trajectory equality for each protected/unprotected pool, and zero direct
CE derivative at protected entries. Hash every own model and input/source
before/after, save pools/probes/results and artifact hashes.

Endpoint saved-record audit reconstructs TRAIN masks/pools/probe identities,
derivative arithmetic and all summaries; directly rechecks frozen-model probe
scores/ranks. It does not claim a full independent GPU-gradient replay.
Stop after this bounded80-arm diagnostic and audit regardless of result.
Do not launch a training pilot, larger-k/8x8 run, VALID evaluation or submission.
Progress at least once per minute with LR0 and unchanged prior VALID MRR
0.8582166512, explicitly not a new score. TRAIN-probe MRR is descriptive only.
