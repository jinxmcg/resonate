# TF1: one TRAIN drug–drug failure, read-only

User clarified on 2026-09-07 that the desired next task is forensic inspection
of an actual training error, not another scoring intervention.

Use only our frozen H35F k12 single checkpoint and official TRAIN/node counts.
Reject VALID, TEST, external checkpoints and prediction/feature caches. CPU,
four threads, LR=0, no gradients, optimizer, parameter edits or submission.

Choose 128 TRAIN drug–drug rows without replacement with RNG seed3681;
inspect tails then heads, scoring every drug with the existing model API.
Report raw ranks and TRAIN-filtered ranks. Filter OTHER known positive
same-relation drug pairs, including reverse copies, but retain the focal
answer. An unobserved competitor is NOT a proven biologically false edge.
This full-catalog TRAIN diagnostic is not official validation MRR and does
not evaluate the retrieval-augmented submission.

Select the first sampled filtered near miss (rank>1 and <=10); if absent,
select the first remaining filtered failure. If none exist, stop. Do not
search additional seeds or pick an example for a preferred explanation.

For that query inspect top candidates, true-answer and competitor scores,
candidate norms, normalized angular alignment, exact symmetric norm/angle
margin decomposition, and every 4x4 block's score contribution. Reconstruct
the existing logit to tolerance. Also inspect the same model's reverse score
and diagnostic unit-target-norm ranking; these are local explanations, not
new recipes or measured improvements. Temperature cannot change ordering.

For TRAIN-neighbor evidence remove the focal drug pair in both orientations
from the same-relation graph. Compare source holders, candidate analogues
and neighbor-set overlap for the true answer and competitor; never credit
direct membership of the inspected TRAIN label as predictive evidence.
Read the existing training sampler/objective code for plausible mechanisms,
but do not claim a cause of learning failure without a controlled experiment.

Save deterministic sample/index/ranks, one case, source/input/model hashes and
verification checks. Record limits: in-sample, selected example, incomplete
TRAIN-only positive filtering, no representative generalization claim.
