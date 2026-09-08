# TF2: TRAIN near-miss objective and existing-pipeline forensics

2026-09-07. User authorized investigation 1, not retraining or architecture
changes. Continue from TF1's fixed 128 TRAIN triples / 256 directed queries.
Select first eight filtered ranks >1 through10 per direction in saved order,
excluding repeated unordered drug pairs across both directions. Include the
original case. If fewer exist, report the actual smaller set; no new sampling.

Use own unchanged H35F single A (k12,144 complex coordinates,36 4x4 blocks)
and its original ten own H24 teachers, with existing provenance SHA256s.
No external trained artifacts. Allow only TRAIN/node-count dataset files,
TF1 TRAIN artifacts, exact pre-existing seed0/fold0 BC1/CS1 JSON hyperparameter
recipes, and the approved own checkpoints. No VALID/TEST edges or prediction
caches, no new selection/fitting of mixture weights, no optimizer or update.
CPU retrieval, local1080Ti gradients, four Torch CPU threads, fp32/complex64.
All parameter values and input/source hashes must remain unchanged. Autograd
is enabled for student inspection only; `.grad` fields stay empty. LR=0.

## Existing retrieval: leave the inspected TRAIN pair out

Rebuild the exact directional same-relation TRAIN graph, not TF1's symmetric
union. Remove all copies of the focal edge and its reverse before every graph
feature. Reuse the existing model/holder max and top3 (signed cube), Jaccard
max/top3, candidate-analogy max/top3, per-catalog fp32 zscores and exact nested
BC1 half-strength/CS1 arithmetic. Use frozen seed0/fold0 recipes, never refit.
Score all10,533 drugs symmetrically, filter OTHER known TRAIN positives using
TF1's symmetric union, retain the inspected answer, average ties. Record
model/pipeline ranks, true versus old competitor feature values and margins,
new winner, and candidate/permutation/reference checks. Candidate-side zero
beta remains zero; no RS1 reverse-scoring addition. The model saw these TRAIN
edges during training; this is not a fresh holdout. The full-catalog candidate
normalization is also different from official501-column evaluation. Thus this
checks existing pipeline behavior on these TRAIN cases, not official MRR.

## Original loss: focal row versus minibatch context

For each selected case, create one2,048-row same-relation TRAIN batch: focal
query first, then2,047 uniform with-replacement rows of that relation, same
direction. Use RNG36820+case_index and4,096 typed random negative draws,
matching the original conditional sampler but NOT replaying historical steps.
Keep duplicate/known-positive negative draws as the original recipe does.

Inspect two fixed candidate pools: the untouched random draws; and the same
draws with the first replaced by the strongest TF1 unobserved competitor ONLY
if it was absent. This forced-comparison diagnostic is not a new training
experiment. Identical pools are reported as such, not independent repeats.
Save batch identities before computing outcomes. Count focal known-positive
collisions and competitor exposure. Unobserved does not mean biologically false.

Compute original CE + T² KD(T=2,weight1) +0.1 trajectory using original own
teacher-mean logits. Differentiate CE, weighted KD, weighted trajectory and
their total separately for (a) focal-row loss, (b) whole-batch mean loss.

Define ranking-relevant margin M=Re(q*(E_positive-E_competitor).conj()).sum,
WITHOUT the positive global temperature. For each loss/component and parameter
group compute -dot_real(grad M,grad loss): positive means infinitesimal plain
gradient descent would improve this pair's margin; negative means worsen it.
Report entities, operators, representation (entities+operators), and temperature
(which has zero direct margin derivative), gradient norms and descent cosines.
Check component gradients sum to total, and synthetic finite-difference signs.
No actual parameter perturbation on real models, optimizer, clipping, Adam
preconditioner, finite update, MRR-gain forecast or historical-cause claim.
Also record each own teacher's positive-minus-competitor raw logit and their
original mean, without selecting teachers or changing distillation.

## Verification and handoff

Synthetic tests: deterministic balanced case selection, known-positive/tie
filtering, pair removal including duplicates/reverse, directional graph usage,
exact old blend arithmetic, direct feature checks and candidate permutation,
conditional batch construction, forced-negative isolation, complex margin
derivative finite differences, component/total gradient parity, frozen values.

Save cases, raw features, normalized features, catalog scores, sampled batches,
per-case/per-pool gradients and summaries. Independent saved-record endpoint
audit reproduces selection, catalog ranks, fixed blending, gradient-sum scalar
statistics and summary; rechecks source/input/artifact/model hashes. State
which checks are direct feature/gradient calculations versus saved-record replay.
No subgroup promotion, new experiment, gradient update or submission change.
Progress at least every minute with LR=0 and clearly labeled TRAIN diagnostic
MRR when available; official validation pipeline remains unchanged at0.85821665.
