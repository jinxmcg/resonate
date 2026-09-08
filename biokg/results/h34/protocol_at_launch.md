# BioKG optimization queue

Updated: 2026-09-05. Goal: a reproducible, rules-compliant **single inference
model**, optionally distilled and/or augmented with TRAIN-graph features.
Work through these hypotheses sequentially; do not launch a broad sweep or
combine unproven changes. A clean top-three / second-place result is a useful
milestone, not a guaranteed outcome.

## Where we stand, and my best guess

The existing distilled student is about 0.8321 validation MRR alone. Its
feature-augmented pipeline reports 0.8532 on a held-out validation half and
0.8537 on full validation after hyperparameter selection (historical README
results, not a fresh audit). Neither number is directly interchangeable with
a fresh full-validation measurement. The published historical test result is
0.8528; do not reopen its test predictions for optimization.

The [official BioKG leaderboard](https://ogb.stanford.edu/docs/leader_linkprop/#ogbl-biokg),
checked on 2026-09-05, lists RelEns first at 0.9618 test MRR, ComplEx² second at
0.8583 (0.8592 validation), and UniBi third at 0.8550 test. Our reported test
gap to third is small, but that is context, **not a test-tuning objective**.
The previously discussed **0.9473 is an optimistic top-10-reranking bound for
the plain baseline**, not an achieved result or an expected improvement.

My leading diagnosis is a representation / ranking-information limitation,
not merely insufficient training or GPU speed:

1. **Retrieval uses a relation-blind similarity.** The current analogy feature
   compares entity embeddings with the same cosine for every relation. Yet
   being similar for drug interactions need not mean being similar for side
   effects. The trained relation operators offer a cheap alternative metric.
2. **The scorer uses one query vector.** It may struggle to rank several
   different modes of valid answers. A low-rank correction to that vector is
   still a linear readout against candidate embeddings; it does not resolve
   this expressivity issue by itself.
3. **Distillation may preserve shared blind spots.** The current teachers are
   variants of the same architecture. Their average may reduce noise without
   teaching the student genuinely new structural evidence.
4. **Most measured deficit is concentrated, not diffuse.** On the plain
   baseline, drug–drug, protein–function and drug–side-effect account for
   roughly 81% of the MRR shortfall. This must be remeasured on the strongest
   student + features before claiming it explains that pipeline's residuals.

These are hypotheses, not established causes. `1 - MRR` is not the fraction
of incorrect queries. More capacity could help, but earlier width sweeps,
generic two-hop features and simple feature recombination did not justify
prioritizing another blind sweep.

## Non-negotiable protocol

- Follow the [standard OGB rules](https://ogb.stanford.edu/docs/leader_rules/):
  model fitting, learned metrics, teachers and structural graphs use TRAIN
  edges only. No gradients on validation, no validation edges as graph input.
- Validation permits ordinary, disclosed, finite hyperparameter selection.
  Split tuning/evaluation by whole triple: its head and tail queries stay
  together. These are development estimates, not a pristine final holdout
  after repeated adaptive experimentation.
- Do not load TEST edges, candidates, score caches or metrics in these trials.
  Freeze the entire final recipe before any authorized final test evaluation.
  Previous test exposure must remain disclosed; a fresh runner does not erase
  history or certify submission eligibility.
- Score every provided candidate symmetrically, both directions, with official
  OGB average-tie ranks and fp32 scores. Do not exploit positive column position,
  candidate ordering, or artifacts of negative generation.
- Keep one checkpoint at inference. Any teachers are training-only. Disclose
  all training stages, feature provenance, parameter accounting, tuning and
  compute; do not assume teacher parameters can silently be omitted.
- Record failures as well as successes. Confirm promising changes across
  seeds, then follow the [submission reporting requirements](https://ogb.stanford.edu/docs/leader_overview/).

## Sequential experiments

| Order | Hypothesis / change | First decision |
| --- | --- | --- |
| 1 — H34 | Relation-aware analogy: replace `cos(E[h], E[u])` by `cos(H[r]E[h], H[r]E[u])` for TRAIN holders `u` of each candidate. Keep max / top-three pooling, model scores and Jaccard unchanged. No training. | Fixed student seed 0, paired validation-only screen. Advance only with useful complementary gain; otherwise archive this exact variant. |
| 2 — H35 | Two query vectors with one shared entity table and a nonlinear `logsumexp` scorer. Jointly train one checkpoint; no averaging separately trained models. | Matched-budget one-query control; inspect head collapse and ranking changes. Do not promote an early-training advantage that disappears at 50k steps. |
| 3 — H36 | Structural cross-fitted distillation. Train teachers / build features without one TRAIN fold, predict that fold, teach the student with those soft targets and TRAIN labels. | First establish that the teacher adds information beyond the existing student. Then test whether the student retains the gain with teachers removed at inference. |
| 4 — H37 | Squared-score typed conditional likelihood: `p(t|h,r) ∝ epsilon + (qᵀe_t)^2`, with exact typed normalizer `epsilon*N + qᵀ(Σ e_t e_tᵀ)q`. | Small correctness / compute pilot, then matched training. This is a new objective, not post-hoc squaring of existing logits. Account for dense Gram/gradient cost. |

H35 has a multi-semantic precedent in [TransG](https://arxiv.org/abs/1509.05488),
not evidence of a BioKG gain. H37 is inspired by
[generative KGE circuits / ComplEx²](https://arxiv.org/abs/2305.15944), but is
our proposed adaptation, not a reproduction of that leaderboard entry.

## H34 preregistration: relation-aware analogy

Register before looking at H34 real-data results:

- Use released BioKG `dist_T2_s0.pt`, checksum verified; one frozen model.
  Read its training configuration and verify entity/relation indexing.
- First screen: 5,000 validation triples sampled without replacement by
  NumPy `default_rng(3400)`, sorted; both directions, all 500 official typed
  negatives. Use the local GTX 1080 Ti (user-approved); leave the rented GPU
  free for other work. CPU fallback does not change the protocol.
- A separate runner explicitly opens only TRAIN and VALID through the
  selective loader. Do not run the legacy end-to-end retrieval shell script:
  it includes test caching/evaluation, and legacy feature loaders call the
  all-split loader even for a validation request.
- Recompute fp32 student, global-cosine analogy (max/top3), relation-cosine
  analogy (max/top3), and Jaccard (max/top3) from TRAIN. Use unique TRAIN edges,
  exclude the query source from holders, preserve the legacy missing-holder
  sentinel -1 and signed cube for cosine features.
- Relation cosine uses normalized `model.hop(model.embed(...), directed_r)`
  for both source and holder; no newly learned projection. Audit how much the
  metric actually changes: a truly unitary H would preserve cosine. Report
  operator non-unitarity and feature differences instead of assuming novelty.
- Control: student + global analogy pair + Jaccard pair. Treatment: replace
  only the analogy pair. The student and Jaccard features are shared unchanged.
- Use a fixed triple-paired tuning/report split from `default_rng(3401)`.
  For each arm separately, select from the same finite global mixture grid:
  model weight in `{0.5, 0.75, 0.9, 1.0}`; share of remaining feature weight
  assigned to analogy in `{0, 0.5, 1}`; within each pair, equal max/top3 weights.
  Include the unchanged existing five-feature uniform mean as a control.
  Normalize each feature row by fp32 z-score. No per-relation fitting at this
  small-screen stage. This is a diagnostic control, not an exact recreation
  of the historical per-relation-tuned submission pipeline.
- Primary screen: held-out paired MRR delta. Also report fixed-uniform delta,
  model-only MRR, Hits@1/10, family/direction deltas, cosine change, ranking
  ties, and paired-triple bootstrap interval (1,000 replicates, seed 3402).
  The interval quantifies query variability, not seed uncertainty or immunity
  to adaptive validation reuse. No running score-driven changes.
- Advance to full validation if held-out delta is at least +0.001 and its
  paired bootstrap lower bound exceeds zero. Borderline results are
  inconclusive, not a success. On promotion, preregister the full-validation
  comparison using matched historical-style finite mixture choices; confirm
  seeds 0, 1, 2 before adopting it. A short-screen loss rejects only this exact
  zero-training metric variant, not all relation-aware retrieval ideas.
- Do not automatically start H35 while H34 is running. Record the outcome
  here before beginning the next hypothesis.

## Experiment ledger

- H33 random auxiliary: stopped at user request after the first completed
  full-length pair lost **0.0023838 MRR** (0.8130752 vs 0.8154589). Seed 1
  baseline completed; its random arm was interrupted. No three-seed claim.
  [Detailed receipt](biokg/results/h33/README.md).
- H34: seven synthetic correctness tests passed (brute-force feature parity,
  official ties, selective loading, candidate permutation, no model mutation,
  tuning-half isolation, and unitary/non-unitary metric behavior). Released
  student SHA256 `b463a8bcc431ada38f4834e235677464f9cd3a86346f88aecd2fd681a01260ef`.
  Local 1080 Ti CUDA smoke passed with about 9.5 GB free. Ready to run the
  fixed screen; no real-data H34 result yet. No rented-GPU job launched.
