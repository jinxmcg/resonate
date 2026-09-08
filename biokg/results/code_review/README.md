# BioKG: learning from other implementations, without their trained weights

2026-09-06. Source/equation investigation, not a training experiment or a
submission result. User explicitly excludes other teams' trained artifacts.

Follow-up: the proposed TRAIN-only gradient diagnostic was subsequently
authorized and [completed on the local 1080 Ti](../gradient_diagnostic/README.md).
It weakens the broad trajectory-conflict hypothesis at our trained snapshots;
no model parameters were changed. The source-review actions below remain
the historical record of this review itself.

## Boundary and actual actions

- Read public source, papers and our existing code/experiment notes only.
- A ComplEx-RP archive had been downloaded during the earlier investigation.
  Its ZIP directory was listed, but its weights were never extracted,
  deserialized, scored, distilled or transferred to the rented machine.
  The 699,710,373-byte archive was deleted after the user's clarification.
- Remaining files under `biokg/external/complex_rp/` are source references
  and their CC-BY-NC-4.0 license, not weights. They are not imported by our
  model or trainer. Do not silently vendor them into the MIT implementation.
- No BioKG splits or checkpoints were opened for the synthetic check below.
  No TEST access, GPU experiment, model update or submission change.
- The supplied RTX 5090 was inspected only. Another process was running;
  it was not stopped or modified and no job/files were placed there.

Learning from equations does not waive attribution, source-code licenses,
or OGB split rules. Our boundary is stricter than an assumption that a
public checkpoint is automatically acceptable. Future trained artifacts
must come from our own documented TRAIN-only pipeline. Historical exposure
and adaptive validation reuse still need disclosure.

## 1. A already contains same-width ComplEx candidate rankings

Our code: `resonate.py:embed`, `hop`, `readout`, and
`biokg/joint_operator.py:retained_loss`.
Published comparison: [ComplEx implementation](https://github.com/facebookresearch/ssl-relation-prediction/blob/a050286473933b6b7b743db43db3529bd305359a/src/models.py#L340).

Ignoring the small normalization epsilon, A scores

`s_A(h,r,t) = tau * Re(<H_r e_h, e_t>) / ||H_r e_h||`.

For fixed `(h,r)`, the denominator and positive `tau` do not depend on the
candidate. With diagonal `H_r = diag(r)`, the numerator is the ComplEx
score. Our free 4x4 blocks can contain that diagonal. Epsilon changes the
positive query scale but not this candidate-order conclusion. Independent
reverse relations can be represented by their own directed operators.

This is a statement about representable rankings at the same width, not
equivalent optimization, probabilities, relation-prediction rankings or
cross-query score calibration. The published large model has a different
width. Merely removing normalization from a frozen A would not improve
its entity ranking in exact arithmetic.

Synthetic verification, CPU torch 2.6.0+cu124, seed 3600: instantiate our
`ResonatE(83,6,k=4,block=True,block_size=4)`, use freshly sampled complex
entity rows and diagonal relations packed into 4x4 blocks, and compare
the raw trilinear scores with the unchanged normalized model.

- 498 directed source/relation queries, all 83 candidates.
- Every candidate ordering matched exactly.
- Maximum absolute difference after restoring query scale: 5.72205e-6.
- No data, saved weights, upstream imports, optimization or GPU involved.

**Implication:** the gap to ComplEx-RP does not establish that A lacks its
basic diagonal interaction. Training and width remain separate hypotheses.

## 2. Their training objective is not our training objective

The published [training engine](https://github.com/facebookresearch/ssl-relation-prediction/blob/a050286473933b6b7b743db43db3529bd305359a/src/engines.py)
and [regularizer](https://github.com/facebookresearch/ssl-relation-prediction/blob/a050286473933b6b7b743db43db3529bd305359a/src/regularizers.py)
show entity CE, optional relation CE, and N3 factor regularization. The
ComplEx forward method scores the full entity table, not a fresh pool of
4,096 type-matched draws. Head prediction is trained through reciprocal
relations. The [published configuration](https://github.com/facebookresearch/ssl-relation-prediction/blob/a050286473933b6b7b743db43db3529bd305359a/doc/hyper-parameters/ogbl-biokg.md)
uses relation weight 0.25, N3 0.01, Adagrad and width 1,000.

Our current retained objective is sampled typed CE + own-teacher T=2 KD +
`0.1 * mean(||q - e_positive||^2)`. The last term both attracts queries and
positive targets and controls target magnitudes. For multiple positive
targets, its expected query gradient pulls toward their mean embedding.
That could conflict with a ranking objective, but could also regularize
beneficially. This is an algebraic hypothesis, not a measured BioKG cause.

Full-entity CE is not synonymous with increasing a sampled negative count:
it uses a different candidate support and does not repeat sampled IDs.
Nevertheless, our previous negative-count trials reduce the case for
assuming a large gain from exhaustive candidates alone.

Potential next diagnostic, not launched: on fixed TRAIN batches measure
weighted CE/KD/trajectory gradient norms and pairwise cosine by parameter
group. If justified, preregister a matched A-only trajectory-weight ablation
while keeping architecture, own-teacher KD, batches and schedule fixed.
Do not fit weights or target gradient-conflict examples using VALID errors.

## 3. TripleRE has a different distance objective, not extra biology inputs

The authors' [published scorer and loss](https://github.com/yulong-CSAI/TripleRE/blob/647c5e5f4863c015cc93e654438d20152ccd73ae/wikikg/model.py)
normalize entity rows, use separate head/tail scales plus translation, and
subtract L1 mismatch from a margin. Its training code offers sampled
log-sigmoid classification with detached self-adversarial negative weights
and frequency weighting. The available directory targets WikiKG; it does
not establish the exact BioKG run configuration.

Interpretation: weighted coordinate mismatches are penalized, whereas our
plain scorer rewards a single inner product. A bigger linear block does
not automatically reproduce an absolute-distance function. However, our
H35E distance branch already tried that general geometry at width 32;
it did not become a competent standalone ranker. H35F warmed the dot branch,
not the distance branch. Neither was a reproduction of full TripleRE.

Do not conclude that L1 is the missing feature or launch another B sweep.

## 4. RelEns is a combination rule, not a molecular feature extractor

The [paper](https://aclanthology.org/2023.emnlp-main.1034.pdf) combines
candidate ranks from independently trained base models, with weights that
vary by relation. The [repository](https://github.com/LARS-research/RelEns)
expects prediction arrays from those models; it does not add chemical
descriptors or protein sequences to our inputs. No published prediction
arrays or trained combination weights were obtained or used here.

Rank aggregation is candidate-set dependent and is not algebraically just
a larger relation matrix. This does not make the reported improvement
automatically compressible into A, nor identify which biomedical semantics
its base models encode.

The official AutoBLM-KGBench source link could not be retrieved during this
review. The accessible AutoSF-OGB branch is an older, different entry; do
not substitute its scoring pattern for the exact RelEns AutoBLM model.
The [AutoBLM paper](https://arxiv.org/abs/2107.00184) establishes bilinear
scoring-function search, but exact BioKG implementation details remain
unverified in this review.

## Previous experiments that must not be forgotten

From the historical research ledger `/mnt/geocore/geocore/PLAN.md`:

- Relation CE: H11 and later stronger-recipe tests did not justify adoption.
- N3: the early 12,500-step best improvement was +0.0019, below its gate;
  it was added to our normalized/trajectory recipe, not an exact replacement
  with the ComplEx training objective. N3 and trajectory attraction are not
  mathematically interchangeable.
- More negatives: 16,384 improved the early screen but only +0.0008 at
  50,000 steps; the proposed recipe was not adopted.
- Entity bias and unconstrained relation gains did not produce a robust
  full-pipeline gain. Our raw target norms already provide a magnitude
  channel; it is incorrect to claim we entirely discard popularity.
- H35C same-table operators remained almost identical. H35E/F established
  that a different or competent B alone does not imply a better whole model.

These results reject particular trials, not whole mechanism families. Old
claims that our architecture necessarily "pre-empts" those objectives are
stronger than the evidence supports.

## Decision

Keep B parked. First compare improved A with the existing feature pipeline
under the same evaluation protocol. For training research, investigate
objective interaction before another width/block search. Learn methods from
source, implement controlled changes ourselves, and train only our own
models on TRAIN. No new training run was launched by this review.
