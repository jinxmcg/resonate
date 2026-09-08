# TF1: a concrete TRAIN drug–drug near miss

Completed 2026-09-07, CPU 5.69 seconds, LR=0. Seven new synthetic tests and
the 84 existing checks passed (91 total). Own frozen H35F k12 single model;
only TRAIN/node counts opened. No gradients, model updates, VALID, TEST,
external trained artifacts or submission change. Source/input/model hashes
and score/rank reconstruction checks passed; see `s0/audit.json`.

This is an in-sample **base-model** case study, not an evaluation of our
retrieval-augmented pipeline or evidence of a generalization improvement.
Fixed RNG3681 sampled 128 TRAIN triples, both directions, all 10,533 drugs.
The first sampled query was also the first eligible filtered top-ten error;
no additional sample or cherry-picked explanatory category was needed.

## The actual case

TRAIN row 1,168,263: drug-local-ID **1381 --relation38--> 786**, tail prediction.
Drug offset is 10,687; global entity IDs are source12,068, positive11,473,
competitor12,215. These are dataset indices, not drug names or biological
interpretations of the relation.

The positive ranks **73rd unfiltered**, but **10th after filtering other known
TRAIN positives** (including reverse copies). The source has 222 known TRAIN
answers for this relation. The raw winner, drug654, is another known positive;
it must not be treated as a demonstrated error. In this fixed sample, 227/256
raw winners were another known TRAIN answer, underscoring this distinction.

The remaining top competitor is drug1528. There is no known TRAIN edge for
this relation/pair, but absence from TRAIN does not prove biological falsity.
We did not consult VALID/TEST to resolve that uncertainty.

| Quantity | Correct drug786 | Competing drug1528 |
| --- | ---: | ---: |
| Catalog model score | 4.69475508 | 4.87140131 |
| Embedding norm | 0.92731661 | 0.90783502 |
| Normalized query–candidate alignment | 0.44279790 | 0.46931849 |
| Best source-holder cosine, focal pair removed | 0.83358395 | 0.79184729 |
| Best candidate-to-source-neighbor cosine, focal pair removed | 0.75745499 | 0.64732528 |

## What makes the model score wrong under this diagnostic?

The exact winner-minus-positive logit margin is about **+0.1766467**.
Symmetric two-factor decomposition gives:

- Alignment advantage: **+0.2782304** for the competitor.
- Norm difference: **−0.1015837**, favoring the correct drug.

Thus the top competitor does **not** win because it has a bigger embedding.
Its learned direction is more aligned with the query. That preference already
exists before the relation operator (cosines0.5074903 versus0.5272278) and
remains afterward. The reverse operator also prefers the competitor
(scores4.7211561 versus5.2399807). Temperature11.4335 is positive and cannot
change candidate ordering.

Among all remaining candidates, removing target-norm differences in a local
diagnostic improves the answer from rank10 to rank3, but does not solve this
top-competitor error. No such normalization was added to the submission.

The **36 independent 4x4 blocks disagree**: 23 contribute to the competitor's
advantage and 13 favor the correct drug. Largest signed margin terms (zero-based
block indices): block33 +0.204789, block17 +0.176152, block6 +0.125991;
block34 −0.272471, block18 −0.229129, block22 −0.162462. Their total reconstructs
the model margin. These are additive coordinate contributions, not established
biological feature labels or causal explanations of how training produced them.
Selecting/removing blocks on this one example would overfit the explanation.

## What TRAIN already tells us

After removing the inspected pair in both directions from the same-relation
graph, source1381 still has 221 neighbors. Positive786 has293 holders versus202
for competitor1528; shared source/candidate neighbors are158 versus134. Counts
alone are degree-sensitive and should not be treated as conclusive evidence.

More specifically, source-like drug1221 links to the correct answer and has
source cosine **0.833584**, versus the competitor's best holder **0.791847**.
Source-neighbor drug1394 resembles the correct candidate with cosine **0.757455**;
the competitor's best analogue reaches only **0.647325**. Both are concrete,
TRAIN-only witnesses independent of directly looking up the focal pair.
The existing retrieval machinery may already exploit such witnesses: we did
not evaluate its full score here, and cannot claim a full-pipeline failure.

## Interpretation and limits

We established the **score-level** reason and found local structural evidence
that could favor the answer. This is more specific than assuming insufficient
k or adding another operator blindly. It does not yet establish whether the
training-level cause is objective conflict, insufficient hard-comparison
exposure, shared-embedding interference, or something else.

The historical H35F training path uses `TrainStream.sample(2048,4096)` and a
CE+retained-KD objective, without masking known positives in that path. For a
source with many valid answers, this is a plausible source of competing
supervision, not a demonstrated cause of this particular ranking. No teachers
were loaded, gradients computed, or historical mini-batches replayed in TF1.

Full-catalog TRAIN-filtered sample MRR is0.1516032 across256 directed queries;
the catalog, filtering, dataset and model-only scope differ from official
500-negative VALID metrics. **Do not compare it to0.85822** or infer a new
leaderboard score. Selection of an illustrative error and partial TRAIN-only
positive knowledge further limit conclusions.

Raw evidence: `s0/findings.json`, `s0/catalog_scores.npy`, `s0/prerun.json`,
`s0/audit.json`. No additional experiment or training launched.
