# TP1: TRAIN-only typed-path feasibility pilot

Registered before real-data loading, feature computation or outcome inspection.
User approved the next typed-path investigation subject to strict split rules.
This first stage is only a small drug–drug feasibility check, not a new
validation comparison, submission change or automatic follow-on experiment.

## Data and model boundary

Open only official `split/random/train.pt` and `raw/num-node-dict.csv.gz`.
Pin their previously verified SHA256 digests and all pilot sources before
deserialization. A file-open guard rejects other dataset files and foreign
binary model/feature caches. Do not open VALID, TEST, raw full-graph edges,
processed graphs, external trained artifacts or previous prediction caches.
No neural checkpoint is loaded: our current A saw TRAIN, so it cannot be a
clean held-out-TRAIN comparator. No gradients, optimizer or GPU use. LR=0.
Any selected structural weights use only the pilot's fit-TRAIN labels.

The current 0.8582166512 validation pipeline remains unchanged. This pilot
cannot establish complementarity with it or predict a leaderboard result.

## Partition and sampling

Globalize entity IDs from TRAIN type fields and raw counts; deduplicate
exact triples. Group **all relations and both orientations of an unordered
entity pair** under `min(h,t)*N + max(h,t)`. A fixed uint64 SplitMix hash,
seed 3650, modulo 100 assigns buckets 0–89 to the construction graph,
90–94 to weight-selection TRAIN, and 95–99 to report TRAIN. Thus neither a
held-out edge, its reverse, nor another relation on that same pair can enter
the graph. Synthetic and real disjointness checks must enforce this.

Evaluate only non-self drug–drug TRAIN triples. Keep one deterministic
orientation per (relation, unordered pair); evaluate that observed triple
in both prediction directions. Independently sample up to 256 such triples
per relation per held-out role, seed 3651 plus role/relation. Do not backfill
sparse relations or change the cap after seeing results. This stratified
pilot is not the official relation-frequency distribution.

Use 500 distinct uniformly sampled drug negatives per directed query,
excluding its positive, source and known positives in the construction
graph for the queried directed relation. Seed each row independently with
`SeedSequence([3653,h,r,t,direction])`, so changes to report queries cannot
change fit negatives. Other held-out positives are not consulted to filter
negatives: unobserved candidates can be false negatives. These are synthetic
TRAIN development pools, not official VALID/TEST candidates.

## Fixed paths and score

Exactly three two-edge paths, in order: drug–protein–drug,
drug–sideeffect–drug, drug–disease–drug. Identify the unique TRAIN relation
for each drug/mediator type signature and preserve its orientation/identity.
Fail if the signature is missing or ambiguous; do not choose a substitute
from results. Reverse traversal is only an index of a construction-TRAIN
edge, not a newly inferred fact. No drug–drug composition paths or three-hop
side-effect predictions in this first pilot.

For path p, binary deduplicated bipartite adjacency B_p maps drugs to its
mediators. Score each candidate c for source s by
`raw_p(s,c) = sum_z B_p[s,z]*B_p[c,z]/degree_p(z)`.
Mediator degree uses construction TRAIN only. This reduces the influence
of hubs. Self candidate c=s has score zero, consistently for all columns.
All candidate IDs use exactly the same score function; no label-position,
duplicate-candidate or negative-generation shortcuts. No path/neighbor cap.

Compute sparse products in float64, cast raw features to fp32, then use
`Z_p = zscore(log1p(raw_p))` within each supplied candidate row, with std+1e-6,
fp32 throughout score combination and official average-tie ranking. Raw
features are candidate-pool independent; normalization is row-dependent.

Select one convex mixture using only fit rows from seven fixed choices:
uniform (first, preferred on ties), three single-path choices, three equal
two-path choices. Select globally then by target relation/direction when
there are at least 128 fit queries; otherwise inherit global. No gradients,
continuous tuning, new weights or report-driven gates. Save the selected
recipe before computing any report metrics. This is a pilot structural
score, not an ensemble of trained neural models.

## Fixed reporting and decision

Report sampled held-out-TRAIN MRR, Hits@1/10 and positive path coverage for
the selected score, each path, uniform normalized paths, pooled raw paths
(`log1p(sum_p raw_p)`), and relation/direction candidate degree from the same
construction graph. The degree score is a popularity control, not a model.
Report all target relations/directions, fit/report sizes and selected weights.

One primary contrast: selected paths minus candidate-degree MRR on report
TRAIN. Average both prediction directions within each sampled triple;
bootstrap unordered-pair clusters (all relations on a pair together), 2,000
replicates, seed 3655, using resampled sums/counts to retain query weighting.
This conditional descriptive interval is not an official generalization
estimate. Selected-vs-uniform and selected-vs-pooled are descriptive only.

Feasibility flag: report positive-path coverage >=10%, primary MRR delta
>=+0.01, and its lower descriptive 95% bound >0. This would justify proposing
a separate pipeline-complementarity test, not automatic promotion. A failed
flag rejects only this three-path pilot, not every possible typed path.

## Audit and stopping

Before running: tests for grouped reverse/cross-relation exclusion, stable
partitions, deduplication, type offsets/signatures, per-row deterministic
negative generation, missing paths, degree weighting, self exclusion,
candidate permutation/duplicates/pool extension, chunk parity, direct-dot
reference, official ties, fit/report poisoning isolation, sparse fallback,
and pair-cluster bootstrap arithmetic.

Persist the pre-receipt, graph counts, sampled triples and candidates, raw
and normalized features, degree scores, selected recipe, report rank vectors,
summary, and artifact hashes. A separate endpoint replay verifies source and
input hashes, reconstructs the construction graph and samples/candidates,
rebuilds representative raw scores by explicit neighbor intersections, and
reproduces all normalization, selection, report ranks and summary statistics.
Check source/input hashes again at completion. Progress every <=35 seconds
where computation is lengthy: stage, queries, elapsed time, LR=0; no new
validation MRR. Stop after the pilot and audit. No full-VALID evaluation,
full-TRAIN refit, checkpoint update, TEST, submission change or next run.

Conceptual precedent only (no code or trained artifacts imported):
[Das et al., 2020, probabilistic case-based reasoning](https://aclanthology.org/2020.findings-emnlp.427/).
Split boundaries follow the [OGB rules](https://ogb.stanford.edu/docs/leader_rules/).
