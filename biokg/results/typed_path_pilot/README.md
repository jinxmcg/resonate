# TP1: TRAIN-only typed-path feasibility

Status: **failed feasibility gate; stopped**, 2026-09-06. All 71 tests passed
(11 TP1 plus 60 existing retrieval/operator checks), followed by a successful
endpoint audit. CPU pilot took **28.19 seconds**, including artifact hashing.
No wider path sweep or validation run launched. [Protocol](../../TYPED_PATH_PILOT.md),
[runner](../../typed_path_pilot.py), [tests](../../test_typed_path_pilot.py),
[endpoint replay](../../audit_typed_path_pilot.py).

Three degree-weighted paths between drugs, through proteins, side effects
or diseases. Construction/fit/report are disjoint subsets of TRAIN, grouped
by unordered entity pair across every relation and orientation. Select
structural weights on fit-TRAIN only, report on separate report-TRAIN.
No VALID, TEST, neural checkpoints, previous prediction caches or GPU use.
No model updates or gradients; LR=0. Existing 0.8582166512 validation
pipeline remains unchanged and is not a comparator in this pilot.

Up to 256 non-self triples per drug–drug relation per held-out role, both
directions, 500 fresh distinct typed negatives. The stratified TRAIN MRR is
not comparable to official validation/test MRR. Only a positive coverage
and ranking-signal check can justify proposing a later pipeline test;
standalone success does not establish complementarity with the current A.

## Results and decision

6,479 sampled triples for weight selection; 6,524 separate report triples
(13,048 directed report queries, 2,614 unique unordered report pairs).
The cap oversamples rare relations compared with the official task; these
scores are not comparable to our 0.85822 validation pipeline or its 0.70214
drug–drug slice. Every score below uses the same pilot report candidates.

| Pilot scorer | Held-out TRAIN MRR | Hits@1 | Hits@10 |
| --- | ---: | ---: | ---: |
| Construction-TRAIN candidate degree (control) | 0.4694555692 | 0.2703862661 | 0.9056560392 |
| Fit-TRAIN selected typed paths | 0.2435084602 | 0.0996321275 | 0.6168761496 |
| Uniform normalized paths | 0.2184519945 | 0.0869865113 | 0.5454475782 |
| Pooled raw paths | 0.2417506873 | 0.1017780503 | 0.5951870018 |
| Shared protein alone | 0.0189550721 | 0.0045984059 | 0.0403126916 |
| Shared side effect alone | 0.2424869743 | 0.0979460454 | 0.6171827100 |
| Shared disease alone | 0.0484768963 | 0.0322654813 | 0.0593194359 |

Primary selected-path minus degree delta: **−0.2259471090**. Descriptive
paired 95% interval **[−0.2398670589, −0.2118654311]**, 2,000 unordered-pair
cluster bootstrap replicates, both directions/all sampled relations on a
pair grouped together. It fails the preregistered +0.01-over-degree gate.

Positive path coverage is high, **96.0300%**, but almost entirely supplied
by shared side effects (95.9381%). Protein coverage is only 8.2005%, disease
5.9319%. Thus broad overall coverage did not provide sufficiently strong
ranking information under this gate. The global fit choice is shared side
effects alone. Of 76 directed target groups, 63 select that single path,
11 select equal sideeffect/disease weights, and 2 select uniform weights;
46 groups use local selection and 30 inherit the global choice.

Selection improves over uniform by +0.0250564657 and pooled paths by
+0.0017577729, descriptively. Neither result overrides the failed primary
gate. **Park this exact three-path variant; do not advance to VALID.** This
is a resource-prioritization decision, not proof that every typed path is
useless or that these paths could never complement the neural model. Such
complementarity was deliberately not measured in this TRAIN-only pilot.

## Verification

The construction graph contains 4,285,608 unique TRAIN triples. One exact
duplicate was removed from the original 4,762,678 TRAIN triples. The
construction/fit/report unordered-pair sets are disjoint across all
relations and orientations. Only the construction graph supplies path
adjacencies and candidate-degree statistics. No known construction positive
or self candidate was accepted as a sampled negative.

The endpoint replay reproduced the graph, all sampled triples/candidates,
degree scores, all feature normalization, fit-only recipe, report ranks,
metrics and clustered interval. Explicit neighbor-intersection references
matched both features and candidates on **203 representative directed
queries, all 501 candidates and all three paths, maximum error 0.0**.
Raw path features were independently rebuilt on those samples, not on every
query. Source/input/artifact hashes verified. Dataset opens were exactly
TRAIN and raw node counts; no VALID, TEST, checkpoint or GPU use.

[Summary and all relation/direction results](s0/summary.json),
[graph record](s0/graph_audit.json), [selected recipe](s0/recipe.json),
[prerun receipt](s0/prerun.json), [runner audit](s0/audit.json),
[endpoint audit](s0/endpoint_audit.json).

## Synthetic fixture correction

The initial synthetic end-to-end fixture was nearly complete and lacked
enough distinct unknown candidates. The sampler correctly failed closed;
added isolated synthetic drug entities to make the fixture valid, and kept
an explicit insufficient-candidates regression. No real-data result was
seen and no sampling rule was relaxed.
