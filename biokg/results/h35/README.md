# H35 — two-bank combination results

Completed 2026-09-05 on the local GTX 1080 Ti. **No adaptation arm passed the
advance gate; retain the original distilled student.** The separately
preregistered adaptive [H35B follow-up](../h35b/README.md) also completed without
an improvement over the original model.

## Full-validation results

Each endpoint uses all 162,886 validation triples, both directions (325,772
queries), all 500 official typed negatives and OGB average-tie ranks. These
are model-only results, not the feature-augmented submission pipeline.

| Arm | Validation MRR | Change vs frozen | Training seconds |
| --- | ---: | ---: | ---: |
| Frozen T=2 student | 0.8321748668 | — | 0 |
| Single-bank refinement | 0.8309516582 | −0.0012232086 | 49.93 |
| Two banks, mean | 0.8312714993 | −0.0009033675 | 63.37 |
| Two banks, smooth OR | 0.8312382745 | −0.0009365923 | 73.43 |
| Two banks, smooth AND | 0.8313424501 | −0.0008324167 | 76.58 |

All four paired-triple bootstrap 98.75% intervals for the change versus frozen
were below zero; exact intervals and all pairwise/slice diagnostics are in
[summary.json](campaign_s0/summary.json). These quantify query variation for
this screen, not seed uncertainty or immunity to adaptive validation reuse.
None meets the +0.001 improvement gate. AND's +0.000071 versus mean is not a
useful architecture win when it still loses to the starting model.

Timing excludes progress probes and full final validation from the training
seconds column. Per-arm elapsed times including probes/final validation were
59.21 / 75.10 / 85.36 / 88.78 seconds. These are not from-scratch training times.

## What the second bank actually learned

The table, first bank and temperature remained frozen. Only B adapted through
TRAIN sampled cross-entropy. All three combinations were initialized from the
same 5%-perturbed bank and saw the exact same TRAIN sampling sequence.

- Mean query cosine A versus B: **0.99865 (mean), 0.99862 (OR), 0.99869 (AND)**.
  This recipe produced very similar queries, not evidence of separate modes.
- B alone scored 0.82953 / 0.82919 / 0.82979 respectively, below A's 0.83217.
  The original A branch reproduced the reference ranks exactly.
- In the mean arm, median absolute positive-logit difference was 0.1044;
  90th percentile 0.2733. The median hypothetical smooth-OR correction to
  mean at T=1 was only 0.00136 logit units. This is a score diagnostic, not
  another reranking or hyperparameter selection experiment.
- Every adaptation lost more previously rank-1 answers than it recovered:
  e.g. AND lost 1,434 and gained 1,068. Full counts are in the receipt.

Interpretation: adding a near-copy with this frozen representation and CE-only
adaptation did not add useful information. This does **not** establish that
jointly learned multimodal operators cannot work, nor that we have identified
the biological cause of the remaining errors. The single-bank control also
loses, so switching away from the student's original training objective is
another limitation of this screen.

## Integrity and artifacts

- [Immutable preregistration](../../H35.md); no early selection or extensions.
- TRAIN/VALID selective loader only; no TEST option, test metrics, validation
  gradients or validation-derived feature inputs. No rented GPU used.
- 5,000 updates each; 117,504 trainable real parameters in every adaptation;
  27,124,129 total for single and 27,241,633 for two-bank variants. One shared
  entity table, no second inference model.
- Source/data/checkpoint hashes: [prerun.json](campaign_s0/prerun.json).
  Common training-stream SHA256:
  `190274fd1869d6dd16cb4074d46fa8e92e15e4bf0e5072ab6328aad312c23bf9`.
- Frozen tensors verified unchanged by hash. Same B initialization verified
  across mean/OR/AND. Each exported checkpoint contains its complete inference
  state and is loadable with `biokg.dual_operator.restore_adapted`.
- Official evaluator agreement checked for every query and both branch scores;
  no fp16 caches or modified negative sets. Source hashes remained equal to
  the launch receipt after execution.
- Before launch: 53 BioKG tests passed, including ten new H35 tests. All four
  full-shape synthetic CUDA smoke tests passed. One additional post-launch
  synthetic test verifies that OR/AND can represent candidate orderings that
  a linear mean cannot; it did not change the executed training code.
- `campaign_s0/progress.jsonl` records LR, probe MRR and timings; each arm has
  its own `.pt`, `.json`, and `_queries.npz`. Per-query artifacts are diagnostic
  only and must not become validation-derived training examples.

## Follow-up

[H35B](../../H35B.md) uses the same parameter budget but gives B a different
structural role: connect original 4-coordinate groups through a fixed shuffle.
An identity-initialized same-block composition is its matched control. This
follow-up is disclosed as adaptive; it does not retroactively alter H35.
It completed at 0.830389 (local) / 0.830610 (shuffle), both below the reference.

The fresh frozen-student diagnostics also confirm where its **model-only**
deficit lies: drug–drug 35.76%, drug–side-effect 23.47%, protein–function 22.37%
(81.60% combined). Their respective full-validation MRRs are 0.686002,
0.255782 and 0.858402. This updates the earlier plain-baseline diagnosis, but
still is not an error analysis of the feature-augmented submission pipeline.
