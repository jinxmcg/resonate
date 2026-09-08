# H35B — identity-initialized cross-block composition

Completed 2026-09-05 on the local GTX 1080 Ti. **No promotion.** Both endpoints
lost to the original distilled student. No subsequent training sweep launched.

This was a separately preregistered, explicitly adaptive follow-up to H35's
negative near-copy combination screen. See [H35B.md](../../H35B.md); it did not
change the original H35 protocol or its completed results.

## Full-validation results

All 325,772 directed queries, all 500 official typed negatives, fp32 scores
and official average ties. Same frozen T=2 student, 5,000 TRAIN updates per
arm, same learning rate/sampling stream and 117,504 trainable parameters.

| Model | MRR | Change vs frozen | Training seconds |
| --- | ---: | ---: | ---: |
| Original distilled student | 0.8321748668 | — | 0 |
| Same-block sequential control | 0.8303888799 | −0.0017859870 | 63.09 |
| Cross-block sequential shuffle | 0.8306100495 | −0.0015648173 | 64.40 |

Shuffle's improvement over local was only **+0.0002211697**. Its paired-triple
97.5% bootstrap interval was **[−0.0000045165, +0.0004369350]**; versus frozen,
**[−0.0018058608, −0.0013269714]**. It fails the preregistered gain and interval
criteria. These intervals address query variation within this adaptive
screen, not cross-seed uncertainty or the entire project's tuning history.

Total elapsed times including probes and final validation were 74.91 and
76.63 seconds; training-only times are in the table. Initial probe MRR for
both arms was **0.8411956954**, exactly matching the original student's fixed
probe. Final probe MRR was 0.8415583309 / 0.8401001308; probes were not used
to choose checkpoints, stop training, or revise hyperparameters.

## Interpretation

The second bank can genuinely connect original coordinate blocks: a synthetic
dependency test verifies that an input from one original block affects another
block only in the shuffled architecture. Nevertheless, this identity-start,
frozen-table/first-bank, CE-only adaptation did not improve validation ranking.
Mean cosine of original versus composed queries remained 0.99861 (local) and
0.99897 (shuffle).

This rejects a cheap frozen-representation add-on under the tested recipe,
not every two-bank model or a jointly learned cross-block representation.
H35's single-bank refinement also lost, so loss of the original distillation
training signal is a plausible confound. We have not isolated that cause.

If the two-bank direction is pursued again, a distinct experiment should
jointly train the embeddings/operators while retaining the original
distillation signal, with a matched single-bank distillation control. Do not
simply select another frozen-bank temperature, initializer or permutation
against the same validation results. No such next experiment is running.

## Verification and artifacts

- No TEST loading or validation gradients; all adaptation uses TRAIN labels.
  One shared entity table and one self-contained inference checkpoint.
- 27,241,633 total real parameters; 117,504 trainable. E, A and log_tau verified
  bit-identical to the source checkpoint after each run. Original A ranks
  exactly match the frozen reference on every validation query.
- Both saved checkpoints restore independently using
  `biokg.sequential_operator.restore_sequential`. Input, checkpoint, prior
  reference and executed-source hashes are in
  [prerun.json](campaign_s0/prerun.json); source hashes rechecked after execution.
- Same TRAIN sequence as all four H35 arms:
  `190274fd1869d6dd16cb4074d46fa8e92e15e4bf0e5072ab6328aad312c23bf9`.
- Four new synthetic tests cover identity parity, local block-product
  equivalence, true cross-block dependence, finite training/frozen parameters,
  checkpoint restore and candidate permutation. Full BioKG suite: **58 passed**.
- [summary.json](campaign_s0/summary.json), `progress.jsonl`, per-arm `.json`,
  `.pt` and `_queries.npz` preserve the outputs. Previous test exposure in the
  project remains disclosed; this pilot does not certify historical provenance.
- Our GPU compute process was gone after completion. The desktop still uses
  the local GPU; the rented 5090 was not accessed.

Reproduce from the repository root into a fresh output path:

```sh
/mnt/geocore/geocore/.venv/bin/python -m biokg.train_sequential_operator \
  --model biokg/checkpoints/dist_T2_s0.pt \
  --data-root /mnt/geocore/geocore/data_ogb \
  --h35 biokg/results/h35/campaign_s0 \
  --out biokg/results/h35b/reproduction_s0 --device cuda
```
