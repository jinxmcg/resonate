# ResonatE: Row-Sparse Knowledge-Graph Embeddings with Composable Relation Operators

> **Preprint / draft (4 September 2026), not peer reviewed.** The results
> below are self-reported; the OGB leaderboard submissions (five entries,
> listed at the end) are being filed from the receipts in this repository.
> Project page: [resonate.page](https://resonate.page). Paper:
> [`paper/resonate.pdf`](paper/resonate.pdf).

One architecture and one training shell, run on two link-prediction
benchmarks of Stanford's Open Graph Benchmark (OGB) of very different
shape, with two capacity dials set on validation per graph:

- **[ogbl-biokg](https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg)** —
  93,773 typed biomedical entities, 51 relations, 4.76M training triples,
  500 type-matched negatives per query. A **27.1M-parameter** model
  (k=12, 4×4 blocks).
- **[ogbl-wikikg2](https://ogb.stanford.edu/docs/linkprop/#ogbl-wikikg2)** —
  2,500,604 Wikidata entities, 535 relations, 16.1M training triples,
  500 uniform negatives per query. A **329M-parameter** model (k=8, one
  dense 64×64 block per relation).

Both use the same entity table (unit-norm vectors, stored through their
real view with row-sparse gradients and row-wise Adagrad), the same
relation operators (free blocks, unitary at initialisation), the same
label-free retrieval members read from the training graph, the same
per-relation blend fixed on validation, and the same distillation of a
fresh model from ten seeds at temperature 2. Every number is the mean
and unbiased standard deviation over ten seeds with **one test read per
seed** under the official OGB Evaluator.

## Results

### ogbl-biokg (`biokg/`)

| | test MRR | valid MRR (held-out / in-sample) | hits@1 | hits@10 | params |
|---|---|---|---|---|---|
| A. single model | 0.8158 ± 0.0006 | 0.8164 ± 0.0006 | 0.7457 | 0.9412 | 27,124,129 |
| B. A + retrieval features | 0.8463 ± 0.0004 | 0.8465 / 0.8472 | 0.7875 | 0.9523 | 27,124,129 |
| C. distilled (T=2) + retrieval features | **0.8528 ± 0.0002** | 0.8532 / 0.8537 | 0.7964 | 0.9539 | 27,124,129 |

`python biokg/scripts/summarize.py` prints this from `biokg/results/sparse/`.
The distilled model alone scores 0.8321 ± 0.0003 on validation; the same
pipeline with T=1 students gives 0.8509 ± 0.0003 on test (the temperature
ablation, `summarize.py` prints it too). The public board as of
2026-09-04: RelEns 0.9618 (849M, ensemble), ComplEx² 0.8583 (188M), UniBi
0.8550 (182M), AutoBLM-KGBench 0.8536 (192M), ComplEx-RP 0.8492 (188M), TripleRE
0.8348 (470M), AutoSF 0.8309 (94M), PairRE 0.8164 (188M), ComplEx 0.8095
(188M). Row C is 5th of 12 and the 4th single model, at the smallest
parameter count on the board by 3.5×.

This is the **second** biokg ladder. The first (`biokg/results/dense/`,
dense Adam on the entity table: A 0.8118 ± 0.0013, B 0.8425 ± 0.0007,
C 0.8505 ± 0.0004, all ten seeds, all read once) is kept as history: the
sparse shell was built for wikikg2, then brought back to biokg under a
pre-registered bar and won on every seed (+0.0040 on A). Only the sparse
ladder is submitted.

### ogbl-wikikg2 (`wikikg2/`)

| | test MRR | valid MRR (held-out / in-sample) | params |
|---|---|---|---|
| A. single model | 0.6676 ± 0.0010 | 0.7030 ± 0.0009 | 328,842,753 |
| B. A + 8 retrieval members, frozen blend | 0.6820 ± 0.0014 | — / 0.7413 | 328,842,753 |
| B+. + self-augmented members | 0.6866 ± 0.0015 | — / 0.7467 | 328,842,753 |
| F. A + 10 members + learned combiner | 0.7222 ± 0.0010 | 0.779 / 0.7800 | 328,842,753 |
| C-F. distilled (T=2) + members + learned combiner | **0.7320 ± 0.0010** | 0.787 / 0.7880 | 328,842,753 |
| ensemble of 10 seeds + members + learned combiner | **0.7426 ± 0.0002** (leave-one-out, n=10; full ten-seed 0.7430) | 0.7992 / 0.7998 | 10 × 328.8M |

`python wikikg2/summarize_wiki.py` prints this from `wikikg2/results/`.
The distilled model alone scores 0.6855 ± 0.0008 on test (0.7190 on
validation). The public board as of 2026-09-04 (28 entries): RelEns 0.7392
(2.18B params, ensemble), StarGraph + TripleRE + Text 0.7305 (1.93B, uses
entity text), InterHT+ 0.7293 (156M), StarGraph + TripleRE 0.7286 (93M),
InterHT+ 256-dim 0.7257 (148M), StarGraph + TripleRE 0.7201 (87M),
CompoundE3D 0.7006 (751M), TranS 0.6939 (38M). The ensemble row would be
1st; C-F would be 2nd and the best single model, without text. The
ensemble is also the largest entry on the board (3.29B); the single model
sits mid-table in size.

The ensemble entry's mean and std are over the ten leave-one-out
ensembles (each of 9 seeds, each read once); the full ten-seed ensemble
was read once, 0.7430. The learned combiner is fit on the full validation
split, so its in-sample validation number is optimistic; the held-out
column is the same fit on a random half of the validation triples,
scored on the other half (`learned_blend.py` default mode), which is
what the submission form carries. It is within 0.001 of in-sample on
every seed where it was computed (the ensemble's one fit, seven of ten
row-F seeds, six C-F seeds whose member caches survived; the machine was
released before the rest).

## The model (`resonate.py`, `resonate_wiki.py`, `rowadagrad.py`)

An entity is a vector of M = k² complex coefficients; a query entity is
read unit-norm, a target entity as its raw row (its norm is a learned
popularity channel). A relation in a direction is a block-diagonal
operator of M/b free b×b complex blocks, initialised unitary by QR. A hop
applies the operator and renormalises; the score is the real inner
product against the target row, times a learned scale. Blocks do not
commute, so the operator for a chain of relations is the product of the
blocks. The "modes" vocabulary in the code is a coordinate label: the
score is invariant under any block-diagonal unitary change of basis, and
"complex" is not a claim either (an equal-parameter real version scores
the same; `resonate.py real=True`).

`resonate_wiki.py` is the same model with the entity table stored as an
(N, 2M) real tensor read through `F.embedding(sparse=True)`: a step
touches only the rows in the batch, and `rowadagrad.py` keeps one
accumulator per row. Dense Adam on the table needs the whole (N, 2M)
gradient and two moments every step, which is what stopped the first
biokg ladder from scaling to 2.5M entities. `resonate_wiki.selftest()`
checks the sparse path against the dense parent to 1e-6.

The two dials, and why they differ (both settled on validation; the
block-size curves are in `PLAN.md` H23 of the research tree and in the
paper): on biokg the block-size curve is flat from 4×4 to 16×16 and falls
at a dense operator; on wikikg2 it rises monotonically to dense. k=8 with
a dense block matches k=12 with 4×4 blocks on wikikg2 at 45% of the
parameters; on biokg k=8 costs 0.02 whatever the block.

## The retrieval members

For a query (h, r, ?) and each candidate c, the *holders* of c are the
training sources already linked to c under (r, direction). Members
(all read from `split["train"]` only; validation and test edges are never
used as graph input):

- **analogy** (both boards): max over holders of cos(e_h, e_holder)³ under
  the model's own embeddings, and the mean of the top three.
- **jaccard** (biokg): the same with raw edge-overlap similarity; model-free,
  computed once.
- **holders**, **cn / cn_aa / linked**, **cn3_aa**, **typed** (wikikg2): the
  holder count, two- and three-hop neighbourhood overlaps, and typed
  two-hop paths, needed because uniform negatives on a sparse graph mostly
  fail on "does c take this relation at all".
- **self-augmented members** (wikikg2, rows B+ and up): the same members
  computed on the training graph plus the seed-0 model's own confident
  proposals (4.18M edges over 18 relations, `augment_graph.py`). The
  per-relation confidence threshold and the set of relations are
  calibrated on validation labels; no validation or test edge is added.
  Base and augmented members are always offered together so the blend can
  ignore either per relation. Effect: row B 0.6820 → B+ 0.6866 test.

Members are blended per (relation, direction) with weights chosen on
validation: by selection from a fixed list on biokg (`ensemble_weights.py`,
`freeze_test.py`; a learned combiner gained ≤ 0.0007 there), by a listwise
logistic regression on wikikg2 (`learned_blend.py`, +0.026 held-out,
because wikikg2's members conflict and need negative weights).

## Distillation

A fresh model of the same recipe is trained with an added
KL(mean of ten frozen teachers ‖ student) over each batch's [positive |
negatives] logits at temperature T, weight T². The ten teachers are the
ten row-A seeds. T=1 transfers +0.012 (validation) on biokg and nothing on
wikikg2; T=2 transfers +0.016 and +0.018 respectively; T=3 and T=4 are
below T=2 on biokg. T=2 is used on both boards.

## Setup

```
git clone https://github.com/jinxmcg/resonate && cd resonate
uv sync                        # Python ≥ 3.12; torch ≥ 2.6, numpy, scipy, ogb
```

Datasets are downloaded by `ogb` into `<folder>/data_ogb/` on first use
(biokg ≈ 2.9 GB, wikikg2 ≈ 1.2 GB + 3 GB processed). Every script sets
`PYTHONPATH` to the repository root for the shared modules; run them from
their folder or via `scripts/`.

## Verify without retraining

```
cd biokg
scripts/fetch_checkpoints.sh              # release assets, see below
uv run python verify.py --device cuda     # split disjointness + re-evaluates the ten row-A models
python scripts/summarize.py               # the biokg table from results/sparse
python scripts/summarize.py dense         # the first ladder from results/dense
cd ../wikikg2 && python summarize_wiki.py # the wikikg2 table from results/
```

`verify.py --ladder sparse` re-scores `checkpoints/sparse_s*.pt` (the
sparse-shell row-A models re-saved in the dense format by
`sparse_to_dense.py`, an exact reinterpretation of the real-view table)
and checks each against its `results/sparse/h24_sparse_s*.log` to 3·10⁻⁴;
`--ladder dense` does the same for the first ladder. To re-apply a frozen
blend the member caches must exist (about an hour of CPU per seed for the
analogy features on biokg); after that `freeze_test.py` /
`learned_blend.py --freeze` with the same members reproduce the receipts
exactly.

## Reproduce from scratch

biokg, per seed (`biokg/scripts/`):

```
scripts/run_campaign_sparse.sh   # row A: train_biokg_comp.py --shell sparse, --eval both (one test read), then sparse_to_dense.py    ~4.5 min/seed on a 5090
scripts/run_retrieval.sh         # row B: ROW=sparse  -> cache_scores, analogy features, shared jaccard, held-out estimate, one frozen test read   ~40 min/seed, mostly CPU
scripts/run_distill_sparse.sh    # row C: T=2 students from the ten row-A checkpoints (--eval valid), then the row-B steps with ROW=distT2      ~9 min/seed GPU + ~40 min/seed CPU
```

wikikg2, per seed (`wikikg2/scripts/`):

```
scripts/run_finals.sh            # row A (train_wiki.py, 800k steps, --eval both) and row B (8 members, frozen blend)     ~40 min/seed on a 5090 + members
scripts/run_aug_finals.sh        # row B+: augment_graph.py proposals from that seed's model, augmented members, frozen blend
scripts/rowf.sh                  # row F: 10 members + learned combiner, one read per seed
scripts/run_distill.sh           # a T=2 student from the ten bf16 teachers (DT=2.0, 400k steps)                             ~50 min/seed
scripts/student_campaign2.sh     # row C-F: student -> members -> learned combiner, one read per seed
scripts/ens_chain.sh, ensplus_chain.sh, loo_ens.sh   # the ensemble rows and the leave-one-out reads
```

The wikikg2 scripts are the ones that ran the campaign across three
rented RTX 5090 boxes, with the box-specific paths removed; they expect
`teachers/model_wiki_s{0..9}.bf16.pt` (row-A checkpoints re-saved with a
bf16 table, `to_bf16.py`) and write to `ens_cache/`, `results/`, `logs/`.

## Protocol

- Every configuration's expected test range was written down before its
  first test read (the research log, `PLAN.md` in the project tree, holds
  the pre-registrations H1–H27 for biokg and the wikikg2 log). Every
  comparison between recipes, members, blend rules, temperatures and
  shells was made on the validation split.
- The test split is read once per run: row A by the trainer's final
  evaluation, the blend rows by `freeze_test.py` / `learned_blend.py
  --freeze`. Test score caches are written without computing any
  per-member test metric.
- **biokg development reads**, each a single shot of a configuration
  chosen on validation and none re-run: the nine listed in the first
  ladder's paper section (25k-step single 0.8024; 2×2 blocks seeds 0/1
  0.8084/0.8080; 1080 Ti seeds 0/1 0.8134/0.8123; a uniform ten-model
  ensemble 0.8428; seed 0 + three analogy features 0.8368; a 48M single +
  features 0.8472; the 48M recipe distilled from twelve checkpoints
  0.8535). The sparse ladder added none: its rows A, B, C(T=1) and C(T=2)
  are the first and only reads of those forty runs.
- **wikikg2 development reads** (all seed 0 unless noted, all recorded in
  the log): gap diagnostics 200k no-N3 0.6637 and 200k N3 0.6487; row B's
  first freeze at blend guard 4000 on seeds 0, 1, 5 (0.6785 / 0.6795 /
  0.6791), **re-read** after the guard was lowered to 500 on validation
  (0.6830 / 0.6848 / 0.6805) — the only runs in this repository read twice,
  and row B is not a submitted entry; T=1 student pilots seeds 2 and 0
  (0.6669 / 0.6672); row E (ten-seed ensemble + 6 frozen members) 0.7129
  and E+ 0.7164; a members-only diagnostic (eight shared members + learned
  combiner, no model) 0.6723. Each submitted wikikg2 run — the ten C-F
  students with their own `--eval both` read, the ten C-F blends, the ten
  leave-one-out ensembles and the full ensemble — was read exactly once.
- The blend weights are selected (biokg) or fit (wikikg2) on the full
  validation split; the held-out estimate of the same procedure is
  reported alongside and, where it exists, is within 0.001 of the
  in-sample value and within 0.001 of test on biokg.
- Row C's ten students share the same ten teachers, so its std reflects
  student-seed variance only. The leave-one-out ensembles share nine of
  ten members pairwise, so their std reflects membership sensitivity only.
- No external data. All members use only the training triples; the
  self-augmented members use the training triples plus edges proposed by
  the model trained on them.

Hardware: the sparse biokg ladder and all wikikg2 rows were trained on
rented RTX 5090 (32 GB) instances (torch 2.14+cu130), at $0.60 per GPU-hour, for a total
rented-GPU bill of $52.93 for the whole three-day project (both biokg
ladders, every wikikg2 row, every probe; about 88 GPU-hours); the first biokg
ladder on RTX 5090 and GTX 1080 Ti (torch 2.6.0+cu124); verification on
the 1080 Ti. `verify.py` shows checkpoints trained on one reproduce on
the other to < 3·10⁻⁴.

## Layout

```
resonate.py, resonate_wiki.py, rowadagrad.py, resonate_comp.py   shared model, sparse table shell, row-wise Adagrad
biokg/
  train_biokg_comp.py      trainer (both shells: --shell dense|sparse; --distill; official evaluation)
  train_ogb.py             the first ladder's trainer (dense shell), kept for results/dense
  sparse_to_dense.py       sparse checkpoint -> dense checkpoint format (exact)
  cache_scores.py analogy_member.py jaccard_member.py ensemble_weights.py freeze_test.py verify.py
  scripts/                 run_campaign_sparse.sh run_distill_sparse.sh run_retrieval.sh (both ladders)
                           run_campaign.sh run_distill.sh (first ladder) summarize.py fetch_checkpoints.sh
  results/sparse/          the submitted ladder: h24_sparse_s*.log, committed_*_s*.json/.log, pure_*_s*.log,
                           h25_dist_s*.log (T=1 students), h26_dist_T2_s*.log (T=2 students), h27_* (T=3/4 probes)
  results/dense/           the first ladder's receipts
wikikg2/
  train_wiki.py            trainer (sparse shell, --distill, official evaluation)
  retrieval_wiki.py cn_wiki.py cn3_wiki.py typed_paths.py augment_graph.py cache_wiki.py ensemble_wiki.py
  blend_wiki.py learned_blend.py summarize_wiki.py query_wiki.py to_bf16.py test_*.py
  scripts/                 run_finals.sh run_aug_finals.sh rowf.sh run_distill.sh student_campaign2.sh
                           ens_chain.sh ensplus_chain.sh loo_ens.sh
  results/                 rowA/ rowB/ rowBplus/ rowF/ rowCF/ ensemble/ xfit/ (receipts, frozen weights, logs)
paper/                     resonate.tex, refs.bib, resonate.pdf
```

Release assets (`scripts/fetch_checkpoints.sh`): biokg `sparse_s{0..9}.pt`
and `dist_T2_s{0..9}.pt` (108 MB each, dense format, `{"model", "offset",
"n_rel", "args"}`), the first ladder's `model_final_seed{0..9}.pt` and
`dist27_s{0..9}.pt`, and wikikg2's ten teachers `model_wiki_s{0..9}.bf16.pt`
(644 MB each) plus the seven surviving T=2 students
`model_dist_s{1,4,5,6,7,8,9}.bf16.pt`; students 0, 2, 3 were deleted by the
campaign script after their reads and cannot be re-verified from a
checkpoint (their receipts and logs are in `results/rowCF/`).

## Leaderboard entries

| board | entry | test MRR | valid MRR | params |
|---|---|---|---|---|
| ogbl-biokg | ResonatE (single model) | 0.8158 ± 0.0006 | 0.8164 ± 0.0006 | 27.1M |
| ogbl-biokg | ResonatE + retrieval features | 0.8463 ± 0.0004 | 0.8465 ± 0.0004 (held-out) | 27.1M |
| ogbl-biokg | ResonatE distilled + retrieval features | 0.8528 ± 0.0002 | 0.8532 ± 0.0003 (held-out) | 27.1M |
| ogbl-wikikg2 | ResonatE distilled + retrieval members + learned combiner (self-augmented graph) | 0.7320 ± 0.0010 | 0.787 (held-out) | 328.8M |
| ogbl-wikikg2 | ResonatE ×10 ensemble + retrieval members + learned combiner (self-augmented graph) | 0.7426 ± 0.0002 | 0.7992 (held-out) | 3.29B |

No external data on either board. One RTX 5090 per run.

## License

MIT — see [`LICENSE`](LICENSE). The checkpoints in the releases are
derived from OGB data under its own terms.
