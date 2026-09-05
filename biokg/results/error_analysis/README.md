# BioKG validation error analysis — 2026-09-05

The 0.7826 MRR baseline does not have a discrete "21.74% failing set."
Each query contributes reciprocal rank: rank 1 contributes 1, rank 2
contributes 0.5, rank 10 contributes 0.1. Its shortfall from 1 is therefore
the average of `1 - 1/rank`. Contributions below partition that shortfall,
not biological error probabilities or guaranteed recoverable improvement.

## Protocol and artifacts

Scored all 162,886 official validation triples in both directions:
325,772 queries, each with the official 500 type-matched negative candidates.
FP32 scores and OGB's optimistic/pessimistic averaged tie rank are used;
the diagnostic checks reciprocal ranks against the installed OGB Evaluator.
No cached FP16 scores or custom filtering change the evaluation.

Only train and valid split files were loaded. Models were frozen and scored
under inference mode; there was no training, gradient-based tuning, test
evaluation, submission, or validation-derived input to the model. Relation
names and entity identifiers are from the supplied OGB mapping files.
Graph coverage and degrees use unique training triples only. Query records
are diagnostic artifacts and must not become training examples or a
validation-derived blacklist, negative filter, or retrieval graph.

- `h29_baseline_s0/summary.json`: 12,500-step H29 baseline, seed 0.
- `full_baseline_s0/summary.json`: existing 50,000-step baseline, seed 0.
- Each directory also contains local `queries.npz` and `queries.csv.gz`:
  every query, true target, highest-scoring negative, exact rank, score
  margin, degrees and train-coverage flags. These large regenerable files
  are ignored by Git. `valid_index` and `direction` identify a query.
- Checkpoint paths, SHA256 hashes and complete saved arguments are in the
  JSON summaries. Four diagnostic tests plus the seven existing low-rank
  tests passed.

The two checkpoints use the same saved training hyperparameters except
training duration/cosine schedule and evaluation-probe frequency; the new
rank-zero arguments match the old defaults. These are separately trained
checkpoints, not a literal continuation or a multi-seed causal experiment.
Neither is the distilled model or retrieval-augmented submission pipeline.
An old checkpoint is evaluated here on validation; this does not certify
the provenance of the earlier submission or erase prior test-informed work.

Reproduction, from repository root with the installed torch/OGB environment:

```sh
python -m unittest discover -s biokg -p 'test_*.py' -v
python -m biokg.analyze_errors --checkpoint biokg/runs/h29/baseline/model.pt --data-root /mnt/geocore/geocore/data_ogb --out biokg/results/error_analysis/h29_baseline_s0 --device cuda
python -m biokg.analyze_errors --checkpoint /mnt/geocore/wiki/models/biokg_sparse_tlr03.pt --data-root /mnt/geocore/geocore/data_ogb --out biokg/results/error_analysis/full_baseline_s0 --device cuda
```

The script refuses to overwrite existing output directories. Choose fresh
output paths when reproducing.

## How the ranks break down

| True-answer rank | 12,500 steps: queries | 50,000 steps: queries |
|---|---:|---:|
| 1 | 229,499 (70.45%) | 242,926 (74.57%) |
| Above 1 through 3 | 42,763 (13.13%) | 39,347 (12.08%) |
| Above 3 through 10 | 29,175 (8.96%) | 24,345 (7.47%) |
| Above 10 through 100 | 19,732 (6.06%) | 15,605 (4.79%) |
| Above 100 | 4,603 (1.41%) | 3,549 (1.09%) |

Overall MRR: **0.7825889281 → 0.8159365266**. The baseline has 96,273
non-top-1 queries, not 22% of the queries. The longer run has 82,846.
Of the original non-top-1 queries, 28,006 rank first in the longer run;
14,579 previously top-1 queries cease to be top-1. There are 68,267 queries
outside rank 1 in both models, and 15,359 outside the top 10 in both.

There are no positive validation triples or their same-relation reverses
in training, and no highest-scoring evaluation negative in training for
its query. The early run has no positive/negative score ties; the full
run has one tied query, handled by the official averaged-rank rule.

## Where the shortfall lives

Family rows aggregate both prediction directions. Their deficit shares
use the 50,000-step model's `1 - MRR = 0.1840634734` denominator.

| Relation family | Queries | Early MRR | Full MRR | Share of full MRR shortfall |
|---|---:|---:|---:|---:|
| Drug–drug, all 38 relations | 62,270 | 0.6444 | 0.6693 | 34.35% |
| Protein–function | 86,362 | 0.7545 | 0.8250 | 25.20% |
| Drug–side-effect | 17,238 | 0.2399 | 0.2392 | 21.87% |
| Function–function | 79,622 | 0.9373 | 0.9554 | 5.93% |
| Disease–protein | 7,554 | 0.5455 | 0.6047 | 4.98% |
| Drug–protein | 13,078 | 0.7872 | 0.8274 | 3.76% |
| Protein–protein, all 8 relations | 59,114 | 0.9530 | 0.9663 | 3.32% |
| Drug–disease | 534 | 0.2614 | 0.3356 | 0.59% |

The first three account for **81.42%** of the full model's MRR shortfall.
The lowest-scoring tiny relation is not necessarily the most useful target.

Distinct patterns:

- **Drug–drug is mainly fine ordering.** Full Hits@1 is 49.96%, but
  Hits@10 is 97.73%. Of 31,160 non-top-1 queries, 29,747 are still in
  the top 10. This is a substantial pool of near misses.
- **Drug–side-effect remains difficult after full training.** Predicting
  the side effect given the drug: 0.1268 → 0.1321 MRR; only 28.45% of
  true side effects reach the top 10 in the full model. Predicting the
  drug given the side effect: 0.3529 → 0.3463. Together they contribute
  8,789 of the full model's 19,154 outside-top-10 queries (45.89%).
- **Protein–function is strongly directional.** Predicting a protein from
  a function improves from 0.6362 to 0.7375, but predicting a function
  from a protein is already 0.9126. The former remains the largest
  single relation-direction contributor (18.91% of full shortfall).
- **Unseen endpoints are not the dominant limitation.** Only 1,394
  queries have an endpoint unseen in that relation's training role;
  these explain 1.43% of the full shortfall. This does not mean low
  degree is irrelevant: full-graph degree 1–5 is a difficult small group.
- **Frequent competitors are an association, not a proven cause.** Among
  full-model drug→side-effect failures, the strongest negative has more
  training links in that relation than the true target in 77.40% of
  cases (medians 97 versus 42 links). Degree, relation and difficulty
  are confounded; this is not evidence that blindly penalizing popularity
  improves ranking.

## Actual query examples

Illustrations chosen deterministically near the median full-model rank
among queries failing in both checkpoints within each specified slice;
they are not worst-case selections or biological explanations. Entity
names below are the dataset's external identifiers, not inferred names.
"Wrong" means a benchmark negative, not established biological falsity.

| Validation index / direction | Relation | Given entity | True target | Strongest negative | Early → full rank |
|---|---|---|---|---|---|
| 41945 / tail | drug-sideeffect | CID000003249 | C0269190 | C0853117 | 16 → 37 |
| 90168 / head | protein-function | GO:0007017 | 5874 | 8452 | 7 → 5 |
| 17400 / tail | drug-drug_hematopoietic_system_disease | CID000002818 | CID000003954 | CID000004510 | 2 → 3 |

All other cases are available in the two query CSV files, including
successes; no selected subset replaces the full evaluation.

## Model-improvement hypotheses, not experiments already performed

1. **Mask known training positives in sampled-negative loss.** The current
   trainer samples 4096 type-matched candidates with replacement and uses
   single-target cross entropy without excluding other known training
   targets or duplicate copies of the current target. From the training
   graph, the expected number of known-positive candidate entries is
   429.30 for function→protein and 145.72 for drug→side-effect. These
   include repeats, not that many distinct mislabeled entities per batch.
   Filtering only training-known positives is a concrete objective
   comparison. It is not proved to fix the observed errors: some
   high-performing relations also have frequent collisions. Never use
   validation/test edges to construct the training filter.
2. **Training-only hard negatives for near misses.** Drug–drug failures
   suggest testing a fixed mixture of uniform and model-mined negatives,
   with known training positives excluded. Mine from training queries,
   not these validation examples. Keep model size and evaluation fixed;
   record extra computation and the incomplete-KG false-negative risk.
3. **A two-query scorer for many-to-many mappings.** One shared entity
   table, two jointly trained query vectors and nonlinear max/logsumexp
   aggregation could represent separate sets of plausible targets. This
   is one model, not an inference ensemble. A simple average would
   collapse back to a single query. Test separately from loss changes,
   with a matched parameter/compute control; the rank deficit alone does
   not establish that multimodality causes the errors.

Useful scale estimates, not forecasts: improving the average drug–side-
effect reciprocal rank by 0.10, with everything else unchanged, adds
0.00529 overall MRR; +0.03 on drug–drug adds 0.00573; +0.03 on protein–
function adds 0.00795. Check the distilled model and its own retrieval
pipeline before assuming any of these gains survive in the submission.
No additional model training was started by this analysis.

Official references: [BioKG evaluation protocol](https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg)
and [OGB train/validation/test rules](https://ogb.stanford.edu/docs/leader_rules/).
