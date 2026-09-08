"""H35D frozen VALID diagnostic; see H35D.md. No training or TEST option."""

import argparse
import csv
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
from ogb.linkproppred import Evaluator

from biokg.relation_analogy import sha256
from biokg.train_biokg_comp import globalize, load, load_teacher
from biokg.train_dual_operator import official_ranks, paired_interval, rank_summary, tensor_digest


def validation_only(root):
    return load(root, include_test=False)


@torch.inference_mode()
def candidate_scores(model, source, relation, candidates):
    """Candidate-position symmetric version of the original raw-logit scorer."""
    query = model.out(model.hop(model.embed(source), relation), relation)
    scores = torch.einsum("bm,bcm->bc", query, model.rows(candidates).conj()).real
    scores = scores * model.log_tau.exp()
    if model.b is not None:
        scores = scores + model.b[candidates]
    return scores


def doubled_candidate_ranks(scores):
    """Twice the descending average-tie rank, exactly integer-valued."""
    if scores.ndim != 2 or scores.shape[1] < 2 or not torch.isfinite(scores).all():
        raise ValueError("Expected finite batch x candidates scores, at least two candidates")
    ordered = scores.sort(dim=1).values.contiguous()
    scores = scores.contiguous()
    lo = torch.searchsorted(ordered, scores, right=False)
    hi = torch.searchsorted(ordered, scores, right=True)
    return 2 * scores.shape[1] + 1 - lo - hi


@torch.inference_mode()
def fusion_batch(teachers, source, relation, candidates, evaluator):
    if not teachers:
        raise ValueError("At least one teacher required")
    logits_sum = rank_sum = None
    teacher_ranks = []
    for teacher in teachers:
        scores = candidate_scores(teacher, source, relation, candidates)
        ranks2 = doubled_candidate_ranks(scores)
        # Compare this all-candidate rank definition with the official evaluator.
        positive_ranks, _ = official_ranks(scores, evaluator)
        torch.testing.assert_close(ranks2[:, 0].float() / 2, positive_ranks, atol=0, rtol=0)
        teacher_ranks.append(positive_ranks)
        logits_sum = scores if logits_sum is None else logits_sum + scores
        rank_sum = ranks2 if rank_sum is None else rank_sum + ranks2
    # Exactly representable integers for ten teachers x 501 candidates.
    if 2 * len(teachers) * candidates.shape[1] >= 2**24:
        raise ValueError("Rank sum exceeds guaranteed exact fp32 integer range")
    return logits_sum / len(teachers), -rank_sum.float(), torch.stack(teacher_ranks)


def model_hashes(models):
    return {name: tensor_digest(model.state_dict()) for name, model in models.items()}


def assert_frozen(models):
    assert all(not model.training for model in models.values())
    assert all(not p.requires_grad and p.grad is None
               for model in models.values() for p in model.parameters())


@torch.inference_mode()
def evaluate(teachers, student, part, offset, n_rel, log, chunk=128):
    ev = Evaluator("ogbl-biokg")
    h, r, t = globalize(part, offset)
    assert len(h) > 0
    dev = student.E.device
    records = {name: [] for name in ("mean_logits", "rank_fusion", "student", "teachers")}
    totals = {name: 0. for name in records if name != "teachers"}
    started = last_log = time.monotonic()
    processed = 0
    for direction in (0, 1):
        source, target = (h, t) if direction == 0 else (t, h)
        types = part["tail_type" if direction == 0 else "head_type"]
        negatives = part["tail_neg" if direction == 0 else "head_neg"]
        assert negatives.shape == (len(h), 500)
        for start in range(0, len(h), chunk):
            sl = slice(start, start + chunk)
            off = np.array([offset[name] for name in types[sl]])
            candidates = np.concatenate([target[sl, None], negatives[sl] + off[:, None]], axis=1)
            args = (torch.as_tensor(source[sl], device=dev),
                    torch.as_tensor(r[sl] + direction * (n_rel // 2), device=dev),
                    torch.as_tensor(candidates, device=dev))
            mean, fused, individual = fusion_batch(teachers, *args, ev)
            scores = dict(mean_logits=mean, rank_fusion=fused,
                          student=candidate_scores(student, *args))
            for name, value in scores.items():
                ranks, _ = official_ranks(value, ev)
                arr = ranks.cpu().numpy()
                records[name].append(arr)
                totals[name] += (1 / arr.astype(np.float64)).sum()
            records["teachers"].append(individual.cpu().numpy())
            processed += len(candidates)
            now = time.monotonic()
            if now - last_log >= 45 or start + chunk >= len(h):
                log(dict(stage="validation_progress", queries=processed, total_queries=2 * len(h),
                         direction="tail" if direction == 0 else "head", seconds=now - started,
                         lr=None, partial_mrr={name: float(value / processed)
                                               for name, value in totals.items()}))
                last_log = now
    result = {name: np.concatenate(rows, axis=1 if name == "teachers" else 0)
              for name, rows in records.items()}
    result.update(triple_index=np.tile(np.arange(len(h)), 2),
                  direction=np.repeat([0, 1], len(h)), relation=np.tile(r, 2))
    return result, time.monotonic() - started


def paired_comparison(treatment, control):
    delta = 1 / treatment.astype(np.float64) - 1 / control.astype(np.float64)
    return dict(delta_mrr=float(delta.mean()),
                bootstrap_95=paired_interval(delta, [.025, .975], np.random.default_rng(3512)),
                improved_queries=int((treatment < control).sum()),
                worsened_queries=int((treatment > control).sum()),
                equal_rank_queries=int((treatment == control).sum()),
                recovered_top1=int(((treatment == 1) & (control > 1)).sum()),
                lost_top1=int(((treatment > 1) & (control == 1)).sum()))


def summarize(records, relation_names):
    ranks = {name: records[name] for name in ("student", "mean_logits", "rank_fusion")}
    ranks.update({f"teacher_s{i}": row for i, row in enumerate(records["teachers"])})
    metrics = {name: rank_summary(rank) for name, rank in ranks.items()}
    primary = paired_comparison(ranks["rank_fusion"], ranks["mean_logits"])
    primary["investigate_rank_kd"] = primary["delta_mrr"] >= .001 and primary["bootstrap_95"][0] > 0
    comparisons = {name + "_vs_student": paired_comparison(ranks[name], ranks["student"])
                   for name in ("mean_logits", "rank_fusion")}
    family = np.array([relation_names[int(r)].split("_")[0] for r in records["relation"]])
    slices = []
    for kind, values in (("direction", records["direction"]), ("family", family),
                         ("relation", records["relation"])):
        for value in np.unique(values):
            mask = values == value
            slices.append(dict(kind=kind, value=str(value), queries=int(mask.sum()),
                               metrics={name: rank_summary(rank[mask]) for name, rank in ranks.items()}))
    pairwise = []
    for i, a in enumerate(records["teachers"]):
        for j in range(i + 1, len(records["teachers"])):
            b = records["teachers"][j]
            pairwise.append(dict(a=i, b=j, same_positive_rank=float((a == b).mean()),
                                 positive_top1_disagreement=float(((a == 1) != (b == 1)).mean())))
    return dict(metrics=metrics, primary_rank_vs_logits=primary,
                descriptive_comparisons=comparisons, descriptive_slices=slices,
                descriptive_teacher_pairs=pairwise)


@torch.inference_mode()
def smoke(device):
    from resonate import ResonatE
    torch.manual_seed(3511)
    models = {str(i): ResonatE(93773, 102, k=12, block=True, block_size=4)
              .to(device).eval().requires_grad_(False) for i in range(11)}
    before = model_hashes(models)
    teachers, student = list(models.values())[:10], models["10"]
    args = (torch.arange(128, device=device), torch.arange(128, device=device) % 102,
            torch.randint(93773, (128, 501), device=device))
    ev = Evaluator("ogbl-biokg")
    mean, fused, ranks = fusion_batch(teachers, *args, ev)
    permutation = torch.randperm(501, device=device)
    perm_mean, perm_fused, _ = fusion_batch(teachers, args[0], args[1], args[2][:, permutation], ev)
    torch.testing.assert_close(perm_mean, mean[:, permutation], atol=0, rtol=0)
    torch.testing.assert_close(perm_fused, fused[:, permutation], atol=0, rtol=0)
    official_ranks(candidate_scores(student, *args), ev)
    assert ranks.shape == (10, 128)
    assert before == model_hashes(models)
    assert_frozen(models)
    print(json.dumps(dict(stage="synthetic_smoke_passed", models=11, candidates=501,
                          batch=128, frozen=True, candidate_permutation_exact=True,
                          peak_allocated_bytes=(torch.cuda.max_memory_allocated()
                                                if torch.device(device).type == "cuda" else None))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="biokg/results/h35d/campaign_s0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.smoke:
        smoke(args.device)
        return
    repo = Path(__file__).resolve().parents[1]
    prior_path = repo / "biokg/results/h35c/campaign_s0/prerun.json"
    prior = json.loads(prior_path.read_text())
    dataset_root = Path(prior["args"]["data_root"])
    teacher_dir = Path(prior["args"]["teacher_dir"])
    checkpoints = {f"teacher_s{i}": teacher_dir / f"sparse_s{i}.pt" for i in range(10)}
    checkpoints["student"] = repo / "biokg/checkpoints/dist_T2_s0.pt"
    expected = {f"teacher_s{i}": prior["teacher_sha256"][f"sparse_s{i}.pt"] for i in range(10)}
    expected["student"] = prior["student_sha256"]
    inputs = {str(path): expected[name] for name, path in checkpoints.items()}
    inputs.update({str(dataset_root / "ogbl_biokg" / name): digest
                   for name, digest in prior["data_hashes"].items()})
    reference = repo / "biokg/results/h35/campaign_s0/frozen_queries.npz"
    inputs[str(reference)] = prior["reference_ranks_sha256"]
    inputs[prior["args"]["checksums"]] = prior["checksums_sha256"]
    assert all(sha256(path) == digest for path, digest in inputs.items()), "Input hash mismatch"
    assert all(sha256(repo / path) == digest for path, digest in prior["source_hashes"].items()), "H35C source changed"
    source_files = ["biokg/H35D.md", "biokg/teacher_fusion_diagnostic.py",
                    "biokg/test_teacher_fusion_diagnostic.py", "biokg/relation_analogy.py",
                    *prior["source_hashes"]]
    source_hashes = {name: sha256(repo / name) for name in source_files}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(protocol="H35D", args=vars(args), inputs_sha256=inputs,
                   prior_receipt_sha256=sha256(prior_path), source_hashes=source_hashes,
                   torch=torch.__version__, numpy=np.__version__,
                   device=torch.cuda.get_device_name() if args.device == "cuda" else args.device,
                   started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   training=False, lr=None, test_loaded=False, weight_fitting=False)
    (out / "prerun.json").write_text(json.dumps(receipt, indent=2) + "\n")
    def log(event):
        print(json.dumps(event), flush=True)
        with (out / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
    log(dict(stage="inputs_verified", models=11, lr=None))
    split, offset, n_ent, _, _ = validation_only(str(dataset_root))
    n_rel = 2 * (int(split["train"]["relation"].max()) + 1)
    assert (n_ent, n_rel, len(split["valid"]["head"])) == (93773, 102, 162886)
    assert set(split) == {"train", "valid"}
    models = {}
    for name, path in checkpoints.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["offset"] == offset and ck["n_rel"] == n_rel
        assert set(ck["model"]) == {"E", "H", "log_tau"}
        if name.startswith("teacher"):
            assert ck["args"]["seed"] == int(name.rsplit("s", 1)[1]) and not ck["args"].get("distill")
        del ck
        models[name] = load_teacher(path, n_ent, n_rel, torch.device(args.device))
        log(dict(stage="model_loaded", model=name, lr=None))
    before = model_hashes(models)
    assert_frozen(models)
    teachers = [models[f"teacher_s{i}"] for i in range(10)]
    records, seconds = evaluate(teachers, models["student"], split["valid"], offset, n_rel, log)
    np.savez_compressed(out / "queries.npz", **records)
    after = model_hashes(models)
    assert before == after, "Frozen model state changed"
    assert_frozen(models)
    assert all(sha256(path) == digest for path, digest in inputs.items())
    assert all(sha256(repo / name) == digest for name, digest in source_hashes.items())
    with np.load(reference, allow_pickle=False) as frozen:
        original = frozen["rank"]
    reference_equal = np.array_equal(original, records["student"])
    audit = dict(state_hashes_before=before, state_hashes_after=after,
                 all_states_unchanged=True, all_gradients_absent=True, all_requires_grad_false=True,
                 test_loaded=False, input_and_source_hashes_unchanged=True,
                 student_reference_exact=reference_equal,
                 student_reference_rank_mismatches=int((original != records["student"]).sum()),
                 query_sha256=sha256(out / "queries.npz"))
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    assert reference_equal, "Student scorer differs from archived H35: audit before interpreting"
    mapping = dataset_root / "ogbl_biokg/mapping/relidx2relname.csv.gz"
    with gzip.open(mapping, "rt") as stream:
        relation_names = {int(row[0]): row[1] for row in list(csv.reader(stream))[1:]}
    result = summarize(records, relation_names)
    result.update(evaluation_seconds=seconds, test_loaded=False, training=False,
                  peak_allocated_bytes=(torch.cuda.max_memory_allocated() if args.device == "cuda" else None),
                  caveat="Adaptive VALID diagnostic, not deployable-student gain or RelEns reproduction")
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    log(dict(stage="complete", seconds=seconds, metrics=result["metrics"],
             primary=result["primary_rank_vs_logits"], audit_passed=True, lr=None))


if __name__ == "__main__":
    main()
