"""TRAIN-only confidence/ordering diagnostic; never update a model checkpoint."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from biokg.gradient_diagnostic import check_dataset_path, sha256, train_only
from biokg.joint_operator import from_student
from biokg.mixed_operator import restore_mixed
from biokg.train_biokg_comp import load_teacher, teacher_logits
from biokg.train_dual_operator import TrainStream, tensor_digest


BATCHES, BATCH, NEGATIVES, SEED = 128, 512, 4096, 3602
TEMPERATURE, LOW, HIGH, ITERATIONS, CHUNK = 2., 1 / 16, 16., 24, 1024


def fit_mask(source, positive, directed_relation, n_rel_base):
    """Stable hash of original (h,r,t); reciprocal queries and repeats stay together."""
    h, t = (source, positive) if directed_relation < n_rel_base else (positive, source)
    h, t = np.asarray(h, np.uint64), np.asarray(t, np.uint64)
    r = np.uint64(directed_relation % n_rel_base)
    x = h * np.uint64(0x9E3779B185EBCA87) ^ t * np.uint64(0xC2B2AE3D27D4EB4F)
    x ^= r + np.uint64(SEED)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return (x & np.uint64(1)) == 0


def prepare_scores(logits):
    return (logits - logits.mean(-1, keepdim=True)) / TEMPERATURE


def kd_rows(x, probabilities, p_log_p, beta=1.):
    """T-squared KL; x contains centered student logits divided by T."""
    if torch.is_tensor(beta) and beta.ndim == 1:
        beta = beta[:, None]
    return TEMPERATURE ** 2 * (p_log_p - (probabilities * (x * beta).log_softmax(-1)).sum(-1))


def scale_derivative(x, probabilities, beta):
    if torch.is_tensor(beta) and beta.ndim == 1:
        beta = beta[:, None]
    return ((torch.softmax(x * beta, -1) - probabilities) * x).sum(-1)


@torch.no_grad()
def fit_global_scale(x, probabilities, indices, progress=None):
    """Convex positive scale fit using only supplied calibration-row indices."""
    if len(indices) == 0:
        raise ValueError("No calibration rows")

    def derivative(beta):
        total = torch.zeros((), dtype=torch.float64, device=x.device)
        for start in range(0, len(indices), CHUNK):
            rows = indices[start:start + CHUNK]
            total += scale_derivative(x[rows], probabilities[rows], beta).sum(dtype=torch.float64)
        result = float(total / len(indices))
        if not np.isfinite(result):
            raise ValueError("Nonfinite calibration derivative")
        return result

    before = derivative(1.)
    lower, upper = derivative(LOW), derivative(HIGH)
    if abs(before) < 1e-10:
        beta = 1.
    elif lower >= 0:
        beta = LOW
    elif upper <= 0:
        beta = HIGH
    else:
        lo, hi = LOW, HIGH
        for step in range(ITERATIONS):
            mid = (lo + hi) / 2
            if derivative(mid) > 0:
                hi = mid
            else:
                lo = mid
            if progress and (step + 1) % 8 == 0:
                progress(step + 1, ITERATIONS)
        beta = (lo + hi) / 2
    return dict(beta=beta, derivative_before=before, derivative_after=derivative(beta),
                boundary=beta in (LOW, HIGH))


@torch.no_grad()
def oracle_row_scales(x, probabilities, p_log_p):
    """Optimistic same-row fit, NOT a learned/deployable calibration policy."""
    lo, hi = torch.full_like(p_log_p, LOW), torch.full_like(p_log_p, HIGH)
    for _ in range(ITERATIONS):
        mid = (lo + hi) / 2
        positive = scale_derivative(x, probabilities, mid) > 0
        hi = torch.where(positive, mid, hi)
        lo = torch.where(positive, lo, mid)
    beta = (lo + hi) / 2
    beta = torch.where(scale_derivative(x, probabilities, LOW) >= 0, LOW, beta)
    beta = torch.where(scale_derivative(x, probabilities, HIGH) <= 0, HIGH, beta)
    baseline = kd_rows(x, probabilities, p_log_p)
    fitted = kd_rows(x, probabilities, p_log_p, beta)
    # Flat distributions or last-bit root errors should not worsen the oracle.
    beta = torch.where((fitted > baseline) | (x.square().sum(-1) < 1e-12), 1., beta)
    return beta


def ranking_stats(student, teacher):
    """Candidate-symmetric, tie-aware top-score-set overlap; no MRR benchmark."""
    student_top = student == student.max(-1, keepdim=True).values
    teacher_top = teacher == teacher.max(-1, keepdim=True).values
    s, t = student - student.mean(-1, keepdim=True), teacher - teacher.mean(-1, keepdim=True)
    den = s.norm(dim=-1) * t.norm(dim=-1)
    correlation = (s * t).sum(-1) / den.clamp_min(1e-12)
    return dict(top1_overlap=(student_top & teacher_top).any(-1),
                student_positive_top=student_top[:, 0], teacher_positive_top=teacher_top[:, 0],
                centered_score_cosine=torch.where(den > 1e-12, correlation, torch.nan))


def describe(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0, mean=None, median=None, p10=None, p90=None)
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)))


def summarize_records(records):
    result = {}
    for arm, record in records.items():
        result[arm] = {}
        for split in ("fit", "report"):
            mask = record["fit"] if split == "fit" else ~record["fit"]
            row = {key: describe(value[mask]) for key, value in record.items()
                   if key not in ("fit", "relation", "batch_index")}
            for name in ("global_kd", "oracle_kd"):
                base = row["baseline_kd"]["mean"]
                row[name + "_relative_reduction"] = (
                    1 - row[name]["mean"] / base if row[name]["mean"] is not None and base > 0 else None)
            row["teacher_positive_only_count"] = int((mask & record["teacher_positive_top"]
                                                      & ~record["student_positive_top"]).sum())
            row["student_positive_only_count"] = int((mask & record["student_positive_top"]
                                                      & ~record["teacher_positive_top"]).sum())
            result[arm][split] = row
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available() or "1080 Ti" not in torch.cuda.get_device_name():
        raise RuntimeError("Use the local GTX 1080 Ti")
    if torch.cuda.mem_get_info()[0] < 7 * 1024 ** 3:
        raise RuntimeError("Need at least 7 GiB free; leave other jobs untouched")
    repo = Path(__file__).resolve().parents[1]
    dataset = Path("/mnt/geocore/geocore/data_ogb/ogbl_biokg")
    opened = set()

    def audit(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes)):
            path = arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]
            relative = check_dataset_path(path, dataset)
            if relative is not None:
                opened.add(relative)

    sys.addaudithook(audit)
    prior_path = repo / "biokg/results/gradient_diagnostic/s0/prerun.json"
    expected = json.loads(prior_path.read_text())["input_source_sha256"]
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"Prior input/source changed: {path}")
    for path in (prior_path, Path(__file__), repo / "biokg/test_kd_calibration_diagnostic.py",
                 repo / "biokg/KD_CALIBRATION_DIAGNOSTIC.md"):
        expected[str(path)] = sha256(path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def log(**event):
        event.update(lr=0., model_updates=0, mrr_evaluated=False)
        print(json.dumps(event, allow_nan=False), flush=True)
        with (out / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")

    write("prerun.json", dict(batches=BATCHES, batch=BATCH, negatives=NEGATIVES, seed=SEED,
                              temperature=TEMPERATURE, calibration_bounds=[LOW, HIGH], iterations=ITERATIONS,
                              chunk=CHUNK, device=torch.cuda.get_device_name(), torch=torch.__version__,
                              numpy=np.__version__, model_updates=0, external_weights=False,
                              input_source_sha256=expected))
    log(stage="loading_own_models")
    train, offset, counts = train_only(dataset)
    stream = TrainStream(train, offset, counts, seed=SEED)
    n_ent, n_rel = sum(counts.values()), 2 * stream.n_rel_base
    old = torch.load(repo / "biokg/checkpoints/dist_T2_s0.pt", map_location="cpu", weights_only=False)
    new = torch.load(repo / "biokg/results/h35f/campaign_s0/single.pt", map_location="cpu", weights_only=False)
    for checkpoint in (old, new):
        if checkpoint["offset"] != offset or checkpoint["n_rel"] != n_rel:
            raise ValueError("Index mismatch")
    if new["mode"] != "single":
        raise ValueError("Only A-only snapshots accepted")
    models = {"original_A": from_student(old, "single", "cuda").eval().requires_grad_(False),
              "improved_A": restore_mixed(new, "cuda").a}
    del old, new
    teachers = [load_teacher(Path("/mnt/geocore/wiki_pull/h24/dense") / f"sparse_s{i}.pt",
                             n_ent, n_rel, torch.device("cuda")) for i in range(10)]
    all_models = dict(models, **{f"own_teacher_{i}": model for i, model in enumerate(teachers)})
    before = {name: tensor_digest(model.state_dict()) for name, model in all_models.items()}
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    total = BATCHES * BATCH
    scores = {arm: torch.empty(total, NEGATIVES + 1, device="cuda") for arm in models}
    probabilities = torch.empty(total, NEGATIVES + 1, device="cuda")
    p_log_p = torch.empty(total, device="cuda")
    fit = np.empty(total, bool)
    relations = np.empty(total, np.int64)
    batches = np.repeat(np.arange(BATCHES), BATCH)
    records = {arm: {key: np.empty(total, bool if key.endswith("top") or key == "top1_overlap" else np.float64)
                     for key in ("top1_overlap", "student_positive_top", "teacher_positive_top", "centered_score_cosine")}
               for arm in models}
    for step in range(BATCHES):
        source, positive, relation, negatives = stream.sample(BATCH, NEGATIVES)
        sl = slice(step * BATCH, (step + 1) * BATCH)
        fit[sl] = fit_mask(source, positive, relation, stream.n_rel_base)
        relations[sl] = relation
        source, positive, negatives = [torch.as_tensor(x, device="cuda") for x in (source, positive, negatives)]
        rel = torch.full((BATCH,), relation, device="cuda", dtype=torch.long)
        batch = source, rel, positive, negatives
        target = teacher_logits(teachers, *batch)
        if not torch.isfinite(target).all():
            raise ValueError("Nonfinite teacher scores")
        log_p = (target / TEMPERATURE).log_softmax(-1)
        probabilities[sl] = log_p.exp()
        p_log_p[sl] = (probabilities[sl] * log_p).sum(-1)
        for arm, model in models.items():
            raw = model.training_outputs(*batch)[0]
            if not torch.isfinite(raw).all():
                raise ValueError("Nonfinite student scores")
            scores[arm][sl] = prepare_scores(raw)
            for key, value in ranking_stats(raw, target).items():
                records[arm][key][sl] = value.cpu().numpy()
        if step == 0 or (step + 1) % 32 == 0:
            log(stage="cache_train_scores", batches_done=step + 1, batches_total=BATCHES,
                elapsed_seconds=time.monotonic() - started)
    del raw, target, log_p
    fit_indices = torch.as_tensor(np.flatnonzero(fit), device="cuda")
    report_indices = torch.as_tensor(np.flatnonzero(~fit), device="cuda")
    fits = {}
    for arm, x in scores.items():
        log(stage="fit_global_scale", arm=arm, fit_queries=len(fit_indices), report_queries=len(report_indices))
        fits[arm] = fit_global_scale(x, probabilities, fit_indices,
                                    lambda done, count: log(stage="fit_global_scale", arm=arm,
                                                            iterations_done=done, iterations_total=count))
        beta = fits[arm]["beta"]
        log(stage="global_scale_fitted", arm=arm, **fits[arm])
        record = records[arm]
        record.update(fit=fit.copy(), relation=relations.copy(), batch_index=batches.copy())
        for key in ("baseline_kd", "global_kd", "oracle_kd", "oracle_beta"):
            record[key] = np.full(total, np.nan)
        for start in range(0, total, CHUNK):
            sl = slice(start, start + CHUNK)
            record["baseline_kd"][sl] = kd_rows(x[sl], probabilities[sl], p_log_p[sl]).cpu().numpy()
            record["global_kd"][sl] = kd_rows(x[sl], probabilities[sl], p_log_p[sl], beta).cpu().numpy()
        if any(not np.isfinite(record[key]).all() for key in ("baseline_kd", "global_kd")):
            raise ValueError("Nonfinite distillation divergence")
        # Per-query best scaling is an optimistic diagnostic on REPORT TRAIN rows,
        # not a policy fitted on or deployed to validation/test inputs.
        for start in range(0, len(report_indices), CHUNK):
            ids = report_indices[start:start + CHUNK]
            local_x, local_p, entropy = x[ids], probabilities[ids], p_log_p[ids]
            oracle_beta = oracle_row_scales(local_x, local_p, entropy)
            oracle_loss = kd_rows(local_x, local_p, entropy, oracle_beta)
            if not torch.isfinite(oracle_loss).all() or not torch.isfinite(oracle_beta).all():
                raise ValueError("Nonfinite oracle")
            # Positive scaling must preserve each query's top-score set.
            top = local_x == local_x.max(-1, keepdim=True).values
            # Verify ordering in float64 so multiplication does not introduce
            # accidental float32 ties between adjacent representable scores.
            scaled = local_x.double() * oracle_beta.double()[:, None]
            assert torch.equal(top, scaled == scaled.max(-1, keepdim=True).values)
            scaled = local_x.double() * beta
            assert torch.equal(top, scaled == scaled.max(-1, keepdim=True).values)
            ii = ids.cpu().numpy()
            record["oracle_beta"][ii] = oracle_beta.cpu().numpy()
            record["oracle_kd"][ii] = oracle_loss.cpu().numpy()
            if start == 0 or (start // CHUNK + 1) % 8 == 0:
                log(stage="query_scale_upper_bound", arm=arm,
                    queries_done=min(start + CHUNK, len(report_indices)), queries_total=len(report_indices))
        np.savez_compressed(out / f"{arm}_records.npz", **record)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    after = {name: tensor_digest(model.state_dict()) for name, model in all_models.items()}
    assert before == after
    assert all(not p.requires_grad and p.grad is None for model in all_models.values() for p in model.parameters())
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"Input/source changed during run: {path}")
    summary = dict(arms=summarize_records(records), global_fits=fits, diagnostic_seconds=elapsed,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(), model_updates=0,
                   validation_loaded=False, test_loaded=False, dataset_files_opened=sorted(opened),
                   all_states_unchanged=True, all_parameter_grads_absent=True, top_score_sets_unchanged=True,
                   model_state_hashes_before=before, model_state_hashes_after=after,
                   stream_sha256=stream.digest.hexdigest(), directed_relations_seen=len(np.unique(relations)),
                   record_sha256={arm: sha256(out / f"{arm}_records.npz") for arm in records})
    write("summary.json", summary)
    log(stage="complete", seconds=elapsed, all_states_unchanged=True,
        peak_allocated_bytes=summary["peak_allocated_bytes"])


if __name__ == "__main__":
    main()
