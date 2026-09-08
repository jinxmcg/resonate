"""Read-only TRAIN loss-gradient diagnostic. Protocol: GRADIENT_DIAGNOSTIC.md."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from biokg.joint_operator import from_student, retained_loss
from biokg.mixed_operator import restore_mixed
from biokg.train_biokg_comp import load_teacher, teacher_logits
from biokg.train_dual_operator import TrainStream, tensor_digest


COMPONENTS = ("ce", "kd", "trajectory")
GROUPS = ("entities", "operators", "log_temperature")
BATCHES, BATCH, NEGATIVES, SEED = 512, 2048, 4096, 3601


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_dataset_path(path, dataset):
    """Fail closed on any dataset file other than TRAIN and node counts."""
    path, dataset = Path(path).resolve(), Path(dataset).resolve()
    if path.is_relative_to(dataset):
        relative = str(path.relative_to(dataset))
        if relative not in ("split/random/train.pt", "raw/num-node-dict.csv.gz"):
            raise PermissionError(f"Diagnostic cannot open dataset file: {relative}")
        return relative
    return None


def train_only(dataset):
    dataset = Path(dataset)
    with gzip.open(dataset / "raw/num-node-dict.csv.gz", "rt") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != 1:
        raise ValueError("Expected one node-count record")
    counts = {name: int(value) for name, value in records[0].items()}
    offsets, cursor = {}, 0
    for name in sorted(counts):
        offsets[name] = cursor
        cursor += counts[name]
    train = torch.load(dataset / "split/random/train.pt", map_location="cpu", weights_only=False)
    if set(train) != {"head", "relation", "tail", "head_type", "tail_type"}:
        raise ValueError("Unexpected TRAIN fields; no candidate caches accepted")
    return train, offsets, counts


def real_flat(tensor):
    tensor = tensor.resolve_conj().contiguous()
    return (torch.view_as_real(tensor) if tensor.is_complex() else tensor).reshape(-1)


def gradient_gram(gradients, params):
    """Real-coordinate Euclidean Gram; complex gradients are not complex dots."""
    grams = []
    for index, parameter in enumerate(params):
        vectors = [real_flat(g[index]) if g[index] is not None else None for g in gradients]
        rows = []
        zero = parameter.real.new_zeros(())
        for i in range(3):
            rows.append(torch.stack([
                torch.dot(vectors[i], vectors[j])
                if vectors[i] is not None and vectors[j] is not None else zero
                for j in range(3)]))
        grams.append(torch.stack(rows))
    return torch.stack(grams).detach().cpu().double().numpy()


def gram_metrics(gram):
    gram = np.asarray(gram, np.float64)
    if not np.isfinite(gram).all():
        raise ValueError("Nonfinite gradient Gram")
    norms = np.sqrt(np.maximum(np.diag(gram), 0))
    result = {f"norm_{name}": float(norms[i]) for i, name in enumerate(COMPONENTS)}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        denominator = norms[i] * norms[j]
        result[f"cos_{COMPONENTS[i]}_{COMPONENTS[j]}"] = (
            float(np.clip(gram[i, j] / denominator, -1, 1)) if denominator > 0 else None)
    ranking_sq = max(float(gram[:2, :2].sum()), 0.)
    ranking_norm = math.sqrt(ranking_sq)
    dot = float(gram[0, 2] + gram[1, 2])
    result.update(
        norm_ce_kd=ranking_norm,
        trajectory_over_ce_kd=(float(norms[2] / ranking_norm) if ranking_norm > 0 else None),
        cos_ce_kd_trajectory=(float(np.clip(dot / (ranking_norm * norms[2]), -1, 1))
                              if ranking_norm * norms[2] > 0 else None),
        ce_kd_descent_projection_ratio=(1 + dot / ranking_sq if ranking_sq > 0 else None),
        total_norm=math.sqrt(max(float(gram.sum()), 0.)))
    return result


def inspect_batch(model, batch, target, verify_sum=False):
    """Autograd only: no .backward(), optimizer, parameter edits or clipping."""
    params = (model.E, model.H_b, model.log_tau)
    total, parts = retained_loss(model.training_outputs(*batch), target)
    losses = (parts["ce"], parts["kd_t_squared"], .1 * parts["trajectory"])
    if not torch.isfinite(total):
        raise ValueError("Nonfinite loss")
    gradients = [torch.autograd.grad(loss, params, retain_graph=(i < 2 or verify_sum),
                                     allow_unused=True) for i, loss in enumerate(losses)]
    if verify_sum:
        combined = torch.autograd.grad(total, params)
        for i, expected in enumerate(combined):
            actual = sum((g[i] for g in gradients if g[i] is not None), torch.zeros_like(expected))
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    grams = gradient_gram(gradients, params)
    assert all(p.grad is None for p in params)
    groups = {name: gram_metrics(gram) for name, gram in zip(GROUPS, grams)}
    groups["representation"] = gram_metrics(grams[:2].sum(0))
    groups["all"] = gram_metrics(grams.sum(0))
    return dict(losses={name: float(loss.detach()) for name, loss in zip(COMPONENTS, losses)},
                groups=groups, grams=grams.tolist())


def distribution(values):
    values = np.asarray([value for value in values if value is not None], np.float64)
    if not len(values):
        return dict(count=0, mean=None, median=None, p10=None, p90=None, negative_fraction=None)
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)),
                negative_fraction=float((values < 0).mean()))


def summarize(rows):
    def subset(items):
        return dict(batches=len(items), losses={name: distribution([r["losses"][name] for r in items])
                                               for name in COMPONENTS},
                    groups={group: {metric: distribution([r["groups"][group][metric] for r in items])
                                    for metric in items[0]["groups"][group]}
                            for group in items[0]["groups"]})
    result = {}
    for arm in sorted({r["arm"] for r in rows}):
        selected = [r for r in rows if r["arm"] == arm]
        result[arm] = dict(overall=subset(selected), directions={}, family_direction={})
        for direction in ("head", "tail"):
            items = [r for r in selected if r["direction"] == direction]
            if items:
                result[arm]["directions"][direction] = subset(items)
        for key in sorted({(r["family"], r["direction"]) for r in selected}):
            items = [r for r in selected if (r["family"], r["direction"]) == key]
            result[arm]["family_direction"]["/".join(key)] = subset(items)
    return result


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
        raise RuntimeError("This protocol requires the local GTX 1080 Ti")
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
    prior_path = repo / "biokg/results/h35f/campaign_s0/prerun.json"
    prior = json.loads(prior_path.read_text())
    old_path = repo / "biokg/checkpoints/dist_T2_s0.pt"
    new_path = repo / "biokg/results/h35f/campaign_s0/single.pt"
    teacher_dir = Path("/mnt/geocore/wiki_pull/h24/dense")
    teacher_paths = [teacher_dir / f"sparse_s{i}.pt" for i in range(10)]
    expected = {old_path: prior["student_sha256"],
                new_path: "df7c783e7b8a6110f126bca1c60d89209735accbaa5ffc0b362f2cab52d9d2f1",
                dataset / "split/random/train.pt": prior["input_sha256"][str(dataset / "split/random/train.pt")]}
    expected.update({path: prior["teacher_sha256"][path.name] for path in teacher_paths})
    for relative in ("resonate.py", "biokg/train_biokg_comp.py", "biokg/train_dual_operator.py",
                     "biokg/dual_operator.py", "biokg/joint_operator.py", "biokg/mixed_operator.py"):
        expected[repo / relative] = prior["source_hashes"][relative]
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"Prior artifact changed: {path}")
    expected[dataset / "raw/num-node-dict.csv.gz"] = sha256(dataset / "raw/num-node-dict.csv.gz")
    for path in (prior_path, Path(__file__), repo / "biokg/test_gradient_diagnostic.py",
                 repo / "biokg/GRADIENT_DIAGNOSTIC.md"):
        expected[path] = sha256(path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def log(event):
        print(json.dumps(event, allow_nan=False), flush=True)
        with (out / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")

    write("prerun.json", dict(protocol="TRAIN-only loss gradients", batches=BATCHES, batch=BATCH,
                              negatives=NEGATIVES, seed=SEED, weights=[1., 1., .1], kd_temperature=2.,
                              device=torch.cuda.get_device_name(), torch=torch.__version__, numpy=np.__version__,
                              optimizer_updates=0, lr=0., mrr_evaluated=False,
                              input_source_sha256={str(path): digest for path, digest in expected.items()}))
    log(dict(stage="loading_own_models", lr=0., mrr_evaluated=False))
    train, offset, counts = train_only(dataset)
    stream = TrainStream(train, offset, counts, seed=SEED)
    n_rel, n_ent = 2 * stream.n_rel_base, sum(counts.values())
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    new = torch.load(new_path, map_location="cpu", weights_only=False)
    for checkpoint in (old, new):
        if checkpoint["offset"] != offset or checkpoint["n_rel"] != n_rel:
            raise ValueError("Student indexing mismatch")
    if new["mode"] != "single":
        raise ValueError("Only the A-only endpoint is in scope")
    models = {"original_A": from_student(old, "single", "cuda"),
              "improved_A": restore_mixed(new, "cuda").a.requires_grad_(True)}
    del old, new
    teachers = []
    for seed, path in enumerate(teacher_paths):
        metadata = torch.load(path, map_location="cpu", weights_only=False)
        if (metadata["offset"] != offset or metadata["n_rel"] != n_rel
                or metadata["args"]["seed"] != seed or metadata["args"].get("distill")):
            raise ValueError("Teacher indexing/recipe mismatch")
        del metadata
        teachers.append(load_teacher(path, n_ent, n_rel, torch.device("cuda")))
    all_models = dict(models, **{f"own_teacher_{i}": model for i, model in enumerate(teachers)})
    before = {name: tensor_digest(model.state_dict()) for name, model in all_models.items()}
    families = [str(train["head_type"][rows[0]]) + "-" + str(train["tail_type"][rows[0]])
                for rows in stream.by_rel]
    rows, relation_counts = [], np.zeros(n_rel, dtype=np.int64)
    torch.cuda.reset_peak_memory_stats()
    started, last_log = time.monotonic(), time.monotonic()
    for step in range(1, BATCHES + 1):
        source, positive, relation, negatives = stream.sample(BATCH, NEGATIVES)
        relation_counts[relation] += 1
        source, positive, negatives = [torch.as_tensor(x, device="cuda") for x in (source, positive, negatives)]
        relations = torch.full((BATCH,), relation, device="cuda", dtype=torch.long)
        batch = source, relations, positive, negatives
        target = teacher_logits(teachers, *batch)
        for arm, model in models.items():
            record = dict(arm=arm, batch_index=step, relation=relation,
                          family=families[relation % stream.n_rel_base],
                          direction="tail" if relation < stream.n_rel_base else "head",
                          **inspect_batch(model, batch, target, verify_sum=(step == 1)))
            rows.append(record)
            with (out / "batches.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        del target
        if step == 1 or step % 64 == 0 or time.monotonic() - last_log >= 45:
            elapsed = time.monotonic() - started
            log(dict(stage="diagnostic", batches_done=step, batches_total=BATCHES,
                     elapsed_seconds=elapsed, estimated_remaining_seconds=elapsed / step * (BATCHES - step),
                     lr=0., optimizer_updates=0, mrr_evaluated=False,
                     latest={r["arm"]: r["groups"]["representation"] for r in rows[-2:]}))
            last_log = time.monotonic()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    after = {name: tensor_digest(model.state_dict()) for name, model in all_models.items()}
    assert before == after, "A model state changed during diagnostics"
    assert all(p.grad is None for model in all_models.values() for p in model.parameters())
    assert all(not p.requires_grad for model in teachers for p in model.parameters())
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"Input/source changed during diagnostic: {path}")
    result = dict(arms=summarize(rows), directed_relation_batch_counts=relation_counts.tolist(),
                  directed_relations_seen=int((relation_counts > 0).sum()),
                  stream_sha256=stream.digest.hexdigest(), diagnostic_seconds=elapsed,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), optimizer_updates=0,
                  dataset_files_opened=sorted(opened), validation_loaded=False, test_loaded=False,
                  model_state_hashes_before=before, model_state_hashes_after=after,
                  all_states_unchanged=True, all_parameter_grads_absent=True,
                  summed_gradient_parity_passed=True)
    write("summary.json", result)
    log(dict(stage="complete", batches=BATCHES, seconds=elapsed, optimizer_updates=0,
             lr=0., mrr_evaluated=False, all_states_unchanged=True,
             peak_allocated_bytes=result["peak_allocated_bytes"]))


if __name__ == "__main__":
    main()
