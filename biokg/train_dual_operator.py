"""H35 fixed four-arm operator adaptation campaign. See biokg/H35.md.

No test option; independent fixed-budget runs; no inference-time ensemble.
"""

import argparse
import csv
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ogb.linkproppred import Evaluator

from biokg.dual_operator import MODES, DualOperator, branch_b_responsibility, from_checkpoint
from biokg.relation_analogy import sha256
from biokg.train_biokg_comp import globalize, load


def validation_only(root):
    return load(root, include_test=False)


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def frozen_digest(model):
    return tensor_digest({name: p for name, p in model.named_parameters() if not p.requires_grad})


class TrainStream:
    def __init__(self, train, offset, num_nodes, seed=3500):
        self.h, self.r, self.t = globalize(train, offset)
        self.n_rel_base = int(self.r.max()) + 1
        self.by_rel = [np.flatnonzero(self.r == r) for r in range(self.n_rel_base)]
        weights = np.array([len(x) for x in self.by_rel], np.float64)
        self.weights = weights / weights.sum()
        self.ranges = []
        for rows in self.by_rel:
            i = rows[0]
            head_type, tail_type = train["head_type"][i], train["tail_type"][i]
            self.ranges.append(((offset[head_type], offset[head_type] + int(num_nodes[head_type])),
                                (offset[tail_type], offset[tail_type] + int(num_nodes[tail_type]))))
        self.rng = np.random.default_rng(seed)
        self.digest = hashlib.sha256()

    def sample(self, batch, negative_count):
        ri = int(self.rng.choice(self.n_rel_base, p=self.weights))
        rows = self.by_rel[ri]
        idx = rows[self.rng.integers(0, len(rows), size=batch)]
        forward = self.rng.random() < .5
        source, target = (self.h[idx], self.t[idx]) if forward else (self.t[idx], self.h[idx])
        relation = ri if forward else ri + self.n_rel_base
        lo, hi = self.ranges[ri][1 if forward else 0]
        neg = self.rng.integers(lo, hi, size=negative_count)
        for arr in (np.array([relation], np.int64), source, target, neg):
            self.digest.update(arr.tobytes())
        return source, target, relation, neg


def official_ranks(scores, evaluator):
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite evaluation scores")
    sp, sn = scores[:, 0], scores[:, 1:]
    gt, ge = (sn > sp[:, None]).sum(1), (sn >= sp[:, None]).sum(1)
    rank = 1 + (gt + ge).float() / 2
    rr = evaluator.eval({"y_pred_pos": sp, "y_pred_neg": sn})["mrr_list"]
    torch.testing.assert_close(rr, rank.reciprocal(), atol=0, rtol=0)
    return rank, rr


def rank_summary(rank):
    rank = rank.astype(np.float64)
    return dict(mrr=float((1 / rank).mean()), hits1=float((rank <= 1).mean()),
                hits10=float((rank <= 10).mean()), queries=len(rank))


@torch.inference_mode()
def evaluate(model, part, offset, n_rel, chunk=128, diagnostics=True):
    was_training = model.training
    model.eval()
    h, r, t = globalize(part, offset)
    dev = model.E.device
    ev = Evaluator("ogbl-biokg")
    records = {}
    for direction in (0, 1):
        src, dst = (h, t) if direction == 0 else (t, h)
        target_types = part["tail_type"] if direction == 0 else part["head_type"]
        neg = part["tail_neg"] if direction == 0 else part["head_neg"]
        for start in range(0, len(h), chunk):
            sl = slice(start, start + chunk)
            off = np.array([offset[x] for x in target_types[sl]])
            cand = np.concatenate([dst[sl, None], neg[sl] + off[:, None]], axis=1)
            scores, a, b, cosine = model.candidate_scores(
                torch.as_tensor(src[sl], device=dev),
                torch.as_tensor(r[sl] + direction * (n_rel // 2), device=dev),
                torch.as_tensor(cand, device=dev))
            rank, rr = official_ranks(scores, ev)
            values = dict(rank=rank, rr=rr)
            if diagnostics:
                ra, _ = official_ranks(a, ev)
                rb, _ = official_ranks(b, ev)
                winner = scores.argmax(1, keepdim=True)
                responsibility = branch_b_responsibility(a, b, model.mode, model.temperature)
                values.update(branch_a_rank=ra, branch_b_rank=rb, query_cosine=cosine,
                              positive_a=a[:, 0], positive_b=b[:, 0],
                              b_positive_responsibility=responsibility[:, 0],
                              b_top_candidate_responsibility=responsibility.gather(1, winner).squeeze(1),
                              b_positive_stronger=(b[:, 0] > a[:, 0]))
            for key, value in values.items():
                records.setdefault(key, []).append(value.cpu().numpy())
    model.train(was_training)
    out = {key: np.concatenate(value) for key, value in records.items()}
    return rank_summary(out["rank"]), out


def paired_interval(delta, quantiles, rng):
    n = len(delta) // 2
    paired = (delta[:n] + delta[n:]) / 2
    # Resample one replicate at a time, not an enormous replicate x N array.
    means = [rng.choice(paired, size=n, replace=True).mean() for _ in range(1000)]
    return np.quantile(means, quantiles).tolist()


def comparisons(query_records, valid, families):
    ranks = {name: q["rank"] for name, q in query_records.items()}
    rrs = {name: 1 / rank.astype(np.float64) for name, rank in ranks.items()}
    rng = np.random.default_rng(3502)
    rel = np.tile(valid["relation"], 2)
    family = np.array([families[int(x)] for x in rel])
    directions = np.repeat([0, 1], len(valid["head"]))
    output = {}
    for name in MODES:
        delta = rrs[name] - rrs["frozen"]
        interval = paired_interval(delta, [.00625, .99375], rng)
        row = dict(metrics=rank_summary(ranks[name]), delta_vs_frozen=float(delta.mean()),
                   adjusted_bootstrap_98_75=interval,
                   worth_confirming_vs_frozen=bool(delta.mean() >= .001 and interval[0] > 0),
                   recovered_beyond_top10_to_top10=int(((ranks["frozen"] > 10) & (ranks[name] <= 10)).sum()),
                   lost_top1=int(((ranks["frozen"] == 1) & (ranks[name] > 1)).sum()),
                   gained_top1=int(((ranks["frozen"] > 1) & (ranks[name] == 1)).sum()),
                   family_direction=[])
        if name != "single":
            ds = rrs[name] - rrs["single"]
            row.update(delta_vs_single=float(ds.mean()),
                       descriptive_95_vs_single=paired_interval(ds, [.025, .975], rng),
                       second_bank_gain_gate=bool(ds.mean() >= .001 and row["worth_confirming_vs_frozen"]))
        if name in ("or", "and"):
            dm = rrs[name] - rrs["mean"]
            row.update(delta_vs_mean=float(dm.mean()),
                       descriptive_95_vs_mean=paired_interval(dm, [.025, .975], rng),
                       nonlinear_gain_gate=bool(dm.mean() >= .001 and row["second_bank_gain_gate"]))
        for fam in np.unique(family):
            for direction in (0, 1):
                mask = (family == fam) & (directions == direction)
                row["family_direction"].append(dict(
                    family=fam, direction=direction, queries=int(mask.sum()),
                    frozen_mrr=float(rrs["frozen"][mask].mean()),
                    mrr=float(rrs[name][mask].mean()), delta=float(delta[mask].mean())))
        q = query_records[name]
        row["branches"] = None if name == "single" else dict(
            a=rank_summary(q["branch_a_rank"]), b=rank_summary(q["branch_b_rank"]),
            mean_query_cosine=float(q["query_cosine"].mean()),
            mean_b_positive_responsibility=float(q["b_positive_responsibility"].mean()),
            mean_b_top_candidate_responsibility=float(q["b_top_candidate_responsibility"].mean()),
            b_positive_stronger_fraction=float(q["b_positive_stronger"].mean()))
        output[name] = row
    return output


def smoke(device):
    """Synthetic, full training-matrix shape; no OGB data or checkpoint read."""
    torch.manual_seed(3500)
    entity = torch.randn(4096, 144, dtype=torch.complex64, device=device)
    entity = entity / entity.abs().square().sum(-1, keepdim=True).sqrt()
    operator = torch.linalg.qr(torch.randn(4, 36, 4, 4, dtype=torch.complex64, device=device))[0]
    source = torch.arange(2048, device=device)
    relation = torch.zeros(2048, dtype=torch.long, device=device)
    positive = (source + 1) % len(entity)
    negatives = torch.arange(4096, device=device)
    for mode in MODES:
        m = DualOperator(entity, operator, torch.tensor(2.3, device=device), mode)
        opt = torch.optim.Adam([m.H_b], lr=.0005)
        before = frozen_digest(m)
        started = time.monotonic()
        for _ in range(10):
            opt.zero_grad(set_to_none=True)
            logits = m.training_scores(source, relation, positive, negatives)
            loss = F.cross_entropy(logits, torch.zeros(len(source), dtype=torch.long, device=device))
            loss.backward()
            assert torch.isfinite(m.H_b.grad).all()
            opt.step()
        assert before == frozen_digest(m)
        print(f"synthetic {mode}: finite, frozen intact, 10 updates "
              f"{time.monotonic()-started:.2f}s, loss={loss.item():.4f}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model")
    p.add_argument("--data-root")
    p.add_argument("--out")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.smoke:
        smoke(args.device)
        return
    if not all((args.model, args.data_root, args.out)):
        p.error("--model, --data-root and --out required for real-data campaign")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).parents[1]
    dataset = Path(args.data_root) / "ogbl_biokg"
    ck_sha = sha256(args.model)
    if ck_sha != "b463a8bcc431ada38f4834e235677464f9cd3a86346f88aecd2fd681a01260ef":
        raise ValueError("H35 screen requires preregistered released seed-0 student")
    receipt = dict(args=vars(args), checkpoint_sha256=ck_sha, torch=torch.__version__,
                   numpy=np.__version__, device=(torch.cuda.get_device_name() if args.device == "cuda" else args.device),
                   source_hashes={name: sha256(repo / name) for name in (
                       "biokg/H35.md", "biokg/dual_operator.py", "biokg/train_dual_operator.py",
                       "biokg/train_biokg_comp.py", "resonate.py")},
                   data_hashes={name: sha256(dataset / name) for name in (
                       "split/random/train.pt", "split/random/valid.pt", "processed/data_processed",
                       "mapping/relidx2relname.csv.gz")},
                   steps=5000, batch=2048, negatives=4096, learning_rate=.0005,
                   seed=3500, test_loaded=False, variants=list(MODES))
    (out / "prerun.json").write_text(json.dumps(receipt, indent=2) + "\n")
    def log(event):
        print(json.dumps(event), flush=True)
        with (out / "progress.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")
    split, offset, n_ent, _, num_nodes = validation_only(args.data_root)
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    n_rel = 2 * (int(split["train"]["relation"].max()) + 1)
    if ck["offset"] != offset or ck["n_rel"] != n_rel or ck["model"]["E"].shape[0] != n_ent:
        raise ValueError("Checkpoint/data indexing mismatch")
    valid = split["valid"]
    indices = np.random.default_rng(3501).choice(len(valid["head"]), 1000, replace=False)
    np.save(out / "probe_indices.npy", indices)
    probe = {name: np.asarray(value)[indices] for name, value in valid.items()}
    records, runs = {}, {}
    reference = from_checkpoint(ck, "single", args.device).requires_grad_(False)
    metrics, records["frozen"] = evaluate(reference, valid, offset, n_rel)
    log(dict(stage="frozen_full_valid", **metrics))
    np.savez_compressed(out / "frozen_queries.npz", **records["frozen"])
    del reference
    for mode in MODES:
        model = from_checkpoint(ck, mode, args.device)
        frozen_before = frozen_digest(model)
        initial_b = tensor_digest({"H_b": model.H_b})
        if mode in ("or", "and") and initial_b != runs["mean"]["initial_b_sha256"]:
            raise AssertionError("Second-bank initializations differ")
        if model.n_params(trainable_only=True) != 117504:
            raise ValueError("Unexpected trainable parameter count")
        opt = torch.optim.Adam([model.H_b], lr=.0005)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000)
        stream = TrainStream(split["train"], offset, num_nodes)
        probe_stats, _ = evaluate(model, probe, offset, n_rel, diagnostics=False)
        last_mrr = probe_stats["mrr"]
        log(dict(arm=mode, step=0, probe_mrr=last_mrr, lr=.0005,
                 parameters=model.n_params(), trainable_parameters=model.n_params(True)))
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started, training_seconds = time.monotonic(), 0.
        for step in range(1, 5001):
            batch_started = time.monotonic()
            src, dst, rel, neg = stream.sample(2048, 4096)
            src = torch.as_tensor(src, device=args.device)
            dst = torch.as_tensor(dst, device=args.device)
            neg = torch.as_tensor(neg, device=args.device)
            rel = torch.full((len(src),), rel, device=args.device, dtype=torch.long)
            model.train()
            opt.zero_grad(set_to_none=True)
            logits = model.training_scores(src, rel, dst, neg)
            loss = F.cross_entropy(logits, torch.zeros(len(src), device=args.device, dtype=torch.long))
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss in {mode} step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([model.H_b], 1., error_if_nonfinite=True)
            opt.step()
            schedule.step()
            if args.device == "cuda":
                torch.cuda.synchronize()
            training_seconds += time.monotonic() - batch_started
            if step % 1000 == 0:
                probe_stats, _ = evaluate(model, probe, offset, n_rel, diagnostics=False)
                last_mrr = probe_stats["mrr"]
            if step % 250 == 0:
                log(dict(arm=mode, step=step, loss=float(loss), lr=opt.param_groups[0]["lr"],
                         last_probe_step=(step // 1000) * 1000, last_probe_mrr=last_mrr,
                         elapsed_seconds=time.monotonic() - started))
        if frozen_digest(model) != frozen_before:
            raise AssertionError("Frozen tensors changed")
        train_digest = stream.digest.hexdigest()
        if runs and train_digest != runs["single"]["train_stream_sha256"]:
            raise AssertionError("Training streams differ")
        metrics, records[mode] = evaluate(model, valid, offset, n_rel)
        runs[mode] = dict(metrics=metrics, training_seconds=training_seconds,
                          elapsed_seconds=time.monotonic() - started,
                          peak_allocated_bytes=(torch.cuda.max_memory_allocated() if args.device == "cuda" else None),
                          train_stream_sha256=train_digest, frozen_sha256=frozen_before,
                          frozen_unchanged=True, initial_b_sha256=initial_b,
                          parameters=model.n_params(), trainable_parameters=model.n_params(True))
        torch.save(dict(model_type="H35DualOperator", model=model.state_dict(), mode=mode,
                        temperature=1., args=vars(args), steps=5000, offset=offset, n_rel=n_rel,
                        original_checkpoint_sha256=ck_sha), out / f"{mode}.pt")
        np.savez_compressed(out / f"{mode}_queries.npz", **records[mode])
        (out / f"{mode}.json").write_text(json.dumps(runs[mode], indent=2) + "\n")
        log(dict(stage="full_valid", arm=mode, **runs[mode]))
        del model, opt, schedule, logits, loss
        if args.device == "cuda":
            torch.cuda.empty_cache()
    with gzip.open(dataset / "mapping/relidx2relname.csv.gz", "rt") as f:
        reader = csv.reader(f)
        next(reader)
        families = {int(i): name.split("_")[0] for i, name in reader}
    result = dict(screen_only=True, test_loaded=False, runs=runs,
                  frozen=rank_summary(records["frozen"]["rank"]),
                  comparisons=comparisons(records, valid, families))
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    log(dict(stage="complete", comparisons={mode: {k: v for k, v in info.items()
                    if k != "family_direction"} for mode, info in result["comparisons"].items()}))


if __name__ == "__main__":
    main()
