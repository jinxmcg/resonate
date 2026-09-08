"""Preregistered H35B two-arm screen; TRAIN-only adaptation, VALID evaluation."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from biokg.relation_analogy import sha256
from biokg.sequential_operator import from_student
from biokg.train_dual_operator import (
    TrainStream, evaluate, frozen_digest, paired_interval, rank_summary, validation_only,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--h35", required=True, help="Completed H35 campaign directory")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    prior = Path(args.h35)
    prior_result = json.loads((prior / "summary.json").read_text())
    prior_receipt = json.loads((prior / "prerun.json").read_text())
    expected_sha = "b463a8bcc431ada38f4834e235677464f9cd3a86346f88aecd2fd681a01260ef"
    if sha256(args.model) != expected_sha or prior_receipt["checkpoint_sha256"] != expected_sha:
        raise ValueError("Wrong source checkpoint")
    dataset = Path(args.data_root) / "ogbl_biokg"
    for name, digest in prior_receipt["data_hashes"].items():
        if sha256(dataset / name) != digest:
            raise ValueError("Dataset changed since H35")
    repo = Path(__file__).parents[1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(args=vars(args), source_checkpoint_sha256=expected_sha,
                   source_hashes={name: sha256(repo / name) for name in (
                       "biokg/H35B.md", "biokg/sequential_operator.py",
                       "biokg/train_sequential_operator.py", "biokg/dual_operator.py",
                       "biokg/train_dual_operator.py", "biokg/train_biokg_comp.py", "resonate.py")},
                   data_hashes=prior_receipt["data_hashes"], prior_summary_sha256=sha256(prior / "summary.json"),
                   prior_reference_ranks_sha256=sha256(prior / "frozen_queries.npz"),
                   steps=5000, seed=3500, batch=2048, negatives=4096, lr=.0005,
                   device=(torch.cuda.get_device_name() if args.device == "cuda" else args.device),
                   torch=torch.__version__, numpy=np.__version__, test_loaded=False)
    (out / "prerun.json").write_text(json.dumps(receipt, indent=2) + "\n")
    def log(event):
        print(json.dumps(event), flush=True)
        with (out / "progress.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")
    split, offset, n_ent, _, num_nodes = validation_only(args.data_root)
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    n_rel = 2 * (int(split["train"]["relation"].max()) + 1)
    if ck["offset"] != offset or ck["n_rel"] != n_rel or ck["model"]["E"].shape[0] != n_ent:
        raise ValueError("Checkpoint indexing mismatch")
    valid = split["valid"]
    ids = np.random.default_rng(3501).choice(len(valid["head"]), 1000, replace=False)
    np.testing.assert_array_equal(ids, np.load(prior / "probe_indices.npy"))
    probe = {name: np.asarray(value)[ids] for name, value in valid.items()}
    prior_events = [json.loads(line) for line in (prior / "progress.jsonl").read_text().splitlines()]
    frozen_probe = next(event["probe_mrr"] for event in prior_events
                        if event.get("arm") == "single" and event.get("step") == 0)
    reference_rank = np.load(prior / "frozen_queries.npz")["rank"]
    all_ranks, runs = {}, {}
    for mixing in ("local", "shuffle"):
        model = from_student(ck, mixing, args.device)
        frozen_before = frozen_digest(model)
        assert model.n_params() == 27241633 and model.n_params(True) == 117504
        initial_probe, _ = evaluate(model, probe, offset, n_rel, diagnostics=False)
        if abs(initial_probe["mrr"] - frozen_probe) > 1e-6:
            raise AssertionError("Identity initialization does not reproduce the reference probe")
        log(dict(arm=mixing, step=0, probe_mrr=initial_probe["mrr"], lr=.0005))
        opt = torch.optim.Adam([model.H_b], lr=.0005)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000)
        stream = TrainStream(split["train"], offset, num_nodes)
        last_probe = initial_probe["mrr"]
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started, training_seconds = time.monotonic(), 0.
        for step in range(1, 5001):
            tick = time.monotonic()
            src, dst, rel, neg = stream.sample(2048, 4096)
            src, dst, neg = [torch.as_tensor(x, device=args.device) for x in (src, dst, neg)]
            rel = torch.full((len(src),), rel, dtype=torch.long, device=args.device)
            opt.zero_grad(set_to_none=True)
            scores = model.training_scores(src, rel, dst, neg)
            loss = F.cross_entropy(scores, torch.zeros(len(src), dtype=torch.long, device=args.device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([model.H_b], 1., error_if_nonfinite=True)
            opt.step()
            schedule.step()
            if args.device == "cuda":
                torch.cuda.synchronize()
            training_seconds += time.monotonic() - tick
            if step % 1000 == 0:
                metrics, _ = evaluate(model, probe, offset, n_rel, diagnostics=False)
                last_probe = metrics["mrr"]
            if step % 250 == 0:
                log(dict(arm=mixing, step=step, lr=opt.param_groups[0]["lr"], loss=float(loss),
                         last_probe_step=(step // 1000) * 1000, last_probe_mrr=last_probe,
                         elapsed_seconds=time.monotonic() - started))
        assert frozen_before == frozen_digest(model), "Frozen parameters changed"
        stream_sha = stream.digest.hexdigest()
        assert stream_sha == prior_result["runs"]["single"]["train_stream_sha256"]
        metrics, queries = evaluate(model, valid, offset, n_rel)
        all_ranks[mixing] = queries["rank"]
        runs[mixing] = dict(metrics=metrics, training_seconds=training_seconds,
                            elapsed_seconds=time.monotonic() - started,
                            train_stream_sha256=stream_sha, frozen_sha256=frozen_before,
                            frozen_unchanged=True, initial_probe_mrr=initial_probe["mrr"],
                            mean_cosine_vs_original_query=float(queries["query_cosine"].mean()),
                            peak_allocated_bytes=(torch.cuda.max_memory_allocated() if args.device == "cuda" else None),
                            parameters=model.n_params(), trainable_parameters=model.n_params(True))
        torch.save(dict(model_type="H35BSequentialOperator", model=model.state_dict(),
                        mixing=mixing, args=vars(args), steps=5000, offset=offset, n_rel=n_rel,
                        source_checkpoint_sha256=expected_sha), out / f"{mixing}.pt")
        np.savez_compressed(out / f"{mixing}_queries.npz", **queries)
        (out / f"{mixing}.json").write_text(json.dumps(runs[mixing], indent=2) + "\n")
        log(dict(stage="full_valid", arm=mixing, **runs[mixing]))
        del model, opt, schedule, scores, loss
        if args.device == "cuda":
            torch.cuda.empty_cache()
    rr_ref = 1 / reference_rank.astype(np.float64)
    rr_local = 1 / all_ranks["local"].astype(np.float64)
    rr_shuffle = 1 / all_ranks["shuffle"].astype(np.float64)
    rng = np.random.default_rng(3552)
    ds_ref, ds_local = rr_shuffle - rr_ref, rr_shuffle - rr_local
    ci_ref = paired_interval(ds_ref, [.0125, .9875], rng)
    ci_local = paired_interval(ds_local, [.0125, .9875], rng)
    summary = dict(screen_only=True, test_loaded=False, adaptive_followup=True, runs=runs,
                   reference=rank_summary(reference_rank),
                   local_delta_vs_frozen=float((rr_local - rr_ref).mean()),
                   shuffle_delta_vs_frozen=float(ds_ref.mean()),
                   shuffle_delta_vs_local=float(ds_local.mean()),
                   shuffle_vs_frozen_adjusted_97_5=ci_ref,
                   shuffle_vs_local_adjusted_97_5=ci_local,
                   advance_gate=bool(ds_ref.mean() >= .001 and ds_local.mean() >= .001
                                     and ci_ref[0] > 0 and ci_local[0] > 0))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(dict(stage="complete", **summary))


if __name__ == "__main__":
    main()
