"""Fixed H35E three-arm campaign; protocol H35E.md. No TEST option."""

import argparse
import csv
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
from ogb.linkproppred import Evaluator

from biokg.joint_operator import JointOperator, retained_loss
from biokg.mixed_operator import MODES, WIDTH, MixedOperator, from_student, restore_mixed
from biokg.relation_analogy import sha256
from biokg.train_biokg_comp import globalize, load, load_teacher, teacher_logits
from biokg.train_dual_operator import TrainStream, official_ranks, paired_interval, rank_summary
from biokg.train_joint_operator import assert_independent, gradient_norms, parameter_hashes, teacher_hashes


STEPS = 5000


def validation_only(root):
    return load(root, include_test=False)


@torch.inference_mode()
def evaluate(model, part, offset, n_rel, diagnostics=False, chunk=128):
    was_training = model.training
    model.eval()
    h, r, t = globalize(part, offset)
    dev = model.E.device
    ev = Evaluator("ogbl-biokg")
    records = {}
    for direction in (0, 1):
        source, target = (h, t) if direction == 0 else (t, h)
        types = part["tail_type" if direction == 0 else "head_type"]
        negatives = part["tail_neg" if direction == 0 else "head_neg"]
        for start in range(0, len(h), chunk):
            sl = slice(start, start + chunk)
            off = np.array([offset[name] for name in types[sl]])
            cand = np.concatenate([target[sl, None], negatives[sl] + off[:, None]], axis=1)
            rel = torch.as_tensor(r[sl] + direction * (n_rel // 2), device=dev)
            score, a, b = model.candidate_outputs(torch.as_tensor(source[sl], device=dev), rel,
                                                  torch.as_tensor(cand, device=dev))
            rank, rr = official_ranks(score, ev)
            values = dict(rank=rank, rr=rr)
            if diagnostics and b is not None:
                ar, _ = official_ranks(a, ev)
                br, _ = official_ranks(b, ev)
                ac, bc = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
                corr = (ac * bc).sum(1) / (ac.square().sum(1) * bc.square().sum(1)).sqrt().clamp_min(1e-12)
                gate = model.gate_logit[rel].sigmoid()
                values.update(branch_a_rank=ar, branch_b_rank=br, gate=gate,
                              candidate_score_correlation=corr, a_score_std=a.std(1, correction=0),
                              b_correction_std=gate * b.std(1, correction=0))
            for name, value in values.items():
                records.setdefault(name, []).append(value.cpu().numpy())
    model.train(was_training)
    result = {name: np.concatenate(rows) for name, rows in records.items()}
    return rank_summary(result["rank"]), result


def lr_values(opt):
    return {group["group_name"]: group["lr"] for group in opt.param_groups}


def compare(treatment, control, rng, quantiles):
    delta = 1 / treatment.astype(np.float64) - 1 / control.astype(np.float64)
    return dict(delta_mrr=float(delta.mean()), bootstrap_interval=paired_interval(delta, quantiles, rng),
                recovered_top1=int(((treatment == 1) & (control > 1)).sum()),
                lost_top1=int(((treatment > 1) & (control == 1)).sum()),
                improved_queries=int((treatment < control).sum()),
                worsened_queries=int((treatment > control).sum()))


def summarize(records, frozen, valid, family_names):
    ranks = {mode: record["rank"] for mode, record in records.items()}
    ranks["frozen"] = frozen
    rng = np.random.default_rng(3514)
    comparisons = {"distance_vs_" + control: compare(ranks["distance"], ranks[control], rng,
                                                     [.05 / 6, 1 - .05 / 6])
                   for control in ("dot", "single", "frozen")}
    advance = all(row["delta_mrr"] >= .001 and row["bootstrap_interval"][0] > 0
                  for row in comparisons.values())
    descriptive = {"dot_vs_single": compare(ranks["dot"], ranks["single"], rng, [.025, .975]),
                   "single_vs_frozen": compare(ranks["single"], ranks["frozen"], rng, [.025, .975])}
    branches = {}
    for mode in ("dot", "distance"):
        q = records[mode]
        branches[mode] = dict(a=rank_summary(q["branch_a_rank"]), b=rank_summary(q["branch_b_rank"]),
                              b_recovers_a_top1=int(((q["branch_b_rank"] == 1) & (q["branch_a_rank"] > 1)).sum()),
                              b_loses_a_top1=int(((q["branch_b_rank"] > 1) & (q["branch_a_rank"] == 1)).sum()),
                              ab_top1_disagreement=float(((q["branch_a_rank"] == 1) != (q["branch_b_rank"] == 1)).mean()),
                              combined_vs_a=compare(q["rank"], q["branch_a_rank"], rng, [.025, .975]))
        for name in ("gate", "candidate_score_correlation", "a_score_std", "b_correction_std"):
            branches[mode][name] = dict(mean=float(q[name].mean()),
                                        quantiles_05_50_95=np.quantile(q[name], [.05, .5, .95]).tolist())
    family = np.array([family_names[int(x)] for x in np.tile(valid["relation"], 2)])
    direction = np.repeat([0, 1], len(valid["head"]))
    slices = []
    for fam in np.unique(family):
        for direct in (0, 1):
            mask = (family == fam) & (direction == direct)
            slices.append(dict(family=fam, direction=direct, queries=int(mask.sum()),
                               mrr={name: rank_summary(rank[mask])["mrr"] for name, rank in ranks.items()}))
    return dict(screen_only=True, test_loaded=False, metrics={name: rank_summary(rank) for name, rank in ranks.items()},
                primary_comparisons=comparisons, comparison_interval_percent=100 * (1 - .05 / 3),
                advance_gate=advance, descriptive_comparisons=descriptive,
                descriptive_branches=branches, descriptive_family_direction=slices)


def synthetic_smoke(device):
    """Full-size tables, teacher count, batch and negatives; no BioKG reads."""
    from resonate import ResonatE
    torch.manual_seed(3513)
    teachers = [ResonatE(93773, 102, k=12, block=True, block_size=4)
                .to(device).eval().requires_grad_(False) for _ in range(10)]
    base = teachers[0]
    students = {mode: MixedOperator(JointOperator(base.E, base.H, base.log_tau, "single"), mode)
                for mode in MODES}
    assert_independent(students, teachers)
    teacher_before = teacher_hashes(teachers)
    before = {mode: parameter_hashes(model) for mode, model in students.items()}
    assert before["dot"] == before["distance"]
    opts = {mode: torch.optim.Adam(model.optimizer_groups()) for mode, model in students.items()}
    source = torch.arange(2048, device=device)
    relation = torch.zeros(2048, dtype=torch.long, device=device)
    positive = (source + 1) % 93773
    negatives = torch.arange(4096, device=device)
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.monotonic()
    for step in range(20):
        target = teacher_logits(teachers, source, relation, positive, negatives)
        for mode, model in students.items():
            opts[mode].zero_grad(set_to_none=True)
            loss, _ = retained_loss(model.training_outputs(source, relation, positive, negatives), target)
            assert torch.isfinite(loss)
            loss.backward()
            if step == 0:
                norms = gradient_norms(model)
                assert all(value is not None and value > 0 and np.isfinite(value) for value in norms.values())
                print(json.dumps(dict(stage="smoke_gradients", arm=mode, norms=norms)), flush=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opts[mode].step()
        del target, loss
    if cuda:
        torch.cuda.synchronize()
    seconds = time.monotonic() - start
    assert teacher_before == teacher_hashes(teachers)
    assert all(p.grad is None and not p.requires_grad for t in teachers for p in t.parameters())
    for mode, model in students.items():
        changed = parameter_hashes(model)
        assert all(before[mode][name] != changed[name] for name in changed)
        with torch.inference_mode():
            candidates = torch.arange(501, device=device)[None, :].expand(128, -1)
            candidates = candidates.clone()
            candidates[:, 1] = candidates[:, 3]
            permutation = torch.randperm(501, device=device)
            original = model.candidate_outputs(source[:128], relation[:128], candidates)[0]
            permuted = model.candidate_outputs(source[:128], relation[:128], candidates[:, permutation])[0]
            torch.testing.assert_close(permuted, original[:, permutation], atol=0, rtol=0)
            torch.testing.assert_close(original[:, 1], original[:, 3], atol=0, rtol=0)
    print(json.dumps(dict(stage="smoke_complete", triplet_updates=20, seconds=seconds,
                         estimated_5000_training_seconds=seconds * 250,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
                         all_groups_updated=True, teachers_unchanged=True, candidate_permutation_exact=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="biokg/results/h35e/campaign_s0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.smoke:
        synthetic_smoke(args.device)
        return
    repo, out = Path(__file__).resolve().parents[1], Path(args.out)
    prior_dir = repo / "biokg/results/h35/campaign_s0"
    prior_path = repo / "biokg/results/h35c/campaign_s0/prerun.json"
    prior = json.loads(prior_path.read_text())
    student_path = repo / "biokg/checkpoints/dist_T2_s0.pt"
    data_root = Path(prior["args"]["data_root"])
    dataset = data_root / "ogbl_biokg"
    teacher_files = [Path(prior["args"]["teacher_dir"]) / f"sparse_s{i}.pt" for i in range(10)]
    inputs = {str(student_path): prior["student_sha256"],
              str(prior_dir / "frozen_queries.npz"): prior["reference_ranks_sha256"],
              prior["args"]["checksums"]: prior["checksums_sha256"]}
    inputs.update({str(path): prior["teacher_sha256"][path.name] for path in teacher_files})
    inputs.update({str(dataset / name): digest for name, digest in prior["data_hashes"].items()})
    assert all(sha256(path) == digest for path, digest in inputs.items()), "Input hash mismatch"
    for previous in (prior, json.loads((repo / "biokg/results/h35d/campaign_s0/prerun.json").read_text())):
        assert all(sha256(repo / name) == digest for name, digest in previous["source_hashes"].items())
    source_files = ["biokg/H35E.md", "biokg/mixed_operator.py", "biokg/train_mixed_operator.py",
                    "biokg/test_mixed_operator.py", "biokg/audit_mixed_operator.py", "biokg/relation_analogy.py",
                    *prior["source_hashes"]]
    source_hashes = {name: sha256(repo / name) for name in source_files}
    cuda = torch.device(args.device).type == "cuda"
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(protocol="H35E", args=vars(args), source_hashes=source_hashes, input_sha256=inputs,
                   prior_receipt_sha256=sha256(prior_path), student_sha256=prior["student_sha256"],
                   teacher_sha256=prior["teacher_sha256"], reference_ranks_sha256=prior["reference_ranks_sha256"],
                   reference_directory=str(prior_dir), steps=STEPS, stream_seed=3510, batch=2048, negatives=4096,
                   width=WIDTH, lr_a=.0001, lr_b=.001, initial_gate=.05,
                   kd_temperature=2., kd_weight=1., trajectory_weight=.1, test_loaded=False,
                   torch=torch.__version__, numpy=np.__version__,
                   device=torch.cuda.get_device_name() if cuda else args.device,
                   started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (out / "prerun.json").write_text(json.dumps(receipt, indent=2) + "\n")
    def log(event):
        print(json.dumps(event), flush=True)
        with (out / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
    log(dict(stage="inputs_verified", steps=STEPS, arms=list(MODES)))
    split, offset, n_ent, _, num_nodes = validation_only(str(data_root))
    n_rel = 2 * (int(split["train"]["relation"].max()) + 1)
    assert (n_ent, n_rel, len(split["valid"]["head"])) == (93773, 102, 162886)
    assert set(split) == {"train", "valid"}
    ck = torch.load(student_path, map_location="cpu", weights_only=False)
    assert ck["offset"] == offset and ck["n_rel"] == n_rel
    teachers = []
    for i, path in enumerate(teacher_files):
        tc = torch.load(path, map_location="cpu", weights_only=False)
        assert tc["offset"] == offset and tc["n_rel"] == n_rel
        assert tc["args"]["seed"] == i and not tc["args"].get("distill")
        del tc
        teachers.append(load_teacher(path, n_ent, n_rel, torch.device(args.device)))
    teacher_before = teacher_hashes(teachers)
    students = {mode: from_student(ck, mode, args.device) for mode in MODES}
    del ck
    assert_independent(students, teachers)
    initial = {mode: parameter_hashes(model) for mode, model in students.items()}
    assert initial["dot"] == initial["distance"], "Dot/distance initial tensors differ"
    assert {mode: model.n_params() for mode, model in students.items()} == dict(single=27124129, dot=30134760, distance=30134760)
    opts = {mode: torch.optim.Adam(model.optimizer_groups()) for mode, model in students.items()}
    schedules = {mode: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS) for mode, opt in opts.items()}
    valid = split["valid"]
    ids = np.random.default_rng(3501).choice(len(valid["head"]), 1000, replace=False)
    np.testing.assert_array_equal(ids, np.load(prior_dir / "probe_indices.npy", allow_pickle=False))
    np.save(out / "probe_indices.npy", ids)
    probe = {name: np.asarray(value)[ids] for name, value in valid.items()}
    probe_mrr = {}
    for mode, model in students.items():
        metrics, _ = evaluate(model, probe, offset, n_rel)
        probe_mrr[mode] = metrics["mrr"]
        log(dict(stage="initial_probe", arm=mode, step=0, lr=lr_values(opts[mode]), **metrics))
    assert abs(probe_mrr["single"] - .8411956954420665) < 1e-12
    stream = TrainStream(split["train"], offset, num_nodes, seed=3510)
    started = last_log = time.monotonic()
    times = dict(teacher=0., **{mode: 0. for mode in MODES})
    updates = {mode: 0 for mode in MODES}
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, STEPS + 1):
        source, positive, relation, negatives = stream.sample(2048, 4096)
        source, positive, negatives = [torch.as_tensor(x, device=args.device) for x in (source, positive, negatives)]
        relation = torch.full((len(source),), relation, device=args.device, dtype=torch.long)
        tick = time.monotonic()
        target = teacher_logits(teachers, source, relation, positive, negatives)
        if cuda:
            torch.cuda.synchronize()
        times["teacher"] += time.monotonic() - tick
        losses = {}
        should_log = step % 100 == 0 or time.monotonic() - last_log >= 40
        for mode, model in students.items():
            tick = time.monotonic()
            model.train()
            opts[mode].zero_grad(set_to_none=True)
            outputs = model.training_outputs(source, relation, positive, negatives)
            loss, parts = retained_loss(outputs, target)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss {mode} at step {step}")
            loss.backward()
            if step == 1:
                norms = gradient_norms(model)
                assert all(x is not None and x > 0 and np.isfinite(x) for x in norms.values())
                log(dict(stage="first_gradients", arm=mode, norms=norms))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opts[mode].step()
            schedules[mode].step()
            if cuda:
                torch.cuda.synchronize()
            times[mode] += time.monotonic() - tick
            updates[mode] += 1
            if should_log:
                losses[mode] = dict(total=float(loss), **{name: float(value) for name, value in parts.items()})
            del outputs, loss, parts
        del target
        if step % 500 == 0:
            for mode, model in students.items():
                metrics, _ = evaluate(model, probe, offset, n_rel)
                probe_mrr[mode] = metrics["mrr"]
        if should_log:
            log(dict(stage="training", step=step, updates=updates, losses=losses,
                     lr={mode: lr_values(opt) for mode, opt in opts.items()},
                     last_probe_step=(step // 500) * 500, probe_mrr=probe_mrr,
                     elapsed_seconds=time.monotonic() - started))
            last_log = time.monotonic()
    assert updates == {mode: STEPS for mode in MODES}
    teacher_after = teacher_hashes(teachers)
    assert teacher_before == teacher_after
    assert all(p.grad is None and not p.requires_grad for t in teachers for p in t.parameters())
    student_changes, checkpoint_hashes = {}, {}
    for mode, model in students.items():
        final = parameter_hashes(model)
        changed = {name: final[name] != initial[mode][name] for name in final}
        assert all(changed.values())
        student_changes[mode] = dict(initial_hashes=initial[mode], final_hashes=final, changed=changed)
        torch.save(dict(model_type="H35EMixedOperator", model=model.state_dict(), mode=mode, width=WIDTH,
                        steps=STEPS, offset=offset, n_rel=n_rel, source_student_sha256=prior["student_sha256"],
                        teacher_sha256=prior["teacher_sha256"]), out / f"{mode}.pt")
        checkpoint_hashes[mode] = sha256(out / f"{mode}.pt")
        restored = restore_mixed(torch.load(out / f"{mode}.pt", map_location="cpu", weights_only=False))
        assert parameter_hashes(restored) == final
        del restored
    assert stream.digest.hexdigest() == "8821d02f21c2b0b50154e368ab9f82271224ab1d699fae9606e4f972ce9cbe44"
    training_audit = dict(updates=updates, teacher_before=teacher_before, teacher_after=teacher_after,
                         teachers_unchanged=True, teachers_without_grad=True, student_changes=student_changes,
                         checkpoint_sha256=checkpoint_hashes, real_parameter_counts={m: x.n_params() for m, x in students.items()},
                         train_stream_sha256=stream.digest.hexdigest(), component_seconds=times,
                         training_and_probes_seconds=time.monotonic() - started,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None)
    (out / "training_audit.json").write_text(json.dumps(training_audit, indent=2) + "\n")
    log(dict(stage="training_complete", updates=updates, elapsed_seconds=time.monotonic() - started))
    del teachers, opts, schedules, model
    if cuda:
        torch.cuda.empty_cache()
    records = {}
    for mode, model in students.items():
        tick = time.monotonic()
        before_eval = parameter_hashes(model)
        metrics, record = evaluate(model, valid, offset, n_rel, diagnostics=True)
        assert parameter_hashes(model) == before_eval
        records[mode] = record
        np.savez_compressed(out / f"{mode}_queries.npz", **record)
        log(dict(stage="full_valid", arm=mode, seconds=time.monotonic() - tick, **metrics))
    with np.load(prior_dir / "frozen_queries.npz", allow_pickle=False) as ref:
        frozen = ref["rank"]
    with gzip.open(dataset / "mapping/relidx2relname.csv.gz", "rt") as file:
        rows = csv.reader(file)
        next(rows)
        family_names = {int(i): name.split("_")[0] for i, name in rows}
    summary = summarize(records, frozen, valid, family_names)
    summary["training_audit"] = training_audit
    summary["query_sha256"] = {mode: sha256(out / f"{mode}_queries.npz") for mode in MODES}
    assert all(sha256(path) == digest for path, digest in inputs.items())
    assert all(sha256(repo / name) == digest for name, digest in source_hashes.items())
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(dict(stage="complete", metrics=summary["metrics"], primary=summary["primary_comparisons"],
             advance_gate=summary["advance_gate"], source_and_input_hashes_unchanged=True))


if __name__ == "__main__":
    main()
