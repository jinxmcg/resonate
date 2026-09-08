"""H35F fixed 10k B warm-up + 5k joint campaign; no TEST option."""

import argparse
import csv
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch

from biokg.branch_warmup import (
    ARMS, STEPS, WARM_STEPS, assert_gradients, configure_phase, effective_lr,
    phase_change_audit, training_loss,
)
from biokg.joint_operator import JointOperator
from biokg.mixed_operator import WIDTH, MixedOperator, from_student, restore_mixed
from biokg.relation_analogy import sha256
from biokg.train_biokg_comp import load, load_teacher, teacher_logits
from biokg.train_dual_operator import TrainStream, rank_summary
from biokg.train_joint_operator import assert_independent, parameter_hashes, teacher_hashes
from biokg.train_mixed_operator import compare, evaluate, lr_values


def validation_only(root):
    return load(root, include_test=False)


def optimizer_steps(model, optimizer):
    return {name: int(optimizer.state.get(p, {}).get("step", 0)) for name, p in model.named_parameters()}


def probe_statistics(model, part, offset, n_rel):
    metrics, query = evaluate(model, part, offset, n_rel, diagnostics=True)
    result = dict(combined_mrr=metrics["mrr"], queries=metrics["queries"])
    if model.mode == "single":
        result["a_mrr"] = metrics["mrr"]
    else:
        result.update(a_mrr=rank_summary(query["branch_a_rank"])["mrr"],
                      b_mrr=rank_summary(query["branch_b_rank"])["mrr"],
                      mean_gate=float(query["gate"].mean()), private_scale=float(model.log_tau_b.exp()))
    return result


def make_checkpoint(model, arm, step, offset, n_rel, prior):
    # Same H35E architecture and standalone loader; training protocol is H35F.
    return dict(model_type="H35EMixedOperator", protocol="H35F", model=model.state_dict(),
                arm=arm, mode=model.mode, width=WIDTH, steps=step, offset=offset, n_rel=n_rel,
                source_student_sha256=prior["student_sha256"], teacher_sha256=prior["teacher_sha256"])


def summarize(records, frozen, valid, families):
    ranks = {name: record["rank"] for name, record in records.items()}
    ranks["frozen"] = frozen
    rng = np.random.default_rng(3516)
    primary = {"warmup_vs_" + control: compare(ranks["warmup"], ranks[control], rng, [.05 / 6, 1 - .05 / 6])
               for control in ("joint", "single", "frozen")}
    branches, cohorts = {}, {}
    for arm in ARMS:
        r = ranks[arm]
        missed, correct = frozen > 1, frozen == 1
        cohorts[arm] = dict(original_errors_fixed=int((missed & (r == 1)).sum()),
                            original_correct_broken=int((correct & (r > 1)).sum()),
                            mrr_on_original_missed=rank_summary(r[missed])["mrr"],
                            mrr_on_original_correct=rank_summary(r[correct])["mrr"])
        if arm != "single":
            q = records[arm]
            ratio = q["b_correction_std"] / np.maximum(q["a_score_std"], 1e-12)
            branches[arm] = dict(a=rank_summary(q["branch_a_rank"]), b=rank_summary(q["branch_b_rank"]),
                                  combined_vs_a=compare(q["rank"], q["branch_a_rank"], rng, [.025, .975]),
                                  b_fixes_a_top1=int(((q["branch_b_rank"] == 1) & (q["branch_a_rank"] > 1)).sum()))
            for name, arr in {**{key: q[key] for key in ("gate", "candidate_score_correlation", "a_score_std", "b_correction_std")},
                              "correction_to_a_std_ratio": ratio}.items():
                branches[arm][name] = dict(mean=float(arr.mean()), quantiles_05_50_95=np.quantile(arr, [.05, .5, .95]).tolist())
    family = np.array([families[int(r)] for r in np.tile(valid["relation"], 2)])
    direction = np.repeat([0, 1], len(valid["head"]))
    slices = []
    for fam in np.unique(family):
        for direct in (0, 1):
            mask = (family == fam) & (direction == direct)
            slices.append(dict(family=fam, direction=direct, queries=int(mask.sum()),
                               mrr={name: rank_summary(rank[mask])["mrr"] for name, rank in ranks.items()}))
    return dict(protocol="H35F", screen_only=True, test_loaded=False,
                metrics={name: rank_summary(rank) for name, rank in ranks.items()},
                primary_comparisons=primary, interval_percent=100 * (1 - .05 / 3),
                advance_gate=all(row["delta_mrr"] >= .001 and row["bootstrap_interval"][0] > 0 for row in primary.values()),
                descriptive_branches=branches, original_error_cohorts=cohorts,
                original_missed_top1=int((frozen > 1).sum()), descriptive_family_direction=slices)


def synthetic_smoke(device):
    from resonate import ResonatE
    torch.manual_seed(3515)
    teachers = [ResonatE(93773, 102, k=12, block=True, block_size=4).to(device)
                .eval().requires_grad_(False) for _ in range(10)]
    base = teachers[0]
    students = {arm: MixedOperator(JointOperator(base.E, base.H, base.log_tau, "single"),
                                  "single" if arm == "single" else "dot") for arm in ARMS}
    assert_independent(students, teachers)
    teacher_before = teacher_hashes(teachers)
    before = {arm: parameter_hashes(model) for arm, model in students.items()}
    assert before["joint"] == before["warmup"]
    configure_phase(students["warmup"], True)
    opts = {arm: torch.optim.Adam(model.optimizer_groups()) for arm, model in students.items()}
    schedules = {arm: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS) for arm, opt in opts.items()}
    batch = (torch.arange(2048, device=device), torch.zeros(2048, dtype=torch.long, device=device),
             (torch.arange(2048, device=device) + 1) % 93773, torch.arange(4096, device=device))
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    timings = {}
    for phase in ("b_warmup", "joint"):
        if phase == "joint":
            phase_change_audit(before["warmup"], parameter_hashes(students["warmup"]), True)
            states = optimizer_steps(students["warmup"], opts["warmup"])
            assert all(step == (0 if name.startswith("a.") or name == "gate_logit" else 20) for name, step in states.items())
            configure_phase(students["warmup"], False)
            assert states == optimizer_steps(students["warmup"], opts["warmup"])
        if cuda:
            torch.cuda.synchronize()
        tick = time.monotonic()
        for step in range(20):
            target = teacher_logits(teachers, *batch)
            for arm, model in students.items():
                warm = arm == "warmup" and phase == "b_warmup"
                opts[arm].zero_grad(set_to_none=True)
                loss, _ = training_loss(model, batch, target, warm)
                assert torch.isfinite(loss)
                loss.backward()
                if step == 0:
                    print(json.dumps(dict(stage="smoke_gradients", phase=phase, arm=arm,
                                          norms=assert_gradients(model, warm))), flush=True)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                opts[arm].step()
                schedules[arm].step()
            del target, loss
        if cuda:
            torch.cuda.synchronize()
        timings[phase] = time.monotonic() - tick
    for arm, model in students.items():
        phase_change_audit(before[arm], parameter_hashes(model), False)
        with torch.inference_mode():
            candidates = torch.arange(501, device=device)[None].expand(128, -1)
            order = torch.randperm(501, device=device)
            a = model.candidate_outputs(batch[0][:128], batch[1][:128], candidates)[0]
            b = model.candidate_outputs(batch[0][:128], batch[1][:128], candidates[:, order])[0]
            torch.testing.assert_close(b, a[:, order], atol=0, rtol=0)
    assert teacher_before == teacher_hashes(teachers)
    assert all(not p.requires_grad and p.grad is None for teacher in teachers for p in teacher.parameters())
    print(json.dumps(dict(stage="smoke_complete", updates_per_phase=20, phase_seconds=timings,
                         estimated_training_seconds=timings["b_warmup"] * 500 + timings["joint"] * 250,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
                         phase_freezing_and_switch_verified=True, optimizer_state_retained=True,
                         teachers_unchanged=True, candidate_permutation_exact=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="biokg/results/h35f/campaign_s0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.smoke:
        synthetic_smoke(args.device)
        return
    repo, out = Path(__file__).resolve().parents[1], Path(args.out)
    prior_path = repo / "biokg/results/h35e/campaign_s0/prerun.json"
    prior = json.loads(prior_path.read_text())
    origin = json.loads((repo / "biokg/results/h35c/campaign_s0/prerun.json").read_text())
    inputs = prior["input_sha256"]
    assert all(sha256(path) == digest for path, digest in inputs.items())
    assert all(sha256(repo / name) == digest for name, digest in prior["source_hashes"].items())
    source_files = ["biokg/H35F.md", "biokg/branch_warmup.py", "biokg/train_branch_warmup.py",
                    "biokg/test_branch_warmup.py", "biokg/audit_branch_warmup.py", *prior["source_hashes"]]
    sources = {name: sha256(repo / name) for name in source_files}
    cuda = torch.device(args.device).type == "cuda"
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(protocol="H35F", args=vars(args), source_hashes=sources, input_sha256=inputs,
                   prior_receipt_sha256=sha256(prior_path), student_sha256=prior["student_sha256"],
                   teacher_sha256=prior["teacher_sha256"], reference_directory=prior["reference_directory"],
                   reference_ranks_sha256=prior["reference_ranks_sha256"], steps=STEPS, warm_steps=WARM_STEPS,
                   stream_seed=3510, width=WIDTH, batch=2048, negatives=4096, lr_a=.0001, lr_b=.001,
                   scheduler="single cosine over 15000, no reset", test_loaded=False,
                   torch=torch.__version__, numpy=np.__version__,
                   device=torch.cuda.get_device_name() if cuda else args.device,
                   started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (out / "prerun.json").write_text(json.dumps(receipt, indent=2) + "\n")
    def log(event):
        print(json.dumps(event), flush=True)
        with (out / "progress.jsonl").open("a") as file:
            file.write(json.dumps(event) + "\n")
    log(dict(stage="inputs_verified", arms=list(ARMS), steps=STEPS))
    root = Path(origin["args"]["data_root"])
    split, offset, n_ent, _, num_nodes = validation_only(str(root))
    n_rel = 2 * (int(split["train"]["relation"].max()) + 1)
    assert (n_ent, n_rel, len(split["valid"]["head"])) == (93773, 102, 162886)
    assert set(split) == {"train", "valid"}
    ck = torch.load(repo / "biokg/checkpoints/dist_T2_s0.pt", map_location="cpu", weights_only=False)
    assert ck["offset"] == offset and ck["n_rel"] == n_rel
    teachers = []
    for i in range(10):
        path = Path(origin["args"]["teacher_dir"]) / f"sparse_s{i}.pt"
        tc = torch.load(path, map_location="cpu", weights_only=False)
        assert tc["offset"] == offset and tc["n_rel"] == n_rel and tc["args"]["seed"] == i
        assert not tc["args"].get("distill")
        del tc
        teachers.append(load_teacher(path, n_ent, n_rel, torch.device(args.device)))
    teacher_before = teacher_hashes(teachers)
    students = {arm: from_student(ck, "single" if arm == "single" else "dot", args.device) for arm in ARMS}
    del ck
    assert_independent(students, teachers)
    initial = {arm: parameter_hashes(model) for arm, model in students.items()}
    assert initial["joint"] == initial["warmup"]
    configure_phase(students["warmup"], True)
    opts = {arm: torch.optim.Adam(model.optimizer_groups()) for arm, model in students.items()}
    schedules = {arm: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS) for arm, opt in opts.items()}
    prior_dir = Path(prior["reference_directory"])
    valid = split["valid"]
    ids = np.random.default_rng(3501).choice(len(valid["head"]), 1000, replace=False)
    np.testing.assert_array_equal(ids, np.load(prior_dir / "probe_indices.npy", allow_pickle=False))
    np.save(out / "probe_indices.npy", ids)
    probe = {name: np.asarray(value)[ids] for name, value in valid.items()}
    probe_scores = {arm: probe_statistics(model, probe, offset, n_rel) for arm, model in students.items()}
    assert abs(probe_scores["single"]["combined_mrr"] - .8411956954420665) < 1e-12
    assert probe_scores["joint"] == probe_scores["warmup"]
    log(dict(stage="initial_probe", step=0, probe=probe_scores))
    stream = TrainStream(split["train"], offset, num_nodes, seed=3510)
    stream_hashes, updates = {}, {arm: 0 for arm in ARMS}
    component_seconds = {name: 0. for name in ("teacher", *ARMS)}
    started = last_log = time.monotonic()
    boundary = None
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, STEPS + 1):
        if step == WARM_STEPS + 1:
            steps_before_switch = optimizer_steps(students["warmup"], opts["warmup"])
            configure_phase(students["warmup"], False)
            assert steps_before_switch == optimizer_steps(students["warmup"], opts["warmup"])
            log(dict(stage="phase_switch", step=step, optimizer_state_reset=False,
                     configured_lr=lr_values(opts["warmup"])))
        source, positive, relation, negatives = stream.sample(2048, 4096)
        source, positive, negatives = [torch.as_tensor(x, device=args.device) for x in (source, positive, negatives)]
        relation = torch.full((len(source),), relation, dtype=torch.long, device=args.device)
        batch = source, relation, positive, negatives
        tick = time.monotonic()
        target = teacher_logits(teachers, *batch)
        if cuda:
            torch.cuda.synchronize()
        component_seconds["teacher"] += time.monotonic() - tick
        should_log = step % 100 == 0 or time.monotonic() - last_log >= 40
        losses = {}
        for arm, model in students.items():
            warm_only = arm == "warmup" and step <= WARM_STEPS
            tick = time.monotonic()
            model.train()
            opts[arm].zero_grad(set_to_none=True)
            loss, parts = training_loss(model, batch, target, warm_only)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite {arm} loss at {step}")
            loss.backward()
            if step in (1, WARM_STEPS + 1):
                log(dict(stage="first_phase_gradients", arm=arm, step=step,
                         warm_only=warm_only, norms=assert_gradients(model, warm_only)))
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            opts[arm].step()
            schedules[arm].step()
            if cuda:
                torch.cuda.synchronize()
            component_seconds[arm] += time.monotonic() - tick
            updates[arm] += 1
            if should_log:
                losses[arm] = dict(total=float(loss), **{name: float(value) for name, value in parts.items()})
            del loss, parts
        del target
        if step % 500 == 0:
            probe_scores = {arm: probe_statistics(model, probe, offset, n_rel) for arm, model in students.items()}
        if step in (5000, WARM_STEPS, STEPS):
            stream_hashes[str(step)] = stream.digest.hexdigest()
            if step == 5000:
                assert stream_hashes[str(step)] == "8821d02f21c2b0b50154e368ab9f82271224ab1d699fae9606e4f972ce9cbe44"
        if step == WARM_STEPS:
            warm_model = students["warmup"]
            changes = phase_change_audit(initial["warmup"], parameter_hashes(warm_model), True)
            adam_steps = optimizer_steps(warm_model, opts["warmup"])
            assert all(value == (0 if name.startswith("a.") or name == "gate_logit" else WARM_STEPS)
                       for name, value in adam_steps.items())
            assert_gradients(warm_model, True)
            torch.save(make_checkpoint(warm_model, "warmup", step, offset, n_rel, prior), out / "warmup_boundary.pt")
            boundary = dict(step=step, parameter_changes=changes, optimizer_steps=adam_steps,
                            a_and_gate_unchanged=True, checkpoint_sha256=sha256(out / "warmup_boundary.pt"),
                            probe=probe_scores["warmup"], train_stream_sha256=stream.digest.hexdigest())
            (out / "boundary_audit.json").write_text(json.dumps(boundary, indent=2) + "\n")
            log(dict(stage="boundary_audit", step=step, a_and_gate_unchanged=True, b_groups_changed=True,
                     probe=probe_scores["warmup"]))
        if should_log:
            log(dict(stage="training", step=step, phase="b_warmup" if step <= WARM_STEPS else "joint",
                     updates=updates, losses=losses,
                     effective_lr={arm: effective_lr(opt, arm == "warmup" and step <= WARM_STEPS, arm != "single")
                                   for arm, opt in opts.items()},
                     configured_lr={arm: lr_values(opt) for arm, opt in opts.items()},
                     last_probe_step=(step // 500) * 500, probe=probe_scores,
                     elapsed_seconds=time.monotonic() - started))
            last_log = time.monotonic()
    assert updates == {arm: STEPS for arm in ARMS} and boundary is not None
    teacher_after = teacher_hashes(teachers)
    assert teacher_before == teacher_after
    assert all(p.grad is None and not p.requires_grad for t in teachers for p in t.parameters())
    changes, steps_by_parameter, checkpoints = {}, {}, {}
    for arm, model in students.items():
        final = parameter_hashes(model)
        changes[arm] = phase_change_audit(initial[arm], final, False)
        steps_by_parameter[arm] = optimizer_steps(model, opts[arm])
        for name, count in steps_by_parameter[arm].items():
            expected = 5000 if arm == "warmup" and (name.startswith("a.") or name == "gate_logit") else STEPS
            assert count == expected, (arm, name, count)
        torch.save(make_checkpoint(model, arm, STEPS, offset, n_rel, prior), out / f"{arm}.pt")
        checkpoints[arm] = sha256(out / f"{arm}.pt")
        restored = restore_mixed(torch.load(out / f"{arm}.pt", map_location="cpu", weights_only=False))
        assert parameter_hashes(restored) == final
        del restored
    train_audit = dict(teacher_before=teacher_before, teacher_after=teacher_after, teachers_unchanged=True,
                       teachers_without_grad=True, updates=updates, parameter_optimizer_steps=steps_by_parameter,
                       student_changes=changes, checkpoint_sha256=checkpoints, stream_sha256=stream_hashes,
                       real_parameter_counts={arm: model.n_params() for arm, model in students.items()},
                       boundary_checkpoint_sha256=boundary["checkpoint_sha256"], component_seconds=component_seconds,
                       training_probes_and_audits_seconds=time.monotonic() - started,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None)
    (out / "training_audit.json").write_text(json.dumps(train_audit, indent=2) + "\n")
    log(dict(stage="training_complete", updates=updates, seconds=time.monotonic() - started))
    del teachers, opts, schedules, model, warm_model
    if cuda:
        torch.cuda.empty_cache()
    records = {}
    for arm, model in students.items():
        before_eval = parameter_hashes(model)
        metrics, record = evaluate(model, valid, offset, n_rel, diagnostics=True)
        assert before_eval == parameter_hashes(model)
        records[arm] = record
        np.savez_compressed(out / f"{arm}_queries.npz", **record)
        log(dict(stage="full_valid", arm=arm, **metrics))
    with np.load(prior_dir / "frozen_queries.npz", allow_pickle=False) as reference:
        frozen = reference["rank"]
    with gzip.open(root / "ogbl_biokg/mapping/relidx2relname.csv.gz", "rt") as file:
        reader = csv.reader(file)
        next(reader)
        families = {int(i): name.split("_")[0] for i, name in reader}
    summary = summarize(records, frozen, valid, families)
    summary.update(training_audit=train_audit, boundary_audit=boundary,
                   query_sha256={arm: sha256(out / f"{arm}_queries.npz") for arm in ARMS})
    assert all(sha256(path) == digest for path, digest in inputs.items())
    assert all(sha256(repo / name) == digest for name, digest in sources.items())
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(dict(stage="complete", metrics=summary["metrics"], primary=summary["primary_comparisons"],
             advance_gate=summary["advance_gate"], source_and_input_hashes_unchanged=True))


if __name__ == "__main__":
    main()
