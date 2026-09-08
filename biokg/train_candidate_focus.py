"""CFKD1 fixed-budget A-only paired experiment. No TEST option."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from biokg.candidate_focus import ARMS, FOCUS_WEIGHT, SEED, STEPS, TOP_K, focused_loss
from biokg.gradient_diagnostic import sha256, train_only
from biokg.joint_operator import JointOperator
from biokg.mixed_operator import MixedOperator, restore_mixed
from biokg.train_biokg_comp import load_teacher, teacher_logits
from biokg.train_dual_operator import TrainStream, rank_summary
from biokg.train_joint_operator import assert_independent, gradient_norms, parameter_hashes, teacher_hashes
from biokg.train_mixed_operator import compare, evaluate


BATCH, NEGATIVES, LR = 2048, 4096, 1e-4


def check_data_path(path, dataset):
    path, dataset = Path(path).resolve(), Path(dataset).resolve()
    if path.is_relative_to(dataset):
        relative = str(path.relative_to(dataset))
        if relative not in ("split/random/train.pt", "split/random/valid.pt", "raw/num-node-dict.csv.gz"):
            raise PermissionError(f"CFKD1 cannot open dataset file: {relative}")
        return relative
    return None


def train_valid(dataset):
    train, offset, counts = train_only(dataset)
    valid = torch.load(Path(dataset) / "split/random/valid.pt", map_location="cpu", weights_only=False)
    return train, valid, offset, counts


def optimizer_steps(model, opt):
    return {name: int(opt.state.get(p, {}).get("step", 0)) for name, p in model.named_parameters()}


def make_checkpoint(model, arm, offset, n_rel, receipt):
    return dict(model_type="H35EMixedOperator", protocol="CFKD1", model=model.state_dict(),
                mode="single", width=model.width, arm=arm, steps=STEPS, offset=offset, n_rel=n_rel,
                source_student_sha256=receipt["student_sha256"], teacher_sha256=receipt["teacher_sha256"],
                training_only_focus_weight=FOCUS_WEIGHT if arm == "focused" else 0., top_k=TOP_K)


def summarize(records, initial_rank, base_relations, families):
    ranks = {arm: record["rank"] for arm, record in records.items()}
    ranks["initial"] = initial_rank
    primary = compare(ranks["focused"], ranks["control"], np.random.default_rng(3604), [.025, .975])
    descriptive = {arm + "_vs_initial": compare(ranks[arm], initial_rank, np.random.default_rng(3605 + i),
                                                [.025, .975]) for i, arm in enumerate(ARMS)}
    metrics = {arm: rank_summary(rank) for arm, rank in ranks.items()}
    family = np.array([families[str(int(r))] for r in np.tile(base_relations, 2)])
    direction = np.repeat(["tail", "head"], len(base_relations))
    slices = []
    for name in np.unique(family):
        for direct in ("head", "tail"):
            mask = (family == name) & (direction == direct)
            slices.append(dict(family=name, direction=direct, queries=int(mask.sum()),
                               mrr={arm: rank_summary(rank[mask])["mrr"] for arm, rank in ranks.items()}))
    return dict(protocol="CFKD1", screen_only=True, test_loaded=False, metrics=metrics,
                primary_focused_vs_control=primary, descriptive_vs_initial=descriptive,
                descriptive_family_direction=slices,
                advance_for_seed_confirmation=(primary["delta_mrr"] >= .001 and primary["bootstrap_interval"][0] > 0
                                               and metrics["focused"]["mrr"] >= metrics["initial"]["mrr"]))


def synthetic_smoke():
    from resonate import ResonatE
    torch.manual_seed(SEED)
    teachers = [ResonatE(93773, 102, k=12, block=True, block_size=4).cuda().eval().requires_grad_(False)
                for _ in range(10)]
    base = teachers[0]
    students = {arm: MixedOperator(JointOperator(base.E, base.H, base.log_tau, "single"), "single") for arm in ARMS}
    assert_independent(students, teachers)
    before = {arm: parameter_hashes(model) for arm, model in students.items()}
    assert before["control"] == before["focused"]
    teacher_before = teacher_hashes(teachers)
    opts = {arm: torch.optim.Adam(model.parameters(), lr=LR) for arm, model in students.items()}
    source = torch.arange(BATCH, device="cuda")
    batch = source, torch.zeros_like(source), (source + 1) % 93773, torch.arange(NEGATIVES, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    for step in range(20):
        target = teacher_logits(teachers, *batch)
        for arm, model in students.items():
            opts[arm].zero_grad(set_to_none=True)
            loss, parts = focused_loss(model.training_outputs(*batch), target, arm == "focused")
            assert torch.isfinite(loss)
            loss.backward()
            if step == 0:
                norms = gradient_norms(model)
                assert all(value is not None and value > 0 and np.isfinite(value) for value in norms.values())
                print(json.dumps(dict(stage="smoke_gradients", arm=arm, norms=norms)), flush=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opts[arm].step()
        del loss, parts, target
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    assert teacher_before == teacher_hashes(teachers)
    assert all(not p.requires_grad and p.grad is None for t in teachers for p in t.parameters())
    for arm, model in students.items():
        after = parameter_hashes(model)
        assert all(after[name] != before[arm][name] for name in after)
        assert set(optimizer_steps(model, opts[arm]).values()) == {20}
        with torch.inference_mode():
            candidates = torch.arange(501, device="cuda")[None].expand(128, -1)
            order = torch.randperm(501, device="cuda")
            a = model.candidate_outputs(batch[0][:128], batch[1][:128], candidates)[0]
            b = model.candidate_outputs(batch[0][:128], batch[1][:128], candidates[:, order])[0]
            torch.testing.assert_close(b, a[:, order], atol=0, rtol=0)
    print(json.dumps(dict(stage="smoke_complete", updates_per_arm=20, seconds=elapsed,
                         estimated_5000_training_seconds=elapsed * 250,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         all_groups_updated=True, teachers_unchanged=True, candidate_symmetry=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="biokg/results/candidate_focus/s0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available() or "1080 Ti" not in torch.cuda.get_device_name():
        raise RuntimeError("CFKD1 requires the local GTX 1080 Ti")
    if args.smoke:
        synthetic_smoke()
        return
    repo, out = Path(__file__).resolve().parents[1], Path(args.out)
    dataset = Path("/mnt/geocore/geocore/data_ogb/ogbl_biokg")
    opened = set()

    def audit(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes)):
            path = arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]
            relative = check_data_path(path, dataset)
            if relative:
                opened.add(relative)

    sys.addaudithook(audit)
    prior = repo / "biokg/results/h35f/campaign_s0"
    prior_receipt = json.loads((prior / "prerun.json").read_text())
    old_receipt = repo / "biokg/results/gradient_diagnostic/s0/prerun.json"
    inputs = json.loads(old_receipt.read_text())["input_source_sha256"]
    for name in ("biokg/train_joint_operator.py", "biokg/train_mixed_operator.py"):
        inputs[str(repo / name)] = prior_receipt["source_hashes"][name]
    inputs[str(dataset / "split/random/valid.pt")] = prior_receipt["input_sha256"][str(dataset / "split/random/valid.pt")]
    old_summary = json.loads((prior / "summary.json").read_text())
    inputs[str(prior / "single_queries.npz")] = old_summary["query_sha256"]["single"]
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise ValueError(f"Prior input/source changed: {path}")
    for name in ("biokg/CFKD1.md", "biokg/candidate_focus.py", "biokg/train_candidate_focus.py",
                 "biokg/test_candidate_focus.py", "biokg/audit_candidate_focus.py"):
        inputs[str(repo / name)] = sha256(repo / name)
    for path in (old_receipt, prior / "summary.json", prior / "probe_indices.npy"):
        inputs[str(path)] = sha256(path)
    student_path = prior / "single.pt"
    receipt = dict(protocol="CFKD1", steps=STEPS, seed=SEED, batch=BATCH, negatives=NEGATIVES,
                   top_k=TOP_K, focused_weight=FOCUS_WEIGHT, lr=LR, kd_temperature=2., trajectory_weight=.1,
                   student_sha256=inputs[str(student_path)], teacher_sha256=prior_receipt["teacher_sha256"],
                   reference_directory=str(prior), input_source_sha256=inputs,
                   torch=torch.__version__, numpy=np.__version__, device=torch.cuda.get_device_name(),
                   test_loaded=False, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    out.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def log(**event):
        print(json.dumps(event, allow_nan=False), flush=True)
        with (out / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")

    write("prerun.json", receipt)
    log(stage="inputs_verified", steps=STEPS, arms=list(ARMS))
    train, valid, offset, counts = train_valid(dataset)
    stream = TrainStream(train, offset, counts, seed=SEED)
    n_ent, n_rel = sum(counts.values()), stream.n_rel_base * 2
    assert (n_ent, n_rel, len(valid["head"])) == (93773, 102, 162886)
    checkpoint = torch.load(student_path, map_location="cpu", weights_only=False)
    assert checkpoint["offset"] == offset and checkpoint["n_rel"] == n_rel and checkpoint["mode"] == "single"
    students = {arm: restore_mixed(checkpoint, "cuda").requires_grad_(True) for arm in ARMS}
    del checkpoint
    teachers = []
    for seed in range(10):
        path = Path("/mnt/geocore/wiki_pull/h24/dense") / f"sparse_s{seed}.pt"
        meta = torch.load(path, map_location="cpu", weights_only=False)
        assert meta["offset"] == offset and meta["n_rel"] == n_rel and meta["args"]["seed"] == seed
        assert not meta["args"].get("distill")
        del meta
        teachers.append(load_teacher(path, n_ent, n_rel, torch.device("cuda")))
    assert_independent(students, teachers)
    initial = {arm: parameter_hashes(model) for arm, model in students.items()}
    assert initial["control"] == initial["focused"]
    assert all(model.n_params() == 27124129 for model in students.values())
    teacher_before = teacher_hashes(teachers)
    opts = {arm: torch.optim.Adam(model.parameters(), lr=LR) for arm, model in students.items()}
    schedules = {arm: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS) for arm, opt in opts.items()}
    ids = np.random.default_rng(3501).choice(len(valid["head"]), 1000, replace=False)
    np.testing.assert_array_equal(ids, np.load(prior / "probe_indices.npy", allow_pickle=False))
    np.save(out / "probe_indices.npy", ids)
    probe = {name: np.asarray(value)[ids] for name, value in valid.items()}
    with np.load(prior / "single_queries.npz", allow_pickle=False) as saved:
        initial_rank = saved["rank"]
    expected_probe = np.r_[initial_rank[ids], initial_rank[len(valid["head"]) + ids]]
    probe_mrr = {}
    for arm, model in students.items():
        metrics, query = evaluate(model, probe, offset, n_rel)
        np.testing.assert_array_equal(query["rank"], expected_probe)
        assert parameter_hashes(model) == initial[arm]
        probe_mrr[arm] = metrics["mrr"]
    log(stage="initial_probe", step=0, lr=LR, probe_mrr=probe_mrr)
    families = {str(i): str(train["head_type"][rows[0]]) + "-" + str(train["tail_type"][rows[0]])
                for i, rows in enumerate(stream.by_rel)}
    write("relation_families.json", families)
    updates = dict.fromkeys(ARMS, 0)
    component_seconds = dict.fromkeys(("teacher", *ARMS), 0.)
    started = last_log = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, STEPS + 1):
        source, positive, relation, negatives = stream.sample(BATCH, NEGATIVES)
        source, positive, negatives = [torch.as_tensor(x, device="cuda") for x in (source, positive, negatives)]
        rel = torch.full((BATCH,), relation, device="cuda", dtype=torch.long)
        batch = source, rel, positive, negatives
        tick = time.monotonic()
        target = teacher_logits(teachers, *batch)
        torch.cuda.synchronize()
        component_seconds["teacher"] += time.monotonic() - tick
        should_log = step % 100 == 0 or time.monotonic() - last_log >= 40
        losses, rates = {}, {}
        for arm, model in students.items():
            tick = time.monotonic()
            model.train()
            opts[arm].zero_grad(set_to_none=True)
            rates[arm] = opts[arm].param_groups[0]["lr"]
            loss, parts = focused_loss(model.training_outputs(*batch), target, arm == "focused")
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite {arm} loss at {step}")
            loss.backward()
            if step == 1:
                norms = gradient_norms(model)
                assert all(value is not None and value > 0 and np.isfinite(value) for value in norms.values())
                log(stage="first_gradients", arm=arm, norms=norms)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opts[arm].step()
            schedules[arm].step()
            updates[arm] += 1
            torch.cuda.synchronize()
            component_seconds[arm] += time.monotonic() - tick
            if should_log:
                losses[arm] = dict(total=float(loss), grad_norm_before_clip=float(grad_norm),
                                   **{name: float(value) for name, value in parts.items()})
            del loss, parts
        del target
        if step % 500 == 0:
            probe_mrr = {arm: evaluate(model, probe, offset, n_rel)[0]["mrr"] for arm, model in students.items()}
        if should_log:
            log(stage="training", step=step, updates=updates, lr=rates,
                next_lr={arm: opt.param_groups[0]["lr"] for arm, opt in opts.items()},
                losses=losses, last_probe_step=(step // 500) * 500, probe_mrr=probe_mrr,
                elapsed_seconds=time.monotonic() - started)
            last_log = time.monotonic()
    assert updates == dict.fromkeys(ARMS, STEPS)
    teacher_after = teacher_hashes(teachers)
    assert teacher_before == teacher_after
    assert all(not p.requires_grad and p.grad is None for teacher in teachers for p in teacher.parameters())
    final, checkpoints, steps_by_parameter = {}, {}, {}
    for arm, model in students.items():
        final[arm] = parameter_hashes(model)
        assert all(final[arm][name] != initial[arm][name] for name in final[arm])
        steps_by_parameter[arm] = optimizer_steps(model, opts[arm])
        assert set(steps_by_parameter[arm].values()) == {STEPS}
        torch.save(make_checkpoint(model, arm, offset, n_rel, receipt), out / f"{arm}.pt")
        checkpoints[arm] = sha256(out / f"{arm}.pt")
        restored = restore_mixed(torch.load(out / f"{arm}.pt", map_location="cpu", weights_only=False))
        assert parameter_hashes(restored) == final[arm]
        del restored
    training_audit = dict(updates=updates, parameter_optimizer_steps=steps_by_parameter,
                          initial_parameter_hashes=initial, final_parameter_hashes=final,
                          teacher_before=teacher_before, teacher_after=teacher_after, teachers_unchanged=True,
                          teachers_without_grad=True, checkpoint_sha256=checkpoints,
                          real_parameter_counts={arm: model.n_params() for arm, model in students.items()},
                          stream_sha256=stream.digest.hexdigest(), component_seconds=component_seconds,
                          training_probes_audits_seconds=time.monotonic() - started,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write("training_audit.json", training_audit)
    log(stage="training_complete", updates=updates, seconds=time.monotonic() - started)
    del teachers, opts, schedules
    torch.cuda.empty_cache()
    records = {}
    for arm, model in students.items():
        metrics, query = evaluate(model, valid, offset, n_rel)
        assert parameter_hashes(model) == final[arm]
        records[arm] = query
        np.savez_compressed(out / f"{arm}_queries.npz", **query, base_relation=valid["relation"])
        log(stage="full_valid", arm=arm, **metrics)
    summary = summarize(records, initial_rank, valid["relation"], families)
    summary.update(query_sha256={arm: sha256(out / f"{arm}_queries.npz") for arm in ARMS},
                   dataset_files_opened=sorted(opened), source_and_input_hashes_unchanged=True,
                   total_seconds=time.monotonic() - started)
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise ValueError(f"Input/source changed: {path}")
    write("summary.json", summary)
    log(stage="complete", metrics=summary["metrics"], primary=summary["primary_focused_vs_control"],
        advance_for_seed_confirmation=summary["advance_for_seed_confirmation"])


if __name__ == "__main__":
    main()
