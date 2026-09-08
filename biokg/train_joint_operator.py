"""H35C lockstep joint KD comparison. Protocol: biokg/H35C.md. No TEST option."""

import argparse
import csv
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch

from biokg.joint_operator import JointOperator, from_student, retained_loss
from biokg.relation_analogy import sha256
from biokg.train_biokg_comp import load_teacher, teacher_logits
from biokg.train_dual_operator import (
    TrainStream, evaluate, paired_interval, rank_summary, tensor_digest, validation_only,
)


def parameter_hashes(model):
    return {name: tensor_digest({name: p}) for name, p in model.named_parameters()}


def gradient_norms(model):
    return {name: float(p.grad.norm()) if p.grad is not None else None
            for name, p in model.named_parameters()}


def teacher_hashes(teachers):
    return [tensor_digest(teacher.state_dict()) for teacher in teachers]


def assert_independent(students, teachers):
    seen = set()
    for model in [*students.values(), *teachers]:
        addresses = {p.data_ptr() for p in model.parameters()}
        if addresses & seen:
            raise AssertionError("Model parameter storage is shared")
        seen.update(addresses)


def synthetic_smoke(device):
    """Full-size entity tables and training candidate shape; no BioKG reads."""
    from resonate import ResonatE
    torch.manual_seed(3510)
    teachers = [ResonatE(93773, 102, k=12, block=True, block_size=4)
                .to(device).eval().requires_grad_(False) for _ in range(10)]
    base = teachers[0]
    students = {mode: JointOperator(base.E, base.H, base.log_tau, mode)
                for mode in ("single", "or")}
    assert_independent(students, teachers)
    before_teachers = teacher_hashes(teachers)
    before_students = {mode: parameter_hashes(model) for mode, model in students.items()}
    opts = {mode: torch.optim.Adam(model.parameters(), lr=.0001) for mode, model in students.items()}
    source = torch.arange(2048, device=device)
    rel = torch.zeros(2048, dtype=torch.long, device=device)
    positive = (source + 1) % 93773
    negatives = torch.arange(4096, device=device)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.monotonic()
    for step in range(20):
        target = teacher_logits(teachers, source, rel, positive, negatives)
        for mode, model in students.items():
            opts[mode].zero_grad(set_to_none=True)
            loss, _ = retained_loss(model.training_outputs(source, rel, positive, negatives), target)
            loss.backward()
            grads = gradient_norms(model) if step == 0 else None
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opts[mode].step()
            if step == 0:
                assert all(value is not None and value > 0 for value in grads.values())
                print(json.dumps(dict(stage="smoke_gradients", arm=mode, norms=grads)), flush=True)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    assert teacher_hashes(teachers) == before_teachers
    assert all(p.grad is None for t in teachers for p in t.parameters())
    for mode, model in students.items():
        after = parameter_hashes(model)
        assert all(after[name] != before_students[mode][name] for name in after)
    print(json.dumps(dict(stage="smoke_complete", paired_updates=20, seconds=elapsed,
                         estimated_5000_pair_training_seconds=elapsed * 250,
                         peak_allocated_bytes=(torch.cuda.max_memory_allocated()
                                               if torch.device(device).type == "cuda" else None),
                         all_student_groups_updated=True, teachers_unchanged=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--teacher-dir")
    parser.add_argument("--checksums")
    parser.add_argument("--data-root")
    parser.add_argument("--h35")
    parser.add_argument("--out")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    cuda = torch.device(args.device).type == "cuda"
    if args.smoke:
        synthetic_smoke(args.device)
        return
    if not all((args.model, args.teacher_dir, args.checksums, args.data_root, args.h35, args.out)):
        parser.error("Missing model, teachers, checksums, data root, H35 reference or output")
    prior = Path(args.h35)
    prior_receipt = json.loads((prior / "prerun.json").read_text())
    expected_student = "b463a8bcc431ada38f4834e235677464f9cd3a86346f88aecd2fd681a01260ef"
    if sha256(args.model) != expected_student or prior_receipt["checkpoint_sha256"] != expected_student:
        raise ValueError("Wrong released student")
    dataset = Path(args.data_root) / "ogbl_biokg"
    for name, digest in prior_receipt["data_hashes"].items():
        if sha256(dataset / name) != digest:
            raise ValueError("Dataset differs from reference")
    expected = {line.split()[1]: line.split()[0] for line in Path(args.checksums).read_text().splitlines()}
    teacher_files = [Path(args.teacher_dir) / f"sparse_s{i}.pt" for i in range(10)]
    teacher_files_sha = {p.name: sha256(p) for p in teacher_files}
    if any(digest != expected[name] for name, digest in teacher_files_sha.items()):
        raise ValueError("Teacher checksum mismatch")
    repo, out = Path(__file__).parents[1], Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(args=vars(args), student_sha256=expected_student, teacher_sha256=teacher_files_sha,
                   checksums_sha256=sha256(args.checksums), data_hashes=prior_receipt["data_hashes"],
                   reference_ranks_sha256=sha256(prior / "frozen_queries.npz"),
                   source_hashes={name: sha256(repo / name) for name in (
                       "biokg/H35C.md", "biokg/joint_operator.py", "biokg/train_joint_operator.py",
                       "biokg/test_joint_operator.py",
                       "biokg/dual_operator.py", "biokg/train_dual_operator.py",
                       "biokg/train_biokg_comp.py", "resonate.py")},
                   steps=5000, seed=3510, batch=2048, negatives=4096, lr=.0001,
                   kd_temperature=2., kd_weight=1., trajectory_weight=.1,
                   torch=torch.__version__, numpy=np.__version__,
                   device=(torch.cuda.get_device_name() if cuda else args.device), test_loaded=False)
    (out / "prerun.json").write_text(json.dumps(receipt, indent=2) + "\n")
    def log(event):
        print(json.dumps(event), flush=True)
        with (out / "progress.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")
    split, offset, n_ent, _, num_nodes = validation_only(args.data_root)
    n_rel = 2 * (int(split["train"]["relation"].max()) + 1)
    student_ck = torch.load(args.model, map_location="cpu", weights_only=False)
    if student_ck["offset"] != offset or student_ck["n_rel"] != n_rel:
        raise ValueError("Student indexing mismatch")
    teachers = []
    for i, path in enumerate(teacher_files):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (checkpoint["offset"] != offset or checkpoint["n_rel"] != n_rel
                or checkpoint["args"]["seed"] != i or checkpoint["args"].get("distill")):
            raise ValueError("Teacher indexing/recipe mismatch")
        teachers.append(load_teacher(str(path), n_ent, n_rel, torch.device(args.device)))
    teacher_before = teacher_hashes(teachers)
    students = {mode: from_student(student_ck, mode, args.device) for mode in ("single", "or")}
    assert_independent(students, teachers)
    initial = {mode: parameter_hashes(model) for mode, model in students.items()}
    for mode, model in students.items():
        assert all(p.requires_grad for p in model.parameters())
        assert model.n_params(True) == (27124129 if mode == "single" else 27241633)
    opts = {mode: torch.optim.Adam(model.parameters(), lr=.0001) for mode, model in students.items()}
    schedules = {mode: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000)
                 for mode, opt in opts.items()}
    valid = split["valid"]
    ids = np.random.default_rng(3501).choice(len(valid["head"]), 1000, replace=False)
    np.testing.assert_array_equal(ids, np.load(prior / "probe_indices.npy"))
    np.save(out / "probe_indices.npy", ids)
    probe = {name: np.asarray(value)[ids] for name, value in valid.items()}
    probe_mrr = {}
    for mode, model in students.items():
        metrics, _ = evaluate(model, probe, offset, n_rel, diagnostics=False)
        probe_mrr[mode] = metrics["mrr"]
        log(dict(stage="initial_probe", arm=mode, step=0, lr=.0001, **metrics))
    prior_events = [json.loads(line) for line in (prior / "progress.jsonl").read_text().splitlines()]
    frozen_probe = next(event["probe_mrr"] for event in prior_events
                        if event.get("arm") == "single" and event.get("step") == 0)
    if abs(probe_mrr["single"] - frozen_probe) > 1e-6:
        raise AssertionError("Control initialization differs from original student")
    stream = TrainStream(split["train"], offset, num_nodes, seed=3510)
    started = time.monotonic()
    teacher_seconds = 0.
    student_seconds = {mode: 0. for mode in students}
    updates = {mode: 0 for mode in students}
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, 5001):
        source, positive, relation, negatives = stream.sample(2048, 4096)
        source, positive, negatives = [torch.as_tensor(x, device=args.device)
                                       for x in (source, positive, negatives)]
        relation = torch.full((len(source),), relation, device=args.device, dtype=torch.long)
        tick = time.monotonic()
        target = teacher_logits(teachers, source, relation, positive, negatives)
        if cuda:
            torch.cuda.synchronize()
        teacher_seconds += time.monotonic() - tick
        losses = {}
        for mode, model in students.items():
            tick = time.monotonic()
            model.train()
            opts[mode].zero_grad(set_to_none=True)
            outputs = model.training_outputs(source, relation, positive, negatives)
            loss, parts = retained_loss(outputs, target)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss: {mode} step {step}")
            loss.backward()
            if step == 1:
                norms = gradient_norms(model)
                assert all(value is not None and value > 0 and np.isfinite(value) for value in norms.values())
                log(dict(stage="first_gradients", arm=mode, norms=norms))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opts[mode].step()
            schedules[mode].step()
            if cuda:
                torch.cuda.synchronize()
            student_seconds[mode] += time.monotonic() - tick
            updates[mode] += 1
            if step % 100 == 0:
                losses[mode] = dict(total=float(loss), **{name: float(value) for name, value in parts.items()})
            del outputs, loss, parts
        del target
        if step % 500 == 0:
            for mode, model in students.items():
                metrics, _ = evaluate(model, probe, offset, n_rel, diagnostics=False)
                probe_mrr[mode] = metrics["mrr"]
        if step % 100 == 0:
            log(dict(stage="training", step=step, updates=updates,
                     lr={mode: opt.param_groups[0]["lr"] for mode, opt in opts.items()},
                     losses=losses, last_probe_step=(step // 500) * 500, probe_mrr=probe_mrr,
                     elapsed_seconds=time.monotonic() - started))
    assert updates == {"single": 5000, "or": 5000}
    teacher_after = teacher_hashes(teachers)
    assert teacher_before == teacher_after, "Teacher tensors changed"
    assert all(p.grad is None and not p.requires_grad for t in teachers for p in t.parameters())
    change_audit = {}
    for mode, model in students.items():
        final_hashes = parameter_hashes(model)
        changed = {name: digest != initial[mode][name] for name, digest in final_hashes.items()}
        assert all(changed.values()), "A student parameter group did not change"
        change_audit[mode] = dict(changed=changed, initial_hashes=initial[mode], final_hashes=final_hashes)
        torch.save(dict(model_type="H35CJointOperator", model=model.state_dict(), mode=mode,
                        args=vars(args), steps=5000, offset=offset, n_rel=n_rel,
                        source_student_sha256=expected_student, teacher_sha256=teacher_files_sha),
                   out / f"{mode}.pt")
    training_audit = dict(teacher_state_sha256=teacher_before, teachers_unchanged=True,
                         teachers_without_grad=True, student_changes=change_audit,
                         train_stream_sha256=stream.digest.hexdigest(), updates=updates,
                         teacher_forward_seconds=teacher_seconds, student_update_seconds=student_seconds,
                         training_and_probes_elapsed_seconds=time.monotonic() - started,
                         peak_allocated_bytes=(torch.cuda.max_memory_allocated() if cuda else None))
    (out / "training_audit.json").write_text(json.dumps(training_audit, indent=2) + "\n")
    log(dict(stage="training_complete", updates=updates, teachers_unchanged=True,
             all_student_groups_changed=True, elapsed_seconds=time.monotonic()-started))
    del teachers, opts, schedules
    if cuda:
        torch.cuda.empty_cache()
    records, metrics_by_arm = {}, {}
    for mode, model in students.items():
        metrics, queries = evaluate(model, valid, offset, n_rel)
        metrics_by_arm[mode], records[mode] = metrics, queries
        np.savez_compressed(out / f"{mode}_queries.npz", **queries)
        log(dict(stage="full_valid", arm=mode, **metrics))
    reference_rank = np.load(prior / "frozen_queries.npz")["rank"]
    rr_frozen = 1 / reference_rank.astype(np.float64)
    rr_single = 1 / records["single"]["rank"].astype(np.float64)
    rr_dual = 1 / records["or"]["rank"].astype(np.float64)
    ds, df = rr_dual - rr_single, rr_dual - rr_frozen
    rng = np.random.default_rng(3512)
    cis, cif = paired_interval(ds, [.0125, .9875], rng), paired_interval(df, [.0125, .9875], rng)
    with gzip.open(dataset / "mapping/relidx2relname.csv.gz", "rt") as f:
        reader = csv.reader(f)
        next(reader)
        family_names = {int(i): name.split("_")[0] for i, name in reader}
    fam = np.array([family_names[int(r)] for r in np.tile(valid["relation"], 2)])
    dirs = np.repeat([0, 1], len(valid["head"]))
    slices = []
    for family in np.unique(fam):
        for direction in (0, 1):
            mask = (fam == family) & (dirs == direction)
            slices.append(dict(family=family, direction=direction, queries=int(mask.sum()),
                               frozen_mrr=float(rr_frozen[mask].mean()), single_mrr=float(rr_single[mask].mean()),
                               dual_mrr=float(rr_dual[mask].mean()), dual_minus_single=float(ds[mask].mean())))
    q = records["or"]
    summary = dict(screen_only=True, test_loaded=False, frozen=rank_summary(reference_rank),
                   metrics=metrics_by_arm, single_minus_frozen=float((rr_single-rr_frozen).mean()),
                   dual_minus_single=float(ds.mean()), dual_minus_frozen=float(df.mean()),
                   dual_vs_single_adjusted_97_5=cis, dual_vs_frozen_adjusted_97_5=cif,
                   advance_gate=bool(ds.mean() >= .001 and df.mean() >= .001 and cis[0] > 0 and cif[0] > 0),
                   branches=dict(a=rank_summary(q["branch_a_rank"]), b=rank_summary(q["branch_b_rank"]),
                                 mean_query_cosine=float(q["query_cosine"].mean()),
                                 median_query_cosine=float(np.median(q["query_cosine"])),
                                 mean_b_positive_responsibility=float(q["b_positive_responsibility"].mean()),
                                 b_positive_stronger_fraction=float(q["b_positive_stronger"].mean())),
                   family_direction=slices, training_audit=training_audit)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(dict(stage="complete", **{name: value for name, value in summary.items()
                                  if name not in ("family_direction", "training_audit")}))


if __name__ == "__main__":
    main()
