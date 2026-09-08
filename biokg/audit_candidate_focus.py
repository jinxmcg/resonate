"""Read-only CFKD1 endpoint audit; loads own endpoints, never dataset splits."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from biokg.candidate_focus import ARMS, FOCUS_WEIGHT, STEPS, TOP_K
from biokg.gradient_diagnostic import sha256
from biokg.mixed_operator import restore_mixed
from biokg.train_candidate_focus import LR, summarize
from biokg.train_joint_operator import parameter_hashes


def audit(directory):
    out = Path(directory)
    receipt = json.loads((out / "prerun.json").read_text())
    training = json.loads((out / "training_audit.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    assert receipt["protocol"] == summary["protocol"] == "CFKD1"
    assert (receipt["steps"], receipt["top_k"], receipt["focused_weight"]) == (STEPS, TOP_K, FOCUS_WEIGHT)
    for path, digest in receipt["input_source_sha256"].items():
        if "/data_ogb/" not in path:
            assert sha256(path) == digest, path
    assert training["teacher_before"] == training["teacher_after"]
    assert len(training["teacher_before"]) == 10 and training["teachers_without_grad"]
    assert training["updates"] == dict.fromkeys(ARMS, STEPS)
    assert training["initial_parameter_hashes"]["control"] == training["initial_parameter_hashes"]["focused"]
    events = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]
    logs = [event for event in events if event["stage"] == "training"]
    assert set(range(100, STEPS + 1, 100)).issubset({row["step"] for row in logs})
    assert [row["step"] for row in logs] == sorted({row["step"] for row in logs})
    for row in logs:
        step = row["step"]
        assert row["updates"] == dict.fromkeys(ARMS, step)
        assert row["last_probe_step"] == (step // 500) * 500
        for arm in ARMS:
            assert math.isclose(row["lr"][arm], LR * (1 + math.cos(math.pi * (step - 1) / STEPS)) / 2, abs_tol=1e-15)
            assert math.isclose(row["next_lr"][arm], LR * (1 + math.cos(math.pi * step / STEPS)) / 2, abs_tol=1e-15)
            parts = row["losses"][arm]
            expected = parts["ce"] + parts["kd_t_squared"] + .1 * parts["trajectory"] + FOCUS_WEIGHT * parts["focused_kd_t_squared"]
            assert math.isclose(parts["total"], expected, rel_tol=2e-6, abs_tol=2e-6)
            if arm == "control":
                assert parts["focused_kd_t_squared"] == parts["selected_candidates"] == parts["selected_teacher_mass"] == 0
            else:
                assert TOP_K <= parts["selected_candidates"] <= receipt["negatives"] + 1
                assert 0 <= parts["selected_teacher_mass"] <= 1.000001
    assert events[-1]["stage"] == "complete"
    reference = Path(receipt["reference_directory"])
    np.testing.assert_array_equal(np.load(out / "probe_indices.npy"), np.load(reference / "probe_indices.npy"))
    with np.load(reference / "single_queries.npz") as saved:
        initial = saved["rank"]
    records, relations = {}, None
    for arm in ARMS:
        checkpoint_path = out / f"{arm}.pt"
        assert sha256(checkpoint_path) == training["checkpoint_sha256"][arm]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["protocol"] == "CFKD1" and checkpoint["arm"] == arm and checkpoint["steps"] == STEPS
        assert checkpoint["source_student_sha256"] == receipt["student_sha256"]
        assert checkpoint["teacher_sha256"] == receipt["teacher_sha256"]
        model = restore_mixed(checkpoint)
        assert model.mode == "single" and model.n_params() == 27124129
        final = parameter_hashes(model)
        assert final == training["final_parameter_hashes"][arm]
        assert all(final[name] != training["initial_parameter_hashes"][arm][name] for name in final)
        assert set(training["parameter_optimizer_steps"][arm].values()) == {STEPS}
        with torch.inference_mode():
            src, rel = torch.tensor([0, 100, 90000]), torch.tensor([0, 55, 101])
            cand = torch.tensor([[1, 2, 3, 2], [11, 22, 33, 22], [999, 888, 777, 999]])
            order = [3, 1, 0, 2]
            a = model.candidate_outputs(src, rel, cand)[0]
            b = model.candidate_outputs(src, rel, cand[:, order])[0]
            torch.testing.assert_close(b, a[:, order], atol=1e-6, rtol=1e-6)
            assert a[0, 1] == a[0, 3]
        del model, checkpoint
        path = out / f"{arm}_queries.npz"
        assert sha256(path) == summary["query_sha256"][arm]
        with np.load(path, allow_pickle=False) as saved:
            rank, rr, rels = saved["rank"], saved["rr"], saved["base_relation"]
        assert len(rank) == 325772 and len(rels) == 162886
        assert np.isfinite(rank).all() and ((rank >= 1) & (rank <= 501)).all()
        np.testing.assert_array_equal(2 * rank, np.round(2 * rank))
        np.testing.assert_array_equal(rr, 1 / rank)
        if relations is not None:
            np.testing.assert_array_equal(rels, relations)
        relations = rels
        records[arm] = dict(rank=rank, rr=rr)
    families = json.loads((out / "relation_families.json").read_text())
    regenerated = summarize(records, initial, relations, families)
    for name, value in regenerated.items():
        assert summary[name] == value, name
    assert summary["dataset_files_opened"] == ["raw/num-node-dict.csv.gz", "split/random/train.pt", "split/random/valid.pt"]
    assert not summary["test_loaded"] and not receipt["test_loaded"]
    return dict(audit_passed=True, matched_initialization_and_updates=True, teachers_unchanged=True,
                losses_and_schedule_verified=True, standalone_restore_and_candidate_symmetry=True,
                all_summary_metrics_and_intervals_reproduced=True, dataset_splits_loaded=False,
                checkpoint_sha256=training["checkpoint_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(json.dumps(audit(parser.parse_args().directory), indent=2))
