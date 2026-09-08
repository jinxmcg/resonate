"""Read-only H35E artifact/standalone-checkpoint audit. No split loads."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from biokg.mixed_operator import MODES, restore_mixed
from biokg.relation_analogy import sha256
from biokg.train_joint_operator import parameter_hashes


def audit(directory):
    out = Path(directory)
    repo = Path(__file__).resolve().parents[1]
    receipt = json.loads((out / "prerun.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    training = json.loads((out / "training_audit.json").read_text())
    assert not receipt["test_loaded"] and not summary["test_loaded"]
    assert all(sha256(repo / name) == digest for name, digest in receipt["source_hashes"].items())
    assert training["teacher_before"] == training["teacher_after"] and len(training["teacher_before"]) == 10
    assert training["teachers_unchanged"] and training["teachers_without_grad"]
    assert training["updates"] == {mode: 5000 for mode in MODES}
    assert training["train_stream_sha256"] == "8821d02f21c2b0b50154e368ab9f82271224ab1d699fae9606e4f972ce9cbe44"
    changes = training["student_changes"]
    assert changes["dot"]["initial_hashes"] == changes["distance"]["initial_hashes"]
    events = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]
    rows = [e for e in events if e["stage"] == "training"]
    assert set(range(100, 5001, 100)).issubset({row["step"] for row in rows})
    assert [row["step"] for row in rows] == sorted({row["step"] for row in rows})
    for row in rows:
        step = row["step"]
        assert row["updates"] == {mode: step for mode in MODES}
        assert row["last_probe_step"] == (step // 500) * 500
        for mode in MODES:
            for group, rate in row["lr"][mode].items():
                expected = (.0001 if group == "a" else .001) * (1 + math.cos(math.pi * step / 5000)) / 2
                assert math.isclose(rate, expected, abs_tol=1e-15)
            loss = row["losses"][mode]
            assert math.isclose(loss["total"], loss["ce"] + loss["kd_t_squared"] + .1 * loss["trajectory"],
                                rel_tol=2e-6, abs_tol=2e-6)
    prior = Path(receipt["reference_directory"])
    assert sha256(prior / "frozen_queries.npz") == receipt["reference_ranks_sha256"]
    np.testing.assert_array_equal(np.load(out / "probe_indices.npy", allow_pickle=False),
                                  np.load(prior / "probe_indices.npy", allow_pickle=False))
    with np.load(prior / "frozen_queries.npz", allow_pickle=False) as ref:
        ranks = dict(frozen=ref["rank"])
    for mode in MODES:
        path = out / f"{mode}.pt"
        assert sha256(path) == training["checkpoint_sha256"][mode]
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["steps"] == 5000 and ck["mode"] == mode and ck["width"] == 32
        assert ck["source_student_sha256"] == receipt["student_sha256"]
        assert ck["teacher_sha256"] == receipt["teacher_sha256"]
        model = restore_mixed(ck)
        assert parameter_hashes(model) == changes[mode]["final_hashes"]
        assert all(changes[mode]["changed"].values())
        assert model.n_params() == (27124129 if mode == "single" else 30134760)
        assert all(not p.requires_grad for p in model.parameters())
        source = torch.tensor([0, 100, 90000])
        rel = torch.tensor([0, 55, 101])
        candidates = torch.tensor([[1, 2, 3, 2], [111, 222, 333, 222], [999, 888, 777, 999]])
        order = [3, 1, 0, 2]
        with torch.inference_mode():
            original = model.candidate_outputs(source, rel, candidates)[0]
            perm = model.candidate_outputs(source, rel, candidates[:, order])[0]
        torch.testing.assert_close(perm, original[:, order], atol=1e-6, rtol=1e-6)
        assert torch.isfinite(original).all() and original[0, 1] == original[0, 3]
        del model, ck
        path = out / f"{mode}_queries.npz"
        assert sha256(path) == summary["query_sha256"][mode]
        with np.load(path, allow_pickle=False) as q:
            rank = q["rank"]
            assert len(rank) == 325772 and np.isfinite(rank).all()
            assert ((rank >= 1) & (rank <= 501)).all()
            np.testing.assert_array_equal(rank * 2, np.round(rank * 2))
            np.testing.assert_array_equal(q["rr"], 1 / rank)
            assert math.isclose(float((1 / rank.astype(np.float64)).mean()), summary["metrics"][mode]["mrr"], abs_tol=1e-15)
            ranks[mode] = rank
    comparisons = summary["primary_comparisons"]
    for control in ("dot", "single", "frozen"):
        delta = (1 / ranks["distance"].astype(np.float64) - 1 / ranks[control].astype(np.float64)).mean()
        assert math.isclose(float(delta), comparisons["distance_vs_" + control]["delta_mrr"], abs_tol=1e-15)
    assert summary["advance_gate"] == all(row["delta_mrr"] >= .001 and row["bootstrap_interval"][0] > 0
                                          for row in comparisons.values())
    return dict(audit_passed=True, source_hashes_preserved=True, matching_update_counts=True,
                matched_dot_distance_initialization=True, retained_loss_verified=True,
                lr_schedules_verified=True, teachers_unchanged=True, all_student_groups_changed=True,
                standalone_restore=True, symmetric_candidate_scores=True, query_artifacts_consistent=True,
                split_files_loaded=False, checkpoint_sha256=training["checkpoint_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(json.dumps(audit(parser.parse_args().directory), indent=2))
