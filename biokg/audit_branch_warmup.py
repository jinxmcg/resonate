"""Read-only H35F endpoint and stage-boundary audit. No dataset split loads."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from biokg.branch_warmup import ARMS, STEPS, WARM_STEPS
from biokg.mixed_operator import restore_mixed
from biokg.relation_analogy import sha256
from biokg.train_joint_operator import parameter_hashes


def audit(directory):
    out, repo = Path(directory), Path(__file__).resolve().parents[1]
    receipt = json.loads((out / "prerun.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    training = json.loads((out / "training_audit.json").read_text())
    boundary = json.loads((out / "boundary_audit.json").read_text())
    assert receipt["protocol"] == summary["protocol"] == "H35F"
    assert not receipt["test_loaded"] and not summary["test_loaded"]
    assert all(sha256(repo / name) == digest for name, digest in receipt["source_hashes"].items())
    assert training["teacher_before"] == training["teacher_after"] and len(training["teacher_before"]) == 10
    assert training["teachers_unchanged"] and training["teachers_without_grad"]
    assert training["updates"] == {arm: STEPS for arm in ARMS}
    assert training["stream_sha256"]["5000"] == "8821d02f21c2b0b50154e368ab9f82271224ab1d699fae9606e4f972ce9cbe44"
    assert boundary["step"] == WARM_STEPS
    assert boundary["train_stream_sha256"] == training["stream_sha256"][str(WARM_STEPS)]
    changes = training["student_changes"]
    assert changes["joint"]["initial_hashes"] == changes["warmup"]["initial_hashes"]
    assert sha256(out / "warmup_boundary.pt") == boundary["checkpoint_sha256"] == training["boundary_checkpoint_sha256"]
    checkpoint = torch.load(out / "warmup_boundary.pt", map_location="cpu", weights_only=False)
    assert checkpoint["steps"] == WARM_STEPS and checkpoint["protocol"] == "H35F"
    model = restore_mixed(checkpoint)
    assert parameter_hashes(model) == boundary["parameter_changes"]["final_hashes"]
    for name, changed in boundary["parameter_changes"]["changed"].items():
        frozen = name.startswith("a.") or name == "gate_logit"
        assert changed == (not frozen)
        assert boundary["optimizer_steps"][name] == (0 if frozen else WARM_STEPS)
        if frozen:
            assert changes["warmup"]["initial_hashes"][name] == boundary["parameter_changes"]["final_hashes"][name]
    del model, checkpoint
    events = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]
    rows = [event for event in events if event["stage"] == "training"]
    assert set(range(100, STEPS + 1, 100)).issubset({row["step"] for row in rows})
    assert [row["step"] for row in rows] == sorted({row["step"] for row in rows})
    switches = [event for event in events if event["stage"] == "phase_switch"]
    assert len(switches) == 1 and switches[0]["step"] == WARM_STEPS + 1
    assert not switches[0]["optimizer_state_reset"]
    for row in rows:
        step = row["step"]
        assert row["updates"] == {arm: step for arm in ARMS}
        assert row["last_probe_step"] == (step // 500) * 500
        for arm in ARMS:
            warm_only = arm == "warmup" and step <= WARM_STEPS
            for group, rate in row["configured_lr"][arm].items():
                expected = (.0001 if group == "a" else .001) * (1 + math.cos(math.pi * step / STEPS)) / 2
                assert math.isclose(rate, expected, abs_tol=1e-15)
            rates = row["effective_lr"][arm]
            assert rates["a"] == (0 if warm_only else row["configured_lr"][arm]["a"])
            if arm != "single":
                assert rates["b"] == row["configured_lr"][arm]["b"]
                assert rates["gate"] == (0 if warm_only else rates["b"])
            parts = row["losses"][arm]
            assert math.isclose(parts["total"], parts["ce"] + parts["kd_t_squared"] + .1 * parts["trajectory"],
                                rel_tol=2e-6, abs_tol=2e-6)
            if warm_only:
                assert parts["trajectory"] == 0
    prior = Path(receipt["reference_directory"])
    assert sha256(prior / "frozen_queries.npz") == receipt["reference_ranks_sha256"]
    np.testing.assert_array_equal(np.load(out / "probe_indices.npy", allow_pickle=False),
                                  np.load(prior / "probe_indices.npy", allow_pickle=False))
    with np.load(prior / "frozen_queries.npz", allow_pickle=False) as ref:
        ranks = dict(frozen=ref["rank"])
    for arm in ARMS:
        assert sha256(out / f"{arm}.pt") == training["checkpoint_sha256"][arm]
        checkpoint = torch.load(out / f"{arm}.pt", map_location="cpu", weights_only=False)
        assert checkpoint["protocol"] == "H35F" and checkpoint["steps"] == STEPS and checkpoint["arm"] == arm
        assert checkpoint["source_student_sha256"] == receipt["student_sha256"]
        assert checkpoint["teacher_sha256"] == receipt["teacher_sha256"]
        model = restore_mixed(checkpoint)
        assert parameter_hashes(model) == changes[arm]["final_hashes"]
        assert all(changes[arm]["changed"].values())
        assert model.n_params() == (27124129 if arm == "single" else 30134760)
        for name, count in training["parameter_optimizer_steps"][arm].items():
            expected = 5000 if arm == "warmup" and (name.startswith("a.") or name == "gate_logit") else STEPS
            assert count == expected
        source, rel = torch.tensor([0, 100, 90000]), torch.tensor([0, 55, 101])
        candidates = torch.tensor([[1, 2, 3, 2], [111, 222, 333, 222], [999, 888, 777, 999]])
        order = [3, 1, 0, 2]
        with torch.inference_mode():
            a = model.candidate_outputs(source, rel, candidates)[0]
            b = model.candidate_outputs(source, rel, candidates[:, order])[0]
        torch.testing.assert_close(b, a[:, order], atol=1e-6, rtol=1e-6)
        assert torch.isfinite(a).all() and a[0, 1] == a[0, 3]
        del model, checkpoint
        path = out / f"{arm}_queries.npz"
        assert sha256(path) == summary["query_sha256"][arm]
        with np.load(path, allow_pickle=False) as q:
            rank = q["rank"]
            assert len(rank) == 325772 and np.isfinite(rank).all() and ((rank >= 1) & (rank <= 501)).all()
            np.testing.assert_array_equal(2 * rank, np.round(2 * rank))
            np.testing.assert_array_equal(q["rr"], 1 / rank)
            assert math.isclose(float((1 / rank.astype(np.float64)).mean()), summary["metrics"][arm]["mrr"], abs_tol=1e-15)
            ranks[arm] = rank
    comparisons = summary["primary_comparisons"]
    for control in ("joint", "single", "frozen"):
        delta = (1 / ranks["warmup"].astype(np.float64) - 1 / ranks[control].astype(np.float64)).mean()
        assert math.isclose(float(delta), comparisons["warmup_vs_" + control]["delta_mrr"], abs_tol=1e-15)
    assert summary["advance_gate"] == all(row["delta_mrr"] >= .001 and row["bootstrap_interval"][0] > 0 for row in comparisons.values())
    return dict(audit_passed=True, sources_unchanged=True, matching_examples_and_optimizer_calls=True,
                correct_parameter_update_counts=True, warmup_a_and_gate_frozen=True,
                phase_switch_and_retained_loss_verified=True, cosine_schedule_no_restart=True,
                teachers_unchanged=True, standalone_restore=True, candidate_symmetry=True,
                query_artifacts_consistent=True, split_files_loaded=False,
                checkpoint_sha256=training["checkpoint_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(json.dumps(audit(parser.parse_args().directory), indent=2))
