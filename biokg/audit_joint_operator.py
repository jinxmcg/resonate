"""Read-only H35C endpoint audit; no dataset split or teacher model is loaded."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from biokg.joint_operator import restore_joint
from biokg.relation_analogy import sha256
from biokg.train_joint_operator import parameter_hashes


def audit(directory):
    out = Path(directory)
    repo = Path(__file__).parents[1]
    receipt = json.loads((out / "prerun.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    train = json.loads((out / "training_audit.json").read_text())
    for name, digest in receipt["source_hashes"].items():
        assert sha256(repo / name) == digest, name
    assert not receipt["test_loaded"] and not summary["test_loaded"]
    assert train["updates"] == {"single": 5000, "or": 5000}
    assert train["teachers_unchanged"] and train["teachers_without_grad"]
    assert len(train["teacher_state_sha256"]) == len(receipt["teacher_sha256"]) == 10
    events = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]
    rows = [e for e in events if e["stage"] == "training"]
    assert [e["step"] for e in rows] == list(range(100, 5001, 100))
    for event in rows:
        step = event["step"]
        assert event["updates"] == {"single": step, "or": step}
        assert event["last_probe_step"] == (step // 500) * 500
        expected_lr = .0001 * (1 + math.cos(math.pi * step / 5000)) / 2
        for mode in ("single", "or"):
            assert math.isclose(event["lr"][mode], expected_lr, abs_tol=1e-15)
            parts = event["losses"][mode]
            assert math.isclose(parts["total"], parts["ce"] + parts["kd_t_squared"]
                                + .1 * parts["trajectory"], rel_tol=2e-6, abs_tol=2e-6)
    reference = Path(receipt["args"]["h35"])
    if not reference.is_absolute():
        reference = repo / reference
    assert sha256(reference / "frozen_queries.npz") == receipt["reference_ranks_sha256"]
    np.testing.assert_array_equal(np.load(out / "probe_indices.npy"),
                                  np.load(reference / "probe_indices.npy"))
    ranks = {"frozen": np.load(reference / "frozen_queries.npz")["rank"]}
    checkpoint_hashes = {}
    parameter_counts = {}
    torch.set_num_threads(2)
    for mode in ("single", "or"):
        ck = torch.load(out / f"{mode}.pt", map_location="cpu", weights_only=False)
        assert ck["steps"] == 5000 and ck["mode"] == mode
        assert ck["source_student_sha256"] == receipt["student_sha256"]
        assert ck["teacher_sha256"] == receipt["teacher_sha256"]
        assert set(ck["model"]) == ({"E", "H_b", "log_tau"} if mode == "single"
                                    else {"E", "H_a", "H_b", "log_tau"})
        model = restore_joint(ck)
        change = train["student_changes"][mode]
        assert all(change["changed"].values())
        assert parameter_hashes(model) == change["final_hashes"]
        assert model.n_params() == (27124129 if mode == "single" else 27241633)
        assert all(not p.requires_grad for p in model.parameters())
        # Arbitrary IDs only: no TRAIN/VALID/TEST read or evaluation here.
        source = torch.tensor([0, 100, 90000])
        rel = torch.tensor([0, 55, 101])
        candidates = torch.tensor([[1, 2, 3, 2], [111, 333, 222, 222], [999, 888, 777, 999]])
        order = [3, 1, 0, 2]
        a = model.candidate_scores(source, rel, candidates)[0]
        b = model.candidate_scores(source, rel, candidates[:, order])[0]
        torch.testing.assert_close(b, a[:, order], atol=1e-6, rtol=1e-6)
        assert torch.isfinite(a).all()
        assert a[0, 1] == a[0, 3]
        checkpoint_hashes[mode] = sha256(out / f"{mode}.pt")
        parameter_counts[mode] = model.n_params()
        q = np.load(out / f"{mode}_queries.npz")
        rank = q["rank"]
        assert len(rank) == 325772 and np.isfinite(rank).all()
        assert ((rank >= 1) & (rank <= 501)).all()
        np.testing.assert_array_equal(rank * 2, np.round(rank * 2))
        np.testing.assert_allclose(q["rr"], 1 / rank, rtol=0, atol=0)
        assert math.isclose(float((1 / rank.astype(np.float64)).mean()),
                            summary["metrics"][mode]["mrr"], abs_tol=1e-15)
        ranks[mode] = rank
        del model, ck
    for key, a, b in (("single_minus_frozen", "single", "frozen"),
                      ("dual_minus_single", "or", "single"),
                      ("dual_minus_frozen", "or", "frozen")):
        delta = (1 / ranks[a].astype(np.float64) - 1 / ranks[b].astype(np.float64)).mean()
        assert math.isclose(float(delta), summary[key], abs_tol=1e-15)
    expected_gate = (summary["dual_minus_single"] >= .001 and summary["dual_minus_frozen"] >= .001
                     and summary["dual_vs_single_adjusted_97_5"][0] > 0
                     and summary["dual_vs_frozen_adjusted_97_5"][0] > 0)
    assert summary["advance_gate"] == expected_gate
    return dict(audit_passed=True, source_hashes_preserved=True, matching_update_counts=True,
                retained_loss_accounting=True, cosine_schedule_verified=True,
                endpoint_checkpoints_match_trained_tensors=True, standalone_restore=True,
                candidate_permutation_invariant=True, official_rank_artifacts_consistent=True,
                audit_split_files_loaded=False, checkpoint_sha256=checkpoint_hashes,
                real_parameter_counts=parameter_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(json.dumps(audit(parser.parse_args().directory), indent=2))
