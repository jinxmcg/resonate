"""Audit saved diagnostic records only; no BioKG split or checkpoint loading."""

import argparse
import json
from pathlib import Path

import numpy as np

from biokg.gradient_diagnostic import sha256
from biokg.kd_calibration_diagnostic import HIGH, LOW, summarize_records


def audit(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    receipt = json.loads((directory / "prerun.json").read_text())
    records = {}
    for arm in ("original_A", "improved_A"):
        path = directory / f"{arm}_records.npz"
        assert sha256(path) == summary["record_sha256"][arm]
        with np.load(path) as saved:
            records[arm] = {key: saved[key] for key in saved.files}
    assert summarize_records(records) == summary["arms"]
    np.testing.assert_array_equal(records["original_A"]["fit"], records["improved_A"]["fit"])
    np.testing.assert_array_equal(records["original_A"]["relation"], records["improved_A"]["relation"])
    numeric = {}
    for arm, record in records.items():
        assert len(record["fit"]) == receipt["batch"] * receipt["batches"]
        mask = ~record["fit"]
        assert np.isnan(record["oracle_kd"][~mask]).all()
        assert np.isnan(record["oracle_beta"][~mask]).all()
        assert np.isfinite(record["oracle_kd"][mask]).all()
        assert ((record["oracle_beta"][mask] >= LOW) & (record["oracle_beta"][mask] <= HIGH)).all()
        delta = record["oracle_kd"][mask] - record["baseline_kd"][mask]
        # Contiguous baseline and indexed oracle batches use different fp32
        # reduction layouts. Preserve raw records and disclose last-bit drift.
        assert delta.max() <= 1e-5
        numeric[arm] = dict(oracle_above_baseline_rows=int((delta > 0).sum()),
                            largest_fp32_oracle_excess=float(delta.max()), tolerance=1e-5)
        for key in ("baseline_kd", "global_kd"):
            assert np.isfinite(record[key]).all()
        assert summary["arms"][arm]["fit"]["global_kd"]["mean"] <= (
            summary["arms"][arm]["fit"]["baseline_kd"]["mean"] + 1e-5)
    assert summary["model_state_hashes_before"] == summary["model_state_hashes_after"]
    assert summary["model_updates"] == 0
    assert summary["all_states_unchanged"] and summary["all_parameter_grads_absent"]
    assert summary["top_score_sets_unchanged"]
    assert not summary["validation_loaded"] and not summary["test_loaded"]
    assert summary["dataset_files_opened"] == ["raw/num-node-dict.csv.gz", "split/random/train.pt"]
    # Rehash sources and checkpoints as bytes, never deserialize them or reopen
    # a dataset split. The runner already audited TRAIN's before/after digest.
    checked = 0
    for path, digest in receipt["input_source_sha256"].items():
        if "/data_ogb/" not in path:
            assert sha256(path) == digest, path
            checked += 1
    return dict(audit_passed=True, records=2 * len(records["original_A"]["fit"]),
                non_dataset_input_source_hashes_checked=checked, fp32_notes=numeric,
                loaded_dataset_or_model=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(json.dumps(audit(parser.parse_args().directory), indent=2))
