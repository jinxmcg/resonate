"""Separate TP1 replay: reconstruct TRAIN partitions and sample direct paths."""

import argparse
import json
from pathlib import Path

import numpy as np

from biokg.typed_path_pilot import (
    EXPECTED, PATH_TYPES, WEIGHTS, candidates_for, direct_path, install_guard,
    normalize, prepare, read_train, report_arrays, select_recipe, summarize,
    verify, write_json,
)


def audit(directory):
    out = Path(directory).resolve()
    opened = set()
    install_guard(out, opened)
    receipt = json.loads((out/'prerun.json').read_text())
    record = json.loads((out/'audit.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    assert receipt['protocol'] == summary['protocol'] == 'TP1'
    assert receipt['paths'] == list(PATH_TYPES) and receipt['weights'] == WEIGHTS.tolist()
    assert receipt['device'] == 'cpu'
    verify(receipt['hashes'])
    verify({str(out/name): value for name, value in record['artifact_sha256'].items()})
    train, offset, counts = read_train()
    queries, paths, direct, graph = prepare(train, offset, counts)
    del train
    np.testing.assert_array_equal(queries, np.load(out/'queries.npy', allow_pickle=False))
    assert graph == json.loads((out/'graph_audit.json').read_text())
    source, relation, candidates, prior = candidates_for(queries, direct, counts['drug'])
    np.testing.assert_array_equal(candidates, np.load(out/'candidates.npy', allow_pickle=False))
    np.testing.assert_array_equal(prior, np.load(out/'degree_scores.npy', allow_pickle=False))
    print(json.dumps(dict(stage='audit_graph_samples_candidates', passed=True)), flush=True)
    raw = np.load(out/'raw_features.npy', mmap_mode='r', allow_pickle=False)
    normalized = np.load(out/'normalized_features.npy', mmap_mode='r', allow_pickle=False)
    assert raw.shape == normalized.shape == (3, len(source), 501)
    assert raw.dtype == normalized.dtype == np.float32
    assert np.isfinite(raw).all() and (raw >= 0).all()
    rng = np.random.default_rng(3656)
    examples = list(rng.choice(len(source), min(128, len(source)), replace=False))
    direction = np.repeat([0, 1], len(queries))
    for rel in np.unique(relation):
        for d in (0, 1):
            examples.append(np.flatnonzero((relation == rel) & (direction == d))[0])
    examples = np.unique(examples)
    maximum_error = 0.
    for j, path in enumerate(paths):
        for row in examples:
            expected = direct_path(path, source[row], candidates[row])
            np.testing.assert_allclose(raw[j, row], expected, atol=1e-6, rtol=1e-6)
            maximum_error = max(maximum_error, float(np.max(np.abs(raw[j, row]-expected))))
        np.testing.assert_array_equal(normalized[j], normalize(np.log1p(raw[j])))
    print(json.dumps(dict(stage='audit_paths_normalization', examples=len(examples),
                          all_501_candidates=True, maximum_error=maximum_error, passed=True)), flush=True)
    fit = np.tile(queries[:, 4] == 0, 2)
    recipe = select_recipe(normalized, fit, relation, direction)
    assert recipe == json.loads((out/'recipe.json').read_text())
    result_ranks = report_arrays(normalized, raw, prior, recipe, relation, direction, ~fit)
    with np.load(out/'report_ranks.npz', allow_pickle=False) as saved:
        assert set(saved.files) == set(result_ranks)
        for name, values in result_ranks.items():
            np.testing.assert_array_equal(saved[name], values)
    expected_summary = summarize(queries, raw, result_ranks)
    for name, value in expected_summary.items():
        assert summary[name] == value, name
    assert sorted(opened) == summary['dataset_files_opened'] == record['dataset_files_opened'] == sorted(EXPECTED)
    verify(receipt['hashes'])
    result = dict(audit_passed=True, source_input_artifact_hashes_verified=True,
        grouped_pair_split_and_graph_reproduced=True, all_samples_and_candidates_exact=True,
        representative_direct_path_queries=len(examples), all_501_candidates=True,
        raw_feature_maximum_error=maximum_error, all_normalization_exact=True,
        fit_only_recipe_and_all_report_ranks_exact=True, summary_and_pair_cluster_interval_reproduced=True,
        valid_loaded=False, test_loaded=False, checkpoint_loaded=False, gpu_used=False)
    write_json(out/'endpoint_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    audit(parser.parse_args().directory)
