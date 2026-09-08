"""Synthetic tests only; no real BioKG files, weights or GPU needed."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from biokg.gradient_diagnostic import (
    check_dataset_path, gradient_gram, gram_metrics, inspect_batch, summarize, train_only,
)
from biokg.joint_operator import from_student
from biokg.train_biokg_comp import teacher_logits
from biokg.train_dual_operator import TrainStream, tensor_digest
from resonate import ResonatE


class GradientDiagnosticTests(unittest.TestCase):
    def test_complex_real_coordinate_gram(self):
        p = torch.nn.Parameter(torch.zeros(2, dtype=torch.complex64))
        vectors = [torch.tensor([1+2j, -3+4j]), torch.tensor([2-1j, 5+3j]), None]
        actual = gradient_gram([(x,) for x in vectors], (p,))[0]
        reference = np.array([[1, 2, -3, 4], [2, -1, 5, 3], [0, 0, 0, 0]], np.float64)
        np.testing.assert_allclose(actual, reference @ reference.T)
        self.assertIsNone(gram_metrics(actual)["cos_ce_trajectory"])

    def test_conflict_projection_and_weight(self):
        vectors = np.array([[1., 0.], [0., 1.], [-.1, -.1]])
        result = gram_metrics(vectors @ vectors.T)
        self.assertAlmostEqual(result["cos_ce_kd_trajectory"], -1)
        self.assertAlmostEqual(result["trajectory_over_ce_kd"], .1)
        self.assertAlmostEqual(result["ce_kd_descent_projection_ratio"], .9)
        self.assertAlmostEqual(result["norm_trajectory"], np.sqrt(.02))

    def test_zero_and_nonfinite(self):
        result = gram_metrics(np.zeros((3, 3)))
        self.assertIsNone(result["cos_ce_kd_trajectory"])
        self.assertIsNone(result["ce_kd_descent_projection_ratio"])
        with self.assertRaises(ValueError):
            gram_metrics(np.full((3, 3), np.nan))

    def test_diagnostic_retains_state_and_recomposes_loss(self):
        torch.manual_seed(3601)
        base = ResonatE(20, 4, k=2, block=True, block_size=2)
        model = from_student(dict(model=base.state_dict(), args={}), "single")
        teachers = [base.eval().requires_grad_(False)]
        batch = (torch.tensor([0, 1, 2]), torch.tensor([0, 0, 0]),
                 torch.tensor([3, 4, 5]), torch.tensor([3, 6, 7, 8]))
        before = tensor_digest(model.state_dict()), tensor_digest(base.state_dict())
        target = teacher_logits(teachers, *batch)
        result = inspect_batch(model, batch, target, verify_sum=True)
        self.assertEqual(before, (tensor_digest(model.state_dict()), tensor_digest(base.state_dict())))
        self.assertTrue(all(p.grad is None for m in (model, base) for p in m.parameters()))
        self.assertEqual(result["groups"]["log_temperature"]["norm_trajectory"], 0)
        self.assertIsNone(result["groups"]["log_temperature"]["cos_ce_trajectory"])
        record = dict(arm="a", direction="head", family="x-y", **result)
        summary = summarize([record])
        self.assertEqual(summary["a"]["overall"]["batches"], 1)
        self.assertEqual(summary["a"]["overall"]["groups"]["log_temperature"]
                         ["cos_ce_trajectory"]["count"], 0)

    def test_dataset_guard(self):
        root = Path("/tmp/diagnostic_guard")
        self.assertEqual(check_dataset_path(root / "split/random/train.pt", root), "split/random/train.pt")
        self.assertIsNone(check_dataset_path("/tmp/unrelated_file", root))
        for relative in ("split/random/valid.pt", "split/random/test.pt", "processed/data_processed"):
            with self.assertRaises(PermissionError):
                check_dataset_path(root / relative, root)

    def test_train_only_opens_one_split(self):
        import gzip
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            with gzip.open(root / "raw/num-node-dict.csv.gz", "wt") as handle:
                handle.write("z,a\n4,3\n")
            fake = {key: np.array([]) for key in ("head", "tail", "relation", "head_type", "tail_type")}
            with patch("biokg.gradient_diagnostic.torch.load", return_value=fake) as loader:
                train, offset, counts = train_only(root)
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(loader.call_args.args[0], root / "split/random/train.pt")
            self.assertIs(train, fake)
            self.assertEqual(offset, {"a": 0, "z": 3})
            self.assertEqual(counts, {"a": 3, "z": 4})

    def test_stream_is_reproducible_and_typed(self):
        train = dict(head=np.array([0, 1, 2]), tail=np.array([0, 1, 3]),
                     relation=np.array([0, 0, 0]), head_type=np.array(["a"] * 3),
                     tail_type=np.array(["b"] * 3))
        streams = [TrainStream(train, {"a": 0, "b": 3}, {"a": 3, "b": 4}, seed=3601) for _ in range(2)]
        directions = set()
        for _ in range(20):
            batch, reference = [s.sample(8, 12) for s in streams]
            for actual, expected in zip(batch, reference):
                np.testing.assert_array_equal(actual, expected)
            source, positive, relation, negative = batch
            directions.add(relation)
            lo, hi = (3, 7) if relation == 0 else (0, 3)
            self.assertTrue(((negative >= lo) & (negative < hi)).all())
            self.assertTrue(((positive >= lo) & (positive < hi)).all())
        self.assertEqual(directions, {0, 1})
        self.assertEqual(streams[0].digest.hexdigest(), streams[1].digest.hexdigest())


if __name__ == "__main__":
    unittest.main()
