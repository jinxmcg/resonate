"""Train-only direction policy and default/auxiliary integration checks."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from biokg.direction_sampling import train_fanout_policy


class DirectionSamplingTests(unittest.TestCase):
    def test_exact_edge_weighted_fanout_and_moderate_skew(self):
        got = train_fanout_policy(np.array([0, 0, 1]), np.zeros(3, dtype=int),
                                  np.array([2, 3, 4]), 8, 1)
        row = got["relations"][0]
        self.assertAlmostEqual(row["tail_fanout"], 5 / 3)
        self.assertEqual(row["head_fanout"], 1.)
        self.assertAlmostEqual(row["p_tail"], np.sqrt(5 / 3) / (1 + np.sqrt(5 / 3)))

    def test_direction_convention_caps_and_reverse_equivariance(self):
        h, r, t = np.zeros(25, dtype=int), np.zeros(25, dtype=int), np.arange(1, 26)
        a = train_fanout_policy(h, r, t, 26, 1)["relations"][0]
        b = train_fanout_policy(t, r, h, 26, 1)["relations"][0]
        self.assertEqual(a["p_tail"], 2 / 3)  # One head, many tails -> more tail prediction.
        self.assertEqual(b["p_tail"], 1 / 3)
        self.assertAlmostEqual(a["p_tail"], b["p_head"])
        self.assertEqual(a["tail_fanout"], b["head_fanout"])

    def test_duplicates_empty_relations_and_symmetric_graph(self):
        h, r, t = np.array([0, 0, 1]), np.zeros(3, dtype=int), np.array([2, 3, 4])
        a = train_fanout_policy(h, r, t, 8, 2)
        b = train_fanout_policy(h[[0, 0, 0, 1, 2]], r[[0, 0, 0, 1, 2]], t[[0, 0, 0, 1, 2]], 8, 2)
        self.assertEqual(a["relations"], b["relations"])
        self.assertNotEqual(a["train_sha256"], b["train_sha256"])
        self.assertEqual(a["tail_probabilities"][1], .5)
        sym = train_fanout_policy(np.r_[h, t], np.r_[r, r], np.r_[t, h], 8, 1)
        self.assertEqual(sym["tail_probabilities"], [.5])

    def test_no_rng_draws_and_same_graph_digest_as_positive_index(self):
        from biokg.train_positive_filter import TrainPositiveFilter
        h, r, t = np.array([0, 0, 1]), np.zeros(3, dtype=int), np.array([2, 3, 4])
        np.random.seed(17)
        before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        got = train_fanout_policy(h, r, t, 8, 1)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertEqual(got["train_sha256"], TrainPositiveFilter(h, r, t, 8, 1).train_sha256)

    def test_invalid_inputs(self):
        for h, r, t, n, nr in [([0.5], [0], [1], 2, 1), ([0], [0, 0], [1], 2, 1),
                                ([-1], [0], [1], 2, 1), ([0], [1], [1], 2, 1),
                                ([0], [0], [2], 2, 1), ([0], [0], [1], 0, 1)]:
            with self.subTest(h=h, r=r, t=t), self.assertRaises(ValueError):
                train_fanout_policy(np.array(h), np.array(r), np.array(t), n, nr)

    def test_all_four_training_arms_train_only_policy_and_unchanged_evaluation(self):
        from biokg import train_biokg_comp as trainer
        train = dict(head=np.array([0, 0, 1]), relation=np.zeros(3, dtype=np.int64), tail=np.array([2, 3, 4]),
                     head_type=["x"] * 3, tail_type=["x"] * 3)
        valid = dict(head=np.array([0]), relation=np.array([0]), tail=np.array([6]), head_type=["x"], tail_type=["x"])
        with tempfile.TemporaryDirectory(prefix="biokg-direction-test-") as directory:
            models = []
            for mode in ("uniform", "train-fanout"):
                for mining in ("none", "random"):
                    path = Path(directory) / f"{mode}_{mining}.pt"
                    argv = ["trainer", "--direction-sampling", mode, "--mining-mode", mining,
                            "--mining-count", "2", "--mining-warmup", "0", "--mining-ramp", "1",
                            "--steps", "3", "--eval", "valid", "--shell", "sparse", "--k", "4",
                            "--block-size", "4", "--batch", "2", "--neg", "4", "--save", str(path)]
                    with patch("sys.argv", argv), redirect_stdout(io.StringIO()), \
                            patch.object(trainer, "load", return_value=({"train": train, "valid": valid},
                                {"x": 0}, 20, ["x"], {"x": 20})) as loader, \
                            patch.object(trainer, "train_fanout_policy", wraps=train_fanout_policy) as builder, \
                            patch.object(trainer, "eval_split") as evaluator:
                        trainer.main()
                    self.assertFalse(loader.call_args.kwargs["include_test"])
                    self.assertIs(evaluator.call_args.args[1], valid)
                    ck = torch.load(path, map_location="cpu", weights_only=False)
                    models.append(ck["model"])
                    self.assertEqual(ck["direction_sampling"]["mode"], mode)
                    self.assertFalse(ck["train_positive_filter"]["enabled"])
                    if mode == "uniform":
                        builder.assert_not_called()
                    else:
                        builder.assert_called_once()
                        for supplied, key in zip(builder.call_args.args[:3], ("head", "relation", "tail")):
                            np.testing.assert_array_equal(supplied, train[key])
                        self.assertEqual(np.asarray(ck["direction_sampling"]["batch_counts_tail_head"]).sum(), 3)
                        if mining == "random":
                            self.assertEqual(ck["direction_sampling"]["train_sha256"], ck["negative_mining"]["train_sha256"])
            for model in models:
                self.assertEqual(model.keys(), models[0].keys())
                self.assertTrue(all(torch.isfinite(x).all() for x in model.values()))


if __name__ == "__main__":
    unittest.main()
