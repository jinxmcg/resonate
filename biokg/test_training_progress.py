"""Progress diagnostics are read-only with respect to optimization and RNG."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

class TrainingProgressTests(unittest.TestCase):
    def test_probed_and_unprobed_training_match_exactly(self):
        from biokg import train_biokg_comp as trainer
        train = dict(head=np.array([0, 0, 1]), relation=np.zeros(3, dtype=np.int64), tail=np.array([2, 3, 4]),
                     head_type=["x"] * 3, tail_type=["x"] * 3)
        valid = dict(head=np.array([0]), relation=np.array([0]), tail=np.array([6]), head_type=["x"], tail_type=["x"],
                     head_neg=np.array([[7, 8]]), tail_neg=np.array([[7, 8]]))
        with tempfile.TemporaryDirectory(prefix="biokg-progress-test-") as directory:
            for mining in ("none", "random"):
                states, rng_states = [], []
                for enabled in (False, True):
                    path = Path(directory) / f"{mining}_{enabled}.pt"
                    argv = ["trainer", "--mining-mode", mining, "--mining-count", "2",
                            "--mining-warmup", "0", "--mining-ramp", "1", "--steps", "3", "--eval", "valid",
                            "--shell", "sparse", "--k", "4", "--block-size", "4", "--batch", "2",
                            "--neg", "4", "--save", str(path)] + (["--probe-every", "1", "--probe-size", "1"] if enabled else [])
                    output = io.StringIO()
                    with patch("sys.argv", argv), redirect_stdout(output), \
                            patch.object(trainer, "load", return_value=({"train": train, "valid": valid},
                                {"x": 0}, 20, ["x"], {"x": 20})) as loader, \
                            patch.object(trainer, "eval_split", wraps=trainer.eval_split) as evaluator:
                        trainer.main()
                    self.assertFalse(loader.call_args.kwargs["include_test"])
                    self.assertEqual(evaluator.call_count, 4 if enabled else 1)
                    self.assertIs(evaluator.call_args.args[1], valid)
                    self.assertEqual("[probe@" in output.getvalue(), enabled)
                    self.assertIn("lr=", output.getvalue())
                    states.append(torch.load(path, map_location="cpu", weights_only=False)["model"])
                    rng_states.append(torch.get_rng_state().clone())
                self.assertEqual(states[0].keys(), states[1].keys())
                for key in states[0]:
                    torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
                self.assertTrue(torch.equal(*rng_states))


if __name__ == "__main__":
    unittest.main()
