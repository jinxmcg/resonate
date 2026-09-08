"""Synthetic H35E tests; no real BioKG data/checkpoints."""

import copy
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from biokg.joint_operator import retained_loss
from biokg.mixed_operator import MODES, from_student, restore_mixed
from biokg.train_biokg_comp import score_batch, teacher_logits
from biokg.train_joint_operator import assert_independent, parameter_hashes, teacher_hashes
from biokg.train_mixed_operator import evaluate, validation_only
from resonate import ResonatE


class MixedOperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3513)
        torch.set_num_threads(2)
        self.original = ResonatE(20, 2, k=2, block=True, block_size=2)
        self.ck = dict(args=dict(k=2, block_size=2), model=copy.deepcopy(self.original.state_dict()))
        self.models = {mode: from_student(self.ck, mode, width=4) for mode in MODES}
        self.source = torch.tensor([0, 1, 2])
        self.relation = torch.tensor([0, 0, 0])
        self.positive = torch.tensor([3, 4, 5])
        self.neg = torch.tensor([3, 4, 6, 8])
        self.cand = torch.cat([self.positive[:, None], self.neg[None, :].expand(3, -1)], dim=1)
        self.batch = (self.source, self.relation, self.positive, self.neg)
        self.teachers = [ResonatE(20, 2, k=2, block=True, block_size=2)
                         .eval().requires_grad_(False) for _ in range(2)]
        self.target = teacher_logits(self.teachers, *self.batch)

    def test_single_score_loss_gradient_parity(self):
        actual = self.models["single"].training_outputs(*self.batch)
        expected = score_batch(self.original, *self.batch)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
        la, _ = retained_loss(actual, self.target)
        lb, _ = retained_loss(expected, self.target)
        torch.testing.assert_close(la, lb)
        la.backward()
        lb.backward()
        for a, b in ((self.models["single"].a.E, self.original.E),
                     (self.models["single"].a.H_b, self.original.H),
                     (self.models["single"].a.log_tau, self.original.log_tau)):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-6)

    def test_matched_private_initialization_and_counts(self):
        self.assertEqual(parameter_hashes(self.models["dot"]), parameter_hashes(self.models["distance"]))
        self.assertEqual(self.models["dot"].n_params(), self.models["distance"].n_params())
        self.assertEqual(self.models["dot"].n_params() - self.models["single"].n_params(), 20 * 4 + 2 * 4 * 3 + 2 + 1)
        assert_independent(self.models, self.teachers)
        self.assertFalse(torch.allclose(self.models["dot"].B, self.original.E.real))
        torch.testing.assert_close(self.models["dot"].gate_logit.sigmoid(), torch.full((2,), .05))

    def test_candidate_and_training_parity_including_gradients(self):
        for mode in MODES:
            model = self.models[mode]
            other = copy.deepcopy(model)
            train = model.training_outputs(*self.batch)[0]
            evaluation = other.candidate_outputs(self.source, self.relation, self.cand)[0]
            torch.testing.assert_close(train, evaluation, atol=2e-6, rtol=2e-6)
            weights = torch.randn_like(train)
            (train * weights).sum().backward()
            (evaluation * weights).sum().backward()
            for (name, p), (other_name, q) in zip(model.named_parameters(), other.named_parameters()):
                self.assertEqual(name, other_name)
                torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-5)

    def test_l1_formula_and_non_dot_geometry(self):
        dot, distance = self.models["dot"], self.models["distance"]
        u = distance.source_b(self.source, self.relation)
        v = distance.target_b(self.cand, self.relation)
        expected = -(u[:, None] - v).abs().sum(-1)
        actual = distance.candidate_outputs(self.source, self.relation, self.cand)[2]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertFalse(torch.allclose(actual, dot.candidate_outputs(self.source, self.relation, self.cand)[2]))

    def test_gate_is_relation_dependent_and_nonzero(self):
        for mode in ("dot", "distance"):
            model = self.models[mode]
            rel = torch.tensor([0, 1, 0])
            score, a, b = model.candidate_outputs(self.source, rel, self.cand)
            torch.testing.assert_close(score, a + .05 * b, atol=1e-6, rtol=1e-6)
            with torch.no_grad():
                model.gate_logit[1] = 0
            changed = model.candidate_outputs(self.source, rel, self.cand)[0]
            torch.testing.assert_close(changed[[0, 2]], score[[0, 2]], atol=0, rtol=0)
            self.assertFalse(torch.allclose(changed[1], score[1]))

    def test_candidate_permutation_duplicates_and_set_independence(self):
        order = [3, 2, 0, 4, 1]
        for model in self.models.values():
            score = model.candidate_outputs(self.source, self.relation, self.cand)[0]
            perm = model.candidate_outputs(self.source, self.relation, self.cand[:, order])[0]
            torch.testing.assert_close(perm, score[:, order], atol=0, rtol=0)
            extra = torch.cat([self.cand, torch.zeros(3, 3, dtype=torch.long)], dim=1)
            padded = model.candidate_outputs(self.source, self.relation, extra)[0]
            torch.testing.assert_close(padded[:, :5], score, atol=1e-6, rtol=1e-6)
            self.assertEqual(score[0, 0], score[0, 1])

    def test_all_groups_update_teachers_frozen(self):
        teachers_before = teacher_hashes(self.teachers)
        for model in self.models.values():
            before = parameter_hashes(model)
            opt = torch.optim.Adam(model.optimizer_groups())
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                loss, _ = retained_loss(model.training_outputs(*self.batch), self.target)
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    and p.grad.norm() > 0 for p in model.parameters()))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
            after = parameter_hashes(model)
            self.assertTrue(all(after[name] != before[name] for name in after))
        self.assertEqual(teacher_hashes(self.teachers), teachers_before)
        self.assertTrue(all(p.grad is None and not p.requires_grad for t in self.teachers for p in t.parameters()))

    def test_trajectory_only_updates_a(self):
        for mode in ("dot", "distance"):
            model = self.models[mode]
            _, parts = retained_loss(model.training_outputs(*self.batch), self.target)
            parts["trajectory"].backward()
            self.assertIsNotNone(model.a.E.grad)
            self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if not name.startswith("a.")))

    def test_optimizer_groups_and_cosine(self):
        for mode, model in self.models.items():
            groups = model.optimizer_groups()
            params = [id(p) for group in groups for p in group["params"]]
            self.assertEqual(len(params), len(set(params)))
            self.assertEqual(set(params), {id(p) for p in model.parameters()})
            self.assertEqual([group["lr"] for group in groups], [.0001] if mode == "single" else [.0001, .001])
            opt = torch.optim.Adam(groups)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2)
            for _ in range(2):
                opt.step()
                sched.step()
            self.assertTrue(all(group["lr"] == 0 for group in opt.param_groups))

    def test_standalone_restore(self):
        for mode, model in self.models.items():
            stream = io.BytesIO()
            torch.save(dict(model_type="H35EMixedOperator", model=model.state_dict(), mode=mode, width=4), stream)
            stream.seek(0)
            restored = restore_mixed(torch.load(stream, weights_only=False))
            self.assertEqual(parameter_hashes(restored), parameter_hashes(model))
            self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
            torch.testing.assert_close(restored.candidate_outputs(self.source, self.relation, self.cand)[0],
                                       model.candidate_outputs(self.source, self.relation, self.cand)[0], atol=0, rtol=0)

    def test_probe_is_rng_neutral_and_frozen(self):
        part = dict(head=np.array([0, 1, 2]), tail=np.array([3, 4, 5]), relation=np.array([0, 0, 0]),
                    head_type=np.array(["x"] * 3), tail_type=np.array(["x"] * 3),
                    head_neg=np.tile(np.array([1, 4, 6]), (3, 1)), tail_neg=np.tile(np.array([2, 4, 6]), (3, 1)))
        for model in self.models.values():
            before = parameter_hashes(model)
            torch_before = torch.random.get_rng_state()
            numpy_before = np.random.get_state()
            metrics, record = evaluate(model, part, {"x": 0}, 2, diagnostics=True, chunk=2)
            self.assertEqual(metrics["queries"], 6)
            self.assertEqual(before, parameter_hashes(model))
            torch.testing.assert_close(torch_before, torch.random.get_rng_state(), atol=0, rtol=0)
            after = np.random.get_state()
            self.assertEqual(numpy_before[0], after[0])
            np.testing.assert_array_equal(numpy_before[1], after[1])
            self.assertEqual(numpy_before[2:], after[2:])
            self.assertTrue(model.training)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            np.testing.assert_allclose(record["rr"], 1 / record["rank"], rtol=0, atol=0)

    def test_selective_loader(self):
        with patch("biokg.train_mixed_operator.load", return_value="sentinel") as mocked:
            self.assertEqual(validation_only("root"), "sentinel")
            mocked.assert_called_once_with("root", include_test=False)

    def test_mixed_training_relations_rejected(self):
        with self.assertRaisesRegex(ValueError, "one directed relation"):
            self.models["distance"].training_outputs(self.source, torch.tensor([0, 1, 0]), self.positive, self.neg)


if __name__ == "__main__":
    unittest.main()
