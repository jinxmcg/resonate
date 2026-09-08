"""H35C synthetic correctness tests; never load BioKG splits/checkpoints."""

import copy
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from biokg.joint_operator import from_student, retained_loss, restore_joint
from biokg.train_biokg_comp import score_batch, teacher_logits
from biokg.train_dual_operator import evaluate, tensor_digest
from biokg.train_joint_operator import (
    assert_independent, gradient_norms, parameter_hashes, teacher_hashes,
    validation_only,
)
from resonate import ResonatE


class JointOperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3510)
        torch.set_num_threads(2)
        self.original = ResonatE(12, 2, k=2, block=True, block_size=2)
        self.checkpoint = dict(args=dict(k=2, block_size=2),
                               model=copy.deepcopy(self.original.state_dict()))
        self.teachers = [ResonatE(12, 2, k=2, block=True, block_size=2)
                         .eval().requires_grad_(False) for _ in range(3)]
        self.source = torch.tensor([0, 1, 2])
        self.relation = torch.tensor([0, 1, 0])
        self.positive = torch.tensor([3, 4, 5])
        self.negatives = torch.tensor([0, 3, 6, 7, 9])
        self.batch = (self.source, self.relation, self.positive, self.negatives)
        self.target = teacher_logits(self.teachers, *self.batch)

    def test_single_original_scores_loss_and_gradients(self):
        joint = from_student(self.checkpoint, "single")
        outputs = joint.training_outputs(*self.batch)
        original_outputs = score_batch(self.original, *self.batch)
        for actual, expected in zip(outputs, original_outputs):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        logits, q, ep = original_outputs
        ce = F.cross_entropy(logits, torch.zeros(3, dtype=torch.long))
        kd = 4 * F.kl_div(F.log_softmax(logits / 2, dim=1),
                          F.softmax(self.target / 2, dim=1), reduction="batchmean")
        trajectory = (q - ep).abs().pow(2).sum(-1).mean()
        loss, parts = retained_loss(outputs, self.target)
        expected_loss = ce + kd + .1 * trajectory
        for actual, expected in ((parts["ce"], ce), (parts["kd_t_squared"], kd),
                                 (parts["trajectory"], trajectory), (loss, expected_loss)):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)
        loss.backward()
        expected_loss.backward()
        for current, original in ((joint.E, self.original.E), (joint.H_b, self.original.H),
                                  (joint.log_tau, self.original.log_tau)):
            torch.testing.assert_close(current.grad, original.grad, atol=2e-6, rtol=1e-5)
        self.assertEqual(joint.n_params(True), self.original.n_params())

    def test_teacher_mean_raw_logits_and_detachment(self):
        raw = [score_batch(teacher, *self.batch)[0] for teacher in self.teachers]
        torch.testing.assert_close(self.target, sum(raw) / len(raw), atol=0, rtol=0)
        self.assertFalse(self.target.requires_grad)
        self.assertFalse(torch.allclose(F.softmax(self.target / 2, dim=1),
                                        torch.stack([F.softmax(x / 2, dim=1) for x in raw]).mean(0)))
        with self.assertRaisesRegex(ValueError, "detached"):
            retained_loss(from_student(self.checkpoint, "single").training_outputs(*self.batch),
                          self.target.clone().requires_grad_())

    def test_identical_branches_reduce_to_single(self):
        single = from_student(self.checkpoint, "single")
        dual = from_student(self.checkpoint, "or", perturbation=0)
        one, one_parts = retained_loss(single.training_outputs(*self.batch), self.target)
        two, two_parts = retained_loss(dual.training_outputs(*self.batch), self.target)
        torch.testing.assert_close(one, two, atol=2e-6, rtol=1e-6)
        for name in one_parts:
            torch.testing.assert_close(one_parts[name], two_parts[name], atol=2e-6, rtol=1e-6)
        one.backward()
        two.backward()
        torch.testing.assert_close(single.E.grad, dual.E.grad, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(single.H_b.grad, dual.H_a.grad + dual.H_b.grad, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(single.log_tau.grad, dual.log_tau.grad, atol=2e-6, rtol=1e-5)
        self.assertEqual(dual.n_params(True), single.n_params() + 2 * single.H_b.numel())

    def test_trajectory_anchors_a_not_b(self):
        dual = from_student(self.checkpoint, "or")
        _, parts = retained_loss(dual.training_outputs(*self.batch), self.target)
        parts["trajectory"].backward()
        self.assertGreater(float(dual.E.grad.norm()), 0)
        self.assertGreater(float(dual.H_a.grad.norm()), 0)
        self.assertIsNone(dual.H_b.grad)
        self.assertIsNone(dual.log_tau.grad)

    def test_all_groups_update_and_teachers_do_not(self):
        teachers_before = teacher_hashes(self.teachers)
        for mode in ("single", "or"):
            model = from_student(self.checkpoint, mode)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            before = parameter_hashes(model)
            opt = torch.optim.Adam(model.parameters(), lr=.0001)
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                target = teacher_logits(self.teachers, *self.batch)
                loss, _ = retained_loss(model.training_outputs(*self.batch), target)
                loss.backward()
                self.assertTrue(all(value is not None and value > 0 and np.isfinite(value)
                                    for value in gradient_norms(model).values()))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
            after = parameter_hashes(model)
            self.assertTrue(all(before[name] != after[name] for name in before))
        self.assertEqual(teacher_hashes(self.teachers), teachers_before)
        self.assertTrue(all(p.grad is None and not p.requires_grad
                            for teacher in self.teachers for p in teacher.parameters()))

    def test_independent_students_and_optimizers(self):
        students = {mode: from_student(self.checkpoint, mode) for mode in ("single", "or")}
        assert_independent(students, self.teachers)
        opts = {mode: torch.optim.Adam(model.parameters(), lr=.0001)
                for mode, model in students.items()}
        before = parameter_hashes(students["or"])
        loss, _ = retained_loss(students["single"].training_outputs(*self.batch), self.target)
        loss.backward()
        opts["single"].step()
        self.assertEqual(before, parameter_hashes(students["or"]))
        self.assertTrue(all(p.grad is None for p in students["or"].parameters()))
        self.assertFalse(opts["or"].state)
        self.assertTrue(opts["single"].state)
        students["or"].E = students["single"].E
        with self.assertRaisesRegex(AssertionError, "shared"):
            assert_independent(students, self.teachers)

    def test_standalone_roundtrip_and_candidate_permutation(self):
        candidates = torch.tensor([[1, 2, 4, 2], [3, 9, 2, 0], [0, 4, 5, 6]])
        order = [3, 2, 0, 1]
        for mode in ("single", "or"):
            model = from_student(self.checkpoint, mode)
            opt = torch.optim.Adam(model.parameters(), lr=.0001)
            retained_loss(model.training_outputs(*self.batch), self.target)[0].backward()
            opt.step()
            expected = model.candidate_scores(self.source, self.relation, candidates)[0]
            buffer = io.BytesIO()
            torch.save(dict(model_type="H35CJointOperator", model=model.state_dict(), mode=mode), buffer)
            buffer.seek(0)
            loaded = restore_joint(torch.load(buffer, weights_only=False))
            actual = loaded.candidate_scores(self.source, self.relation, candidates)[0]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            shuffled = loaded.candidate_scores(self.source, self.relation, candidates[:, order])[0]
            torch.testing.assert_close(shuffled, actual[:, order], atol=1e-6, rtol=1e-6)
            self.assertEqual(actual[0, 1], actual[0, 3])
            self.assertFalse(loaded.training)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))

    def test_probe_preserves_parameters_gradients_and_rng(self):
        model = from_student(self.checkpoint, "or")
        model.train()
        retained_loss(model.training_outputs(*self.batch), self.target)[0].backward()
        before = tensor_digest(model.state_dict())
        before_grads = tensor_digest({name: p.grad for name, p in model.named_parameters()})
        rng = torch.get_rng_state().clone()
        part = dict(head=np.array([0, 1]), tail=np.array([4, 5]),
                    relation=np.array([0, 0]), head_type=np.array(["x", "x"]),
                    tail_type=np.array(["x", "x"]),
                    head_neg=np.tile(np.arange(500) % 12, (2, 1)),
                    tail_neg=np.tile(np.arange(500) % 12, (2, 1)))
        metrics, _ = evaluate(model, part, {"x": 0}, 2, chunk=1)
        self.assertEqual(metrics["queries"], 4)
        self.assertEqual(before, tensor_digest(model.state_dict()))
        self.assertEqual(before_grads, tensor_digest({name: p.grad for name, p in model.named_parameters()}))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(model.training)

    def test_unsupported_configurations_rejected(self):
        with self.assertRaises(ValueError):
            from_student(self.checkpoint, "mean")
        bad = copy.deepcopy(self.checkpoint)
        bad["args"]["comp"] = True
        with self.assertRaises(ValueError):
            from_student(bad, "single")
        with self.assertRaises(ValueError):
            restore_joint(dict(model_type="H35DualOperator"))
        outputs = from_student(self.checkpoint, "single").training_outputs(*self.batch)
        with self.assertRaises(ValueError):
            retained_loss(outputs, self.target, temperature=0)
        with self.assertRaises(ValueError):
            retained_loss(outputs, self.target[:, :2])

    def test_validation_only_loader(self):
        with patch("biokg.train_dual_operator.load", return_value="safe") as mocked:
            self.assertEqual(validation_only("root"), "safe")
            mocked.assert_called_once_with("root", include_test=False)


if __name__ == "__main__":
    unittest.main()
