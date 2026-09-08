"""Synthetic H35F tests, no real checkpoints or BioKG split access."""

import copy
import io
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from biokg.branch_warmup import (
    assert_gradients, b_training_logits, configure_phase, direct_loss, effective_lr,
    phase_change_audit, training_loss,
)
from biokg.joint_operator import retained_loss
from biokg.mixed_operator import from_student, restore_mixed
from biokg.train_biokg_comp import teacher_logits
from biokg.train_joint_operator import assert_independent, parameter_hashes, teacher_hashes
from biokg.train_branch_warmup import optimizer_steps, probe_statistics, validation_only
from resonate import ResonatE


class BranchWarmupTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3515)
        torch.set_num_threads(2)
        self.original = ResonatE(20, 2, k=2, block=True, block_size=2)
        self.ck = dict(args=dict(k=2, block_size=2), model=copy.deepcopy(self.original.state_dict()))
        self.models = {name: from_student(self.ck, "single" if name == "single" else "dot", width=4)
                       for name in ("single", "joint", "warmup")}
        self.source, self.rel = torch.tensor([0, 1, 2]), torch.zeros(3, dtype=torch.long)
        self.pos, self.neg = torch.tensor([3, 4, 5]), torch.tensor([3, 4, 6, 8])
        self.batch = self.source, self.rel, self.pos, self.neg
        self.cand = torch.cat([self.pos[:, None], self.neg[None].expand(3, -1)], dim=1)
        self.teachers = [ResonatE(20, 2, k=2, block=True, block_size=2).eval().requires_grad_(False)
                         for _ in range(2)]
        self.target = teacher_logits(self.teachers, *self.batch)

    def test_initial_tensors_matched_not_shared(self):
        self.assertEqual(parameter_hashes(self.models["joint"]), parameter_hashes(self.models["warmup"]))
        assert_independent(self.models, self.teachers)

    def test_ungated_scores_and_gradient_parity(self):
        model = self.models["warmup"]
        other = copy.deepcopy(model)
        actual = b_training_logits(model, *self.batch)
        expected = other.candidate_outputs(self.source, self.rel, self.cand)[2]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        weights = torch.randn_like(actual)
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        for name, p in model.named_parameters():
            q = dict(other.named_parameters())[name]
            if name.startswith("a.") or name == "gate_logit":
                self.assertIsNone(p.grad)
                self.assertIsNone(q.grad)
            else:
                torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)

    def test_b_loss_does_not_evaluate_a_or_gate(self):
        model = self.models["warmup"]
        configure_phase(model, True)
        with patch.object(model.a, "training_outputs", side_effect=AssertionError("A called")):
            before = training_loss(model, self.batch, self.target, True)[0]
            with torch.no_grad():
                model.gate_logit.fill_(-90)
                model.a.E.mul_(2)
            after = training_loss(model, self.batch, self.target, True)[0]
        torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_direct_ce_kd_formula(self):
        logits = b_training_logits(self.models["warmup"], *self.batch)
        actual, parts = direct_loss(logits, self.target)
        ce = F.cross_entropy(logits, torch.zeros(3, dtype=torch.long))
        kd = 4 * F.kl_div(F.log_softmax(logits / 2, dim=1), F.softmax(self.target / 2, dim=1), reduction="batchmean")
        torch.testing.assert_close(actual, ce + kd, atol=0, rtol=0)
        self.assertEqual(parts["trajectory"].item(), 0)
        with self.assertRaises(ValueError):
            direct_loss(logits, self.target.clone().requires_grad_())

    def test_warm_phase_freezing_and_optimizer_state(self):
        model = self.models["warmup"]
        configure_phase(model, True)
        before = parameter_hashes(model)
        teacher_before = teacher_hashes(self.teachers)
        opt = torch.optim.Adam(model.optimizer_groups())
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            training_loss(model, self.batch, self.target, True)[0].backward()
            assert_gradients(model, True)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            opt.step()
        phase_change_audit(before, parameter_hashes(model), True)
        steps = optimizer_steps(model, opt)
        for name, count in steps.items():
            self.assertEqual(count, 0 if name.startswith("a.") or name == "gate_logit" else 3)
        self.assertEqual(teacher_before, teacher_hashes(self.teachers))

    def test_switch_retains_moments_and_uses_original_loss(self):
        model = self.models["warmup"]
        configure_phase(model, True)
        opt = torch.optim.Adam(model.optimizer_groups())
        training_loss(model, self.batch, self.target, True)[0].backward()
        opt.step()
        moments = opt.state[model.B]["exp_avg"].clone()
        steps = optimizer_steps(model, opt)
        configure_phase(model, False)
        torch.testing.assert_close(moments, opt.state[model.B]["exp_avg"], atol=0, rtol=0)
        self.assertEqual(steps, optimizer_steps(model, opt))
        before = parameter_hashes(model)
        loss, parts = training_loss(model, self.batch, self.target, False)
        reference, expected = retained_loss(model.training_outputs(*self.batch), self.target)
        torch.testing.assert_close(loss, reference, atol=0, rtol=0)
        for name in parts:
            torch.testing.assert_close(parts[name], expected[name], atol=0, rtol=0)
        loss.backward()
        assert_gradients(model, False)
        opt.step()
        phase_change_audit(before, parameter_hashes(model), False)
        self.assertEqual(optimizer_steps(model, opt)["a.E"], 1)
        self.assertEqual(optimizer_steps(model, opt)["gate_logit"], 1)
        self.assertEqual(optimizer_steps(model, opt)["B"], 2)

    def test_shared_schedule_no_restart_and_effective_lr(self):
        model = self.models["warmup"]
        configure_phase(model, True)
        opt = torch.optim.Adam(model.optimizer_groups())
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15000)
        for _ in range(10000):
            opt.step()
            schedule.step()
        rates = effective_lr(opt, True, True)
        self.assertEqual(rates["a"], 0)
        self.assertEqual(rates["gate"], 0)
        self.assertAlmostEqual(rates["b"], .00025, places=12)
        configure_phase(model, False)
        rates = effective_lr(opt, False, True)
        self.assertAlmostEqual(rates["a"], .000025, places=12)
        self.assertEqual(rates["b"], rates["gate"])
        self.assertEqual(len(opt.state), 0)

    def test_probe_branch_metrics_are_read_only(self):
        part = dict(head=np.array([0, 1, 2]), tail=np.array([3, 4, 5]), relation=np.array([0, 0, 0]),
                    head_type=np.array(["x"] * 3), tail_type=np.array(["x"] * 3),
                    head_neg=np.tile([1, 4, 6], (3, 1)), tail_neg=np.tile([2, 4, 6], (3, 1)))
        model = self.models["warmup"]
        configure_phase(model, True)
        before = parameter_hashes(model)
        rng = torch.random.get_rng_state()
        result = probe_statistics(model, part, {"x": 0}, 2)
        self.assertIn("combined_mrr", result)
        self.assertIn("a_mrr", result)
        self.assertIn("b_mrr", result)
        self.assertEqual(before, parameter_hashes(model))
        torch.testing.assert_close(rng, torch.random.get_rng_state(), atol=0, rtol=0)
        self.assertFalse(model.a.E.requires_grad)
        self.assertFalse(model.gate_logit.requires_grad)
        self.assertTrue(model.training)

    def test_restore_after_warmup(self):
        model = self.models["warmup"]
        configure_phase(model, True)
        opt = torch.optim.Adam(model.optimizer_groups())
        training_loss(model, self.batch, self.target, True)[0].backward()
        opt.step()
        stream = io.BytesIO()
        torch.save(dict(model_type="H35EMixedOperator", protocol="H35F", model=model.state_dict(),
                        mode="dot", width=4), stream)
        stream.seek(0)
        restored = restore_mixed(torch.load(stream, weights_only=False))
        self.assertEqual(parameter_hashes(model), parameter_hashes(restored))
        torch.testing.assert_close(model.candidate_outputs(self.source, self.rel, self.cand)[0],
                                   restored.candidate_outputs(self.source, self.rel, self.cand)[0], atol=0, rtol=0)

    def test_selective_loader_and_invalid_arm(self):
        with patch("biokg.train_branch_warmup.load", return_value="sentinel") as mocked:
            self.assertEqual(validation_only("root"), "sentinel")
            mocked.assert_called_once_with("root", include_test=False)
        with self.assertRaises(ValueError):
            configure_phase(self.models["single"], True)


if __name__ == "__main__":
    unittest.main()
