"""Synthetic CFKD1 tests; no real BioKG data or checkpoint access."""

import copy
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from biokg.candidate_focus import FOCUS_WEIGHT, candidate_mask, conditional_kd, focused_loss
from biokg.joint_operator import retained_loss
from biokg.mixed_operator import from_student, restore_mixed
from biokg.train_biokg_comp import teacher_logits
from biokg.train_candidate_focus import check_data_path, make_checkpoint, optimizer_steps, train_valid
from biokg.train_joint_operator import assert_independent, parameter_hashes, teacher_hashes
from resonate import ResonatE


class CandidateFocusTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3603)
        torch.set_num_threads(2)

    def test_mask_union_cutoff_ties_and_known_positive(self):
        s = torch.tensor([[0., 7., 7., 7., 1., -2.]])
        t = torch.tensor([[0., 1., 1., 1., 9., -2.]])
        mask = candidate_mask(s, t, torch.tensor([5]), k=2)
        np.testing.assert_array_equal(mask, [[False, True, True, True, True, True]])
        self.assertFalse(mask.requires_grad)

    def test_conditional_loss_and_gradient_match_explicit_slices(self):
        s = torch.randn(3, 11, dtype=torch.float64, requires_grad=True)
        t = torch.randn(3, 11, dtype=torch.float64)
        positive = torch.tensor([0, 2, 4])
        loss, mask = conditional_kd(s, t, positive, k=2)
        reference = sum(4 * F.kl_div((s[i, mask[i]] / 2).log_softmax(0),
                                     (t[i, mask[i]] / 2).softmax(0), reduction="sum") for i in range(3)) / 3
        torch.testing.assert_close(loss, reference, atol=1e-12, rtol=1e-12)
        actual = torch.autograd.grad(loss, s, retain_graph=True)[0]
        expected = torch.autograd.grad(reference, s)[0]
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.all(actual[~mask] == 0))

    def test_permutation_loss_gradient_and_positive_remapping(self):
        s = torch.randn(4, 13, requires_grad=True)
        t = torch.randn(4, 13)
        pos = torch.tensor([0, 2, 4, 6])
        order = torch.randperm(13)
        perm = s.detach()[:, order].clone().requires_grad_(True)
        one, mask = conditional_kd(s, t, pos, k=3)
        two, other_mask = conditional_kd(perm, t[:, order], torch.argsort(order)[pos], k=3)
        torch.testing.assert_close(two, one, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(other_mask, mask[:, order], atol=0, rtol=0)
        ga = torch.autograd.grad(one, s)[0]
        gb = torch.autograd.grad(two, perm)[0]
        torch.testing.assert_close(gb, ga[:, order], atol=2e-6, rtol=2e-6)

    def test_full_support_equals_original_kd(self):
        s, t = torch.randn(4, 9), torch.randn(4, 9)
        loss, mask = conditional_kd(s, t, torch.zeros(4, dtype=torch.long), k=9)
        self.assertTrue(mask.all())
        expected = 4 * F.kl_div((s / 2).log_softmax(1), (t / 2).softmax(1), reduction="batchmean")
        torch.testing.assert_close(loss, expected, atol=2e-6, rtol=2e-6)

    def test_all_ties_are_finite_and_included(self):
        s = torch.zeros(3, 10, requires_grad=True)
        t = torch.ones(3, 10)
        loss, mask = conditional_kd(s, t, torch.tensor([0, 1, 2]), k=2)
        self.assertTrue(mask.all())
        loss.backward()
        self.assertTrue(torch.isfinite(s.grad).all())
        torch.testing.assert_close(s.grad, torch.zeros_like(s), atol=1e-6, rtol=0)
        self.assertAlmostEqual(float(loss), 0., places=6)

    def test_rejects_teacher_gradients_and_invalid_arguments(self):
        s, t = torch.randn(2, 6), torch.randn(2, 6)
        pos = torch.zeros(2, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "detached"):
            conditional_kd(s, t.requires_grad_(True), pos, k=2)
        t.requires_grad_(False)
        for k in (0, 7):
            with self.assertRaises(ValueError):
                candidate_mask(s, t, pos, k=k)
        with self.assertRaises(ValueError):
            conditional_kd(s, t, pos, k=2, temperature=0)

    def fixture(self):
        base = ResonatE(96, 4, k=2, block=True, block_size=2)
        ck = dict(model=copy.deepcopy(base.state_dict()), args={})
        students = {arm: from_student(ck, "single") for arm in ("control", "focused")}
        teachers = [base.eval().requires_grad_(False)]
        batch = (torch.tensor([0, 1, 2, 3]), torch.zeros(4, dtype=torch.long),
                 torch.tensor([4, 5, 6, 7]), torch.arange(64))
        return students, teachers, batch

    def test_control_is_exact_original_loss_and_no_conditional_call(self):
        students, teachers, batch = self.fixture()
        a, b = students.values()
        target = teacher_logits(teachers, *batch)
        with patch("biokg.candidate_focus.conditional_kd", side_effect=AssertionError("Unexpected focus")):
            actual, parts = focused_loss(a.training_outputs(*batch), target, False)
        expected, _ = retained_loss(b.training_outputs(*batch), target)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.backward()
        expected.backward()
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
        self.assertEqual(parts["focused_kd_t_squared"].item(), 0)

    def test_added_loss_is_exactly_declared_weight(self):
        students, teachers, batch = self.fixture()
        target = teacher_logits(teachers, *batch)
        outputs = students["focused"].training_outputs(*batch)
        actual, parts = focused_loss(outputs, target, True)
        base, original_parts = retained_loss(outputs, target)
        conditional, _ = conditional_kd(outputs[0], target, torch.zeros(4, dtype=torch.long))
        torch.testing.assert_close(actual, base + FOCUS_WEIGHT * conditional, atol=0, rtol=0)
        for key in original_parts:
            torch.testing.assert_close(parts[key], original_parts[key], atol=0, rtol=0)
        self.assertGreaterEqual(float(parts["selected_candidates"]), 32)
        self.assertLessEqual(float(parts["selected_teacher_mass"]), 1.000001)

    def test_both_students_update_teachers_frozen_and_standalone_restore(self):
        students, teachers, batch = self.fixture()
        assert_independent(students, teachers)
        before = {arm: parameter_hashes(model) for arm, model in students.items()}
        self.assertEqual(before["control"], before["focused"])
        teacher_before = teacher_hashes(teachers)
        for arm, model in students.items():
            opt = torch.optim.Adam(model.parameters(), lr=1e-4)
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                target = teacher_logits(teachers, *batch)
                loss, _ = focused_loss(model.training_outputs(*batch), target, arm == "focused")
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                opt.step()
            final = parameter_hashes(model)
            self.assertTrue(all(before[arm][name] != final[name] for name in final))
            self.assertEqual(set(optimizer_steps(model, opt).values()), {3})
            receipt = dict(student_sha256="synthetic", teacher_sha256={})
            stream = io.BytesIO()
            torch.save(make_checkpoint(model, arm, {}, 4, receipt), stream)
            stream.seek(0)
            restored = restore_mixed(torch.load(stream, weights_only=False))
            self.assertEqual(parameter_hashes(restored), final)
            self.assertEqual(restored.n_params(), students["control"].n_params())
        self.assertEqual(teacher_before, teacher_hashes(teachers))
        self.assertTrue(all(p.grad is None and not p.requires_grad for t in teachers for p in t.parameters()))

    def test_dataset_guard_excludes_test_and_processed_graph(self):
        root = Path("/tmp/cfkd1_guard")
        for name in ("train", "valid"):
            self.assertEqual(check_data_path(root / f"split/random/{name}.pt", root), f"split/random/{name}.pt")
        for name in ("split/random/test.pt", "processed/data_processed", "mapping/relidx2relname.csv.gz"):
            with self.assertRaises(PermissionError):
                check_data_path(root / name, root)

    def test_loader_only_requests_valid_after_train_only_helper(self):
        fake_train, fake_valid, offset, counts = {}, {}, {"x": 0}, {"x": 2}
        with patch("biokg.train_candidate_focus.train_only", return_value=(fake_train, offset, counts)) as train_loader:
            with patch("biokg.train_candidate_focus.torch.load", return_value=fake_valid) as loader:
                result = train_valid("/tmp/cfkd1_loader")
        self.assertEqual(result, (fake_train, fake_valid, offset, counts))
        train_loader.assert_called_once_with("/tmp/cfkd1_loader")
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(loader.call_args.args[0], Path("/tmp/cfkd1_loader/split/random/valid.pt"))


if __name__ == "__main__":
    unittest.main()
