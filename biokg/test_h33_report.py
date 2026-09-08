"""Do not turn a partial or cherry-picked seed set into confirmation."""
import unittest

from biokg.summarize_h33 import aggregate_results


def rows(baseline, random):
    return {f"{arm}_s{s}": {"overall": {"mrr": score}}
            for arm, scores in (("baseline", baseline), ("random", random))
            for s, score in enumerate(scores)}


class H33ReportTests(unittest.TestCase):
    def test_paired_statistics_and_sample_std(self):
        got = aggregate_results(rows([.810, .811, .812], [.813, .815, .817]))
        for actual, expected in zip(got["paired_gains"], [.003, .004, .005]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(got["mean_paired_gain"], .004)
        self.assertAlmostEqual(got["sample_std_paired_gain"], .001)
        self.assertAlmostEqual(got["arms"]["baseline"]["sample_std"], .001)
        self.assertTrue(got["followup_gate"])

    def test_mixed_or_small_gains_do_not_pass(self):
        for random in ([.820, .820, .809], [.811, .811, .811], [.810, .820, .820]):
            self.assertFalse(aggregate_results(rows([.810] * 3, random))["followup_gate"])

    def test_missing_extra_and_invalid_results_fail(self):
        complete = rows([.81] * 3, [.82] * 3)
        missing = dict(complete)
        missing.pop("baseline_s2")
        extra = {**complete, "random_s3": {"overall": {"mrr": .99}}}
        for value in (missing, extra, rows([.81] * 3, [.82, float("nan"), .82]),
                      rows([.81] * 3, [.82, 1.1, .82])):
            with self.assertRaises(ValueError):
                aggregate_results(value)


if __name__ == "__main__":
    unittest.main()
