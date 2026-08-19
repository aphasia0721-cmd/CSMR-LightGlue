from __future__ import annotations

import unittest

import numpy as np

from csmr.metrics import (
    error_auc,
    paired_bootstrap_auc_difference,
    paired_bootstrap_auc_differences,
    summarize,
)


class MetricsTests(unittest.TestCase):
    def test_error_auc_known_values(self) -> None:
        auc = error_auc([1.0, 2.0, 3.0])
        self.assertAlmostEqual(auc["auc@5"], 0.70)
        self.assertAlmostEqual(auc["auc@10"], 0.85)
        self.assertAlmostEqual(auc["auc@20"], 0.925)

    def test_error_auc_handles_failed_pose(self) -> None:
        auc = error_auc([1.0, np.inf], thresholds=(5,))
        self.assertTrue(np.isfinite(auc["auc@5"]))
        self.assertGreaterEqual(auc["auc@5"], 0.0)
        self.assertLessEqual(auc["auc@5"], 1.0)

    def test_empty_summary(self) -> None:
        summary = summarize([])
        self.assertEqual(summary["pairs"], 0)
        self.assertEqual(summary["mean_matches"], 0.0)
        self.assertEqual(summary["failed_pairs"], 0)

    def test_paired_bootstrap_is_deterministic(self) -> None:
        baseline = np.asarray([4.0, 7.0, 12.0, 18.0])
        method = np.asarray([3.0, 6.0, 10.0, 16.0])
        first = paired_bootstrap_auc_difference(baseline, method, samples=50, seed=7)
        second = paired_bootstrap_auc_difference(baseline, method, samples=50, seed=7)
        self.assertEqual(first, second)

    def test_paired_bootstrap_reports_all_thresholds(self) -> None:
        baseline = np.asarray([4.0, 7.0, 12.0, 18.0])
        method = np.asarray([3.0, 6.0, 10.0, 16.0])
        result = paired_bootstrap_auc_differences(baseline, method, samples=20, seed=7)
        self.assertEqual(set(result), {"auc@5", "auc@10", "auc@20"})
        self.assertEqual({value["threshold"] for value in result.values()}, {5, 10, 20})


if __name__ == "__main__":
    unittest.main()
