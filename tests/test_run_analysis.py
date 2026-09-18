import unittest

from analyze_runs import iou, percentile


class RunAnalysisTests(unittest.TestCase):
    def test_percentile_and_iou(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5], .9), 5)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 1, 1), (2, 2, 3, 3)), 0.0)


if __name__ == "__main__":
    unittest.main()
