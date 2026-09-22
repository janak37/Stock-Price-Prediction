import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import build_confusion_matrix, detect_saturation_point


class MovementAnalysisTests(unittest.TestCase):
    def test_confusion_matrix_counts_rise_and_fall(self):
        actual = ["rise", "fall", "flat", "rise"]
        predicted = ["rise", "rise", "flat", "fall"]
        matrix = build_confusion_matrix(actual, predicted)
        self.assertEqual(matrix["labels"], ["Rise", "Fall", "Flat"])
        self.assertEqual(matrix["matrix"][0][0], 1)
        self.assertEqual(matrix["matrix"][1][0], 1)
        self.assertEqual(matrix["matrix"][2][2], 1)

    def test_saturation_point_detects_stable_window(self):
        prices = [100, 103, 106, 109, 111, 111.2, 111.3, 111.2, 111.1]
        result = detect_saturation_point(prices)
        self.assertTrue(result["is_saturated"])
        self.assertGreaterEqual(result["saturation_index"], 0)
        self.assertIn(result["trend"], {"Rising", "Falling", "Stable"})


if __name__ == "__main__":
    unittest.main()
