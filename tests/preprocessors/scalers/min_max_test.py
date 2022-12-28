import unittest
import numpy as np

from src.preprocessors.scalers.min_max import MinMax


class TestMinMax(unittest.TestCase):
    def test_min_max_works_correctly(self):
        data = np.array([10, 20, 30, 40, 50])
        min_max = MinMax()
        scaled_data = min_max.preprocess(data)
        expected = np.array([0, 0.25, 0.5, 0.75, 1])
        np.testing.assert_array_equal(expected, scaled_data, "min max should scale correctly")
