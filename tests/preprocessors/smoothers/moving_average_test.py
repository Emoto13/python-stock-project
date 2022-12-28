import unittest
import numpy as np

from src.preprocessors.smoothers import MovingAverage


class TestMovingAverage(unittest.TestCase):
    def test_moving_average_works_correctly(self):
        data = np.array([10, 20, 30, 40, 50])
        moving_average = MovingAverage()
        scaled_data = moving_average.preprocess(data, window=2)
        expected = np.array([10., 15., 25., 35., 45.])
        np.testing.assert_array_equal(expected, scaled_data, "moving average should scale correctly")
