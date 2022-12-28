import unittest

from src.preprocessors.smoothers import SmootherFactory, MovingAverage


class TestSmootherFactory(unittest.TestCase):
    def test_smoother_factory_works_correctly(self):
        test_cases = [
            {
                "name": "moving_average",
                "class": MovingAverage
            },
        ]

        for test_case in test_cases:
            scaler = SmootherFactory.create_smoother(test_case["name"])
            self.assertIsInstance(scaler, test_case["class"], "smoother should be of correct type")

    def test_smoother_factory_throws_error_on_invalid_input(self):
        non_existent_scaler_name = "non_existent"

        def test_function():
            _ = SmootherFactory.create_smoother(non_existent_scaler_name)

        self.assertRaises(ValueError, test_function)

