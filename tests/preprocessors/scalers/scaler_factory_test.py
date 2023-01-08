import unittest

from src.preprocessors.scalers.min_max import MinMax
from src.preprocessors.scalers.scaler_factory import ScalerFactory


class TestScalerFactory(unittest.TestCase):
    def test_scaler_factory_works_correctly(self):
        test_cases = [
            {
                "scaler_name": "min_max",
                "class": MinMax
            },
        ]

        for test_case in test_cases:
            scaler = ScalerFactory.create_scaler(test_case["scaler_name"])
            self.assertIsInstance(scaler, test_case["class"], "scaler should be of correct type")

    def test_scaler_factory_throws_error_on_invalid_input(self):
        non_existent_scaler_name = "non_existent"

        def test_function():
            _ = ScalerFactory.create_scaler(non_existent_scaler_name)

        self.assertRaises(ValueError, test_function)
