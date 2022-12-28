import unittest

from src.predictors.prophet.predictor.utils import get_seasonality, get_seasonality_parameters, get_frequency_parameters


class TestUtils(unittest.TestCase):
    def test_get_seasonality_works_correctly(self):
        test_cases = [
            {
                "name": "get_seasonality throws error",
                "periodicity": "invalid",
                "callback": get_seasonality,
                "should_throw_error": True,
                "error_type": ValueError,
            },
            {
                "name": "get_seasonality_parameters throws error",
                "periodicity": "invalid",
                "callback": get_seasonality_parameters,
                "should_throw_error": True,
                "error_type": ValueError,
            },
            {
                "name": "get_frequency_parameters throws error",
                "periodicity": "invalid",
                "callback": get_frequency_parameters,
                "should_throw_error": True,
                "error_type": ValueError,
            },
        ]

        for test_case in test_cases:
            def test_func():
                _ = test_case["callback"](test_case["periodicity"])

            print(test_case["name"])
            if test_case["should_throw_error"]:
                self.assertRaises(test_case["error_type"], test_func)
            else:
                pass
