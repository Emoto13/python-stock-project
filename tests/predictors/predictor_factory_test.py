import unittest

from src.predictors import PredictorFactory
from src.predictors.prophet.wrapper import ProphetWrapper
from src.predictors.transformer.wrapper import TransformerWrapper


class TestPredictorFactory(unittest.TestCase):
    def test_predictor_factory_works_correctly(self):
        test_cases = [
            {
                "model": "prophet",
                "class": ProphetWrapper
            },
            {
                "model": "transformer",
                "class": TransformerWrapper
            },
        ]

        for test_case in test_cases:
            model = PredictorFactory.create_predictor(test_case["model"])
            self.assertIs(model, test_case["class"], "model should be of correct type")

    def test_predictor_factory_throws_error_on_invalid_input(self):
        non_existent_model = "non_existent"

        def test_function():
            _ = PredictorFactory.create_predictor(non_existent_model)

        self.assertRaises(ValueError, test_function)
