import unittest
from unittest.mock import MagicMock, Mock
import pandas as pd

from src.predictors.prophet.predictor import ProphetPredictor
from src.predictors.prophet.wrapper import ProphetWrapper


# NOTE: This test is put in the root of test pacakge for legacy reasons
# related to PyStan


class TestProphetWrapper(unittest.TestCase):
    def setUp(self):
        self.wrapper = ProphetWrapper(
            data=pd.DataFrame.from_dict(
                {
                    "price": [0.5, 1.5, 2.5],
                    "index": ["01-01-2010", "01-02-2010", "01-03-2010"],
                }
            )
        )
        self.wrapper.model = Mock()

    def test_run_experiment_works_correctly(self):
        expected_predictions = pd.DataFrame.from_dict(
            {
                "price": [1.5, 2.5, 3.5],
                "index": ["01-04-2010", "01-05-2010", "01-06-2010"],
            }
        )
        self.wrapper.model.test = MagicMock()
        self.wrapper.model.predict = MagicMock(
            return_value=expected_predictions)

        predictions = self.wrapper.run_experiment(
            should_load=True,
            should_train=True,
            should_test=True)

        self.assertTrue(
            expected_predictions.equals(predictions),
            "predict_ahead should work correctly")
        self.wrapper.model.test.assert_called_once()
        self.wrapper.model.predict.assert_called_once()

    def test_predict_once_works_correctly(self):
        expected_predictions = pd.DataFrame.from_dict(
            {
                "price": [1.5],
                "index": ["01-04-2010"],
            }
        )
        self.wrapper.model.predict = MagicMock(
            return_value=expected_predictions)
        predictions = self.wrapper.predict_once(
            data=self.wrapper.data)
        self.assertTrue(
            expected_predictions.equals(predictions),
            "predict_once should work correctly")
        self.wrapper.model.predict.assert_called_once()

    def test_train(self):
        self.wrapper.train()

    def test_load(self):
        self.wrapper.load()

    def test_create_model(self):
        model = self.wrapper.create_model()
        self.assertIsInstance(model, ProphetPredictor)
