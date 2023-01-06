import os.path
import unittest

from unittest.mock import MagicMock, Mock
import pandas as pd
import tensorflow as tf

from src.predictors.transformer.predictor import TransformerPredictor
from src.predictors.transformer.wrapper import TransformerWrapper


class TestTransformerWrapper(unittest.TestCase):
    def setUp(self):
        self.wrapper = TransformerWrapper(
            dataframe=pd.DataFrame.from_dict(
                {
                    "price": [0.5 + i for i in range(1000)],
                    "date": pd.date_range(start="2010-01-01", periods=1000).to_pydatetime().tolist()
                }
            ).set_index("date"),
        )
        self.wrapper.model = Mock()
        self.wrapper.model.load = MagicMock(return_value=True)
        self.wrapper.model.train = MagicMock(return_value=True)
        self.wrapper.model.test = MagicMock(return_value=True)
        self.wrapper.model.predict = MagicMock(return_value=tf.convert_to_tensor([[1.0]]))

    def test_run_experiment_load_test_train(self):
        os.path.exists = MagicMock(return_value=True)
        self.wrapper.run_experiment(should_load=True, should_test=True, should_train=True)
        self.wrapper.model.load.assert_called_once()
        self.wrapper.model.train.assert_called_once()
        self.wrapper.model.test.assert_called_once()

    def test_predict_once_works_correctly(self):
        expected_prediction = 1.0
        prediction = self.wrapper.predict_once()
        self.assertEqual(expected_prediction, prediction, "predict_once should work correctly")
        self.wrapper.model.predict.assert_called_once()

    def test_train_works_correctly(self):
        self.wrapper.train()
        self.wrapper.model.train.assert_called_once()

    def test_load_works_correctly(self):
        self.wrapper.load()
        self.wrapper.model.load.assert_called_once()

    def test_create_model_works_correctly(self):
        model = self.wrapper.create_model()
        self.assertIsInstance(model, TransformerPredictor)
