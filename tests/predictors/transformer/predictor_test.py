import unittest
from unittest.mock import MagicMock, Mock

from src.predictors.transformer.predictor import TransformerPredictor


class TestTransformerPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = TransformerPredictor()

    def test_train_works_correctly(self):
        self.predictor.model = Mock()
        self.predictor.model.fit = MagicMock()
        self.predictor.model.compile = MagicMock()
        self.predictor.train()
        self.predictor.model.fit.assert_called_once()
        self.predictor.model.compile.assert_called_once()

    def test_test_works_correctly(self):
        self.predictor.model = Mock()
        self.predictor.model.evaluate = MagicMock()
        self.predictor.test()
        self.predictor.model.evaluate.assert_called_once()

    def test_predict_works_correctly(self):
        self.predictor.model = MagicMock()
        self.predictor.predict(None)
        self.predictor.model.assert_called_once()

    def test_load_works_correctly(self):
        self.predictor.model = Mock()
        self.predictor.model.load_weights = MagicMock()
        self.predictor.load()
        self.predictor.model.load_weights.assert_called_once()
