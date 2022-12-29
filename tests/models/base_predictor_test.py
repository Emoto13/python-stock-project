import unittest
from unittest.mock import patch

from src.models import BasePredictor


class TestBasePredictor(unittest.TestCase):
    @patch.multiple(BasePredictor, __abstractmethods__=set())
    def test_base_predictor(self):
        try:
            predictor = BasePredictor()
            predictor.create_model()
            predictor.train()
            predictor.load()
            predictor.test()
            predictor.predict()
        except TypeError:
            self.fail("base predictor raised unexpected error")

