import unittest
from unittest.mock import patch

from src.models import BaseWrapper


class TestBaseWrapper(unittest.TestCase):
    @patch.multiple(BaseWrapper, __abstractmethods__=set())
    def test_base_predictor(self):
        try:
            wrapper = BaseWrapper()
            wrapper.create_model()
            wrapper.train()
            wrapper.load()
            wrapper.test()
            wrapper.predict_once()
            wrapper.predict_ahead()
            wrapper.run_experiment()
        except TypeError:
            self.fail("base wrapper raised unexpected error")

