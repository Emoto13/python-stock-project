import unittest
from unittest.mock import patch

from src.models import BasePreprocessor


class TestBasePreprocessor(unittest.TestCase):
    @patch.multiple(BasePreprocessor, __abstractmethods__=set())
    def test_base_predictor(self):
        try:
            preprocessor = BasePreprocessor()
            preprocessor.preprocess(None)
        except TypeError:
            self.fail("base preprocessor raised unexpected error")

