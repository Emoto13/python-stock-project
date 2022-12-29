import unittest
from unittest.mock import patch

from src.models import BaseIntegrator


class TestBaseIntegrator(unittest.TestCase):
    @patch.multiple(BaseIntegrator, __abstractmethods__=set())
    def test_base_integrator(self):
        try:
            integrator = BaseIntegrator()
            self.assertIsNone(integrator.credentials)
            some_credentials = "some_credentials"
            integrator = integrator.with_credentials(some_credentials)
            self.assertEqual(some_credentials, integrator.credentials)
            _ = integrator.get_data(stock_symbol="SYMBOL", periodicity="daily")
        except TypeError:
            self.fail("base integrator raised unexpected error")

