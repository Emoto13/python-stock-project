import os
import unittest

from src.config import Config


class TestConfig(unittest.TestCase):
    def test_config_works_correctly(self):
        os.environ["INTEGRATOR"] = "value_integrator"
        os.environ["ALPHAVANTAGE_API_KEY"] = "value_av_key"
        cfg = Config.get_config()

        cfg.integrator = os.getenv("INTEGRATOR")
        cfg.api_key = os.getenv("ALPHAVANTAGE_API_KEY")

        self.assertEqual(cfg, Config.get_config())
