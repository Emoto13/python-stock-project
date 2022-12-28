import unittest

from src.integrators import AlphaVantageIntegrator, IntegratorFactory


class TestIntegratorFactory(unittest.TestCase):
    def test_integrator_factory_works_correctly(self):
        test_cases = [
            {
                "integrator_name": "alpha_vantage",
                "class": AlphaVantageIntegrator
            },
        ]

        for test_case in test_cases:
            integrator = IntegratorFactory.get_integrator(test_case["integrator_name"])
            self.assertIsInstance(integrator, test_case["class"], "integrator should be of correct type")

    def test_integrator_factory_throws_error_on_invalid_input(self):
        non_existent_integrator_name = "non_existent"

        def test_function():
            _ = IntegratorFactory.get_integrator(non_existent_integrator_name)

        self.assertRaises(ValueError, test_function)

