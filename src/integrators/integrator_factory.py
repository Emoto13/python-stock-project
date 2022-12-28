from .alpha_vantage_integrator import AlphaVantageIntegrator

INTEGRATORS = {
    "alpha_vantage": AlphaVantageIntegrator,
}


class IntegratorFactory:
    @staticmethod
    def get_integrator(integrator_name=""):
        if integrator_name not in INTEGRATORS:
            raise ValueError("Invalid integrator name")
        return INTEGRATORS[integrator_name]()
