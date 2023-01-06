import os


class Config:
    __instance = None

    def __init__(self):
        self.integrator = os.getenv("INTEGRATOR", "alpha_vantage")
        self.api_key = os.getenv("ALPHAVANTAGE_API_KEY", "not set")

    @classmethod
    def get_config(cls):
        if not cls.__instance:
            cls.__instance = Config()
        return cls.__instance
