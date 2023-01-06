from abc import ABC, abstractmethod


class BaseIntegrator(ABC):
    def __init__(self, credentials=None):
        self.credentials = credentials

    def with_credentials(self, credentials=None):
        self.credentials = credentials
        return self

    @abstractmethod
    def get_data(self, stock_symbol="", periodicity="daily", *args,
                 **kwargs):
        pass
