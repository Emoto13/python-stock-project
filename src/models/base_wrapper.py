from abc import ABC, abstractmethod


class BaseWrapper(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_model(self, **kwargs):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict_once(self, data=None):
        pass

    @abstractmethod
    def predict_ahead(self, data=None, time_ahead=1):
        pass

    @abstractmethod
    def run_experiment(self, should_load=False, should_train=False, should_test=False, time_ahead=30):
        pass
