from abc import ABC, abstractmethod


class BasePredictor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def predict(self, data=None):
        pass

    @abstractmethod
    def load(self):
        pass
