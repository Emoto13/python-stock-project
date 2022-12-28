from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    @abstractmethod
    def preprocess(self, data, **kwargs):
        pass

