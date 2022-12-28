from src.models import BaseWrapper
from src.predictors.prophet.predictor import ProphetPredictor
from .preprocessor import PreProcessor


class ProphetWrapper(BaseWrapper):
    def __init__(self, data=None, periodicity="weekly", symbol="undefined") -> None:
        super().__init__()
        self.data = PreProcessor.prepare_data(data)
        self.periodicity = periodicity
        self.symbol = symbol
        self.model = None

    def create_model(self, **kwargs):
        self.model = ProphetPredictor(periodicity=self.periodicity, **kwargs)
        return self.model

    def train(self):
        pass

    def load(self):
        pass

    def test(self):
        self.model.test()

    def predict_once(self, data=None):
        return self.model.predict(data, time_ahead=1)

    def predict_ahead(self, data=None, time_ahead=1):
        return self.model.predict(data=data, time_ahead=time_ahead)

    def run_experiment(self, should_load=False, should_train=False, should_test=False, time_ahead=30):
        predictions = self.predict_ahead(data=self.data, time_ahead=time_ahead)
        if should_test:
            self.test()
        return predictions
