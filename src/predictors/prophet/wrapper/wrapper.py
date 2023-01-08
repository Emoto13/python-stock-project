from src.models import BaseWrapper
from src.models.internal import ExperimentResult
from src.predictors.prophet.predictor import ProphetPredictor
from .preprocessor import PreProcessor


class ProphetWrapper(BaseWrapper):
    def __init__(self, dataframe=None, periodicity="weekly",
                 stock_symbol="undefined"):
        super().__init__()
        self.data = PreProcessor.prepare_data(dataframe)
        self.periodicity = periodicity
        self.stock_symbol = stock_symbol
        self.model = None

    def create_model(self, **kwargs):
        self.model = ProphetPredictor(periodicity=self.periodicity, **kwargs)
        return self.model

    def train(self):
        pass

    def load(self):
        pass

    def test(self):
        return self.model.test()

    def predict_once(self, data=None):
        return self.model.predict(data, time_ahead=1)

    def predict_ahead(self, data=None, time_ahead=1):
        return self.model.predict(data=data, time_ahead=time_ahead)

    def run_experiment(self, should_load=False, should_train=False,
                       should_test=False, time_ahead=30):
        experiment_result = ExperimentResult()

        predictions = self.predict_ahead(data=self.data, time_ahead=time_ahead)
        experiment_result.with_predictions(predictions)

        if should_test:
            test_results = self.test()
            experiment_result.with_test_results(test_results)
        return experiment_result
