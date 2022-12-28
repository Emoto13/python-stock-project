from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from src.models import BasePredictor
from .utils import get_seasonality, get_seasonality_parameters, get_frequency_parameters


class ProphetPredictor(BasePredictor):
    def __init__(self, periodicity='weekly', additional_params=None):
        """[summary]

        Args:
            periodicity (str, optional): [description]. Defaults to 'weekly'.
                changepoint_prior_scale (float, optional): [description]
                seasonality_prior_scale (float, optional): [description]
                holidays_prior_scale (float, optional): [description]
                seasonality_mode (str, optional): [description]
        """
        super().__init__()
        if additional_params is None:
            additional_params = {}
        self.periodicity = periodicity
        self.additional_params = additional_params
        self.model = self.create_model()
        self.data = None

    def create_model(self):
        seasonality = get_seasonality(self.periodicity)
        model = Prophet(**seasonality, **self.additional_params)
        seasonality_params = get_seasonality_parameters(self.periodicity)
        model.add_seasonality(
            name=self.periodicity,
            period=seasonality_params['period'],
            fourier_order=seasonality_params['fourier_order']
        )
        return model

    def train(self):
        pass

    def load(self):
        pass

    def test(self, horizon_percentage=0.2):
        horizon = (horizon_percentage * len(self.data))
        df_cv = cross_validation(self.model, initial=f"{3 * horizon} days", horizon=f'{horizon} days')
        df = performance_metrics(df_cv)
        return df

    def predict(self, data=None, time_ahead=1):
        self.model = self.model.fit(data)
        self.data = data
        frequency_params = get_frequency_parameters(self.periodicity)
        future_df = self.model.make_future_dataframe(periods=time_ahead,
                                                     freq=frequency_params['freq'])
        raw_predictions = self.model.predict(future_df)
        predictions = raw_predictions[['ds', 'yhat']]
        predictions = predictions.rename(columns={"ds": "date", "yhat": "price"})
        return predictions
