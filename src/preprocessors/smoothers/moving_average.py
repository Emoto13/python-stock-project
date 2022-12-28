import pandas as pd
from src.models import BasePreprocessor


class MovingAverage(BasePreprocessor):
    def preprocess(self, data, window=10, **kwargs):
        smoothed_signal = pd.Series(data).rolling(window, min_periods=1).mean()
        return smoothed_signal.to_numpy()
