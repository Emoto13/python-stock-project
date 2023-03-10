from sklearn.preprocessing import MinMaxScaler
from src.models import BasePreprocessor


class MinMax(BasePreprocessor):
    def preprocess(self, data, **kwargs):
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(data.reshape(-1, 1))[:, 0]
