import pandas as pd
import pandas_datareader as web

from src.models import BaseIntegrator

MAP_PERIODICITY_TO_AV_PERIODICITY = {
    "daily": "daily-adjusted",
}


class AlphaVantageIntegrator(BaseIntegrator):
    def __init__(self, api_key=""):
        super().__init__(credentials=api_key)

    def get_data(self, stock_symbol="", periodicity="daily", **kwargs):
        if periodicity in MAP_PERIODICITY_TO_AV_PERIODICITY:
            periodicity = MAP_PERIODICITY_TO_AV_PERIODICITY[periodicity]

        dataframe = web.DataReader(stock_symbol, f"av-{periodicity}",
                                   api_key=self.credentials)
        dataframe["price"] = (dataframe["high"] + dataframe["low"]) / 2
        dataframe.index.name = "date"
        dataframe.index = pd.to_datetime(dataframe.index, format='%Y-%m-%d')
        return dataframe[["price"]]
