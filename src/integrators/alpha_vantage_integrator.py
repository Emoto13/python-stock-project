import pandas as pd

from src.models import BaseIntegrator
import pandas_datareader as web

MAP_PERIODICITY_TO_AV_PERIODICITY = {
    "daily": "daily-adjusted",
}


class AlphaVantageIntegrator(BaseIntegrator):
    def __init__(self, api_key=""):
        super().__init__(credentials=api_key)

    def get_data(self, stock_symbol="", periodicity="daily", *args, **kwargs):
        if periodicity in MAP_PERIODICITY_TO_AV_PERIODICITY:
            periodicity = MAP_PERIODICITY_TO_AV_PERIODICITY[periodicity]

        df = web.DataReader(stock_symbol, f"av-{periodicity}",
                            api_key=self.credentials)
        df["price"] = (df["high"] + df["low"]) / 2
        df.index.name = "date"
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
        return df[["price"]]
