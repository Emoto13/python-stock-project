import unittest
from unittest.mock import MagicMock

import pandas as pd
import pandas_datareader as web

from src.integrators import AlphaVantageIntegrator


class TestAlphaVantageIntegrator(unittest.TestCase):
    def test_get_data_works_correctly(self):
        web.DataReader = MagicMock(return_value=pd.DataFrame.from_dict(
            {
                "high": [1, 2, 3],
                "low": [0, 1, 2],
                "index": ["01-01-2010", "01-02-2010", "01-03-2010"],
            }
        ))

        stock_symbol = "GOOGL"
        integrator = AlphaVantageIntegrator().with_credentials(
            credentials="credentials")
        data = integrator.get_data(
            stock_symbol=stock_symbol,
            periodicity="daily")
        self.assertEqual({"price"}, set(data.columns))
        self.assertEqual("date", data.index.name)
        self.assertListEqual(
            [0.5, 1.5, 2.5],
            list(data.price.values))
