import unittest

import pandas as pd
from prophet import Prophet

from src.predictors.prophet.predictor import ProphetPredictor


# NOTE: This test is put in the root of test pacakge for legacy reasons
# related to PyStan

class TestProphetPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = ProphetPredictor(periodicity="daily")

    def test_test_works_correctly(self):
        data = pd.DataFrame.from_dict(
            {
                "y": [0.5 + i for i in range(1000)],
                "ds": pd.date_range(
                    start="2010-01-01",
                    periods=1000).to_pydatetime().tolist()
            }
        )
        time_ahead = 3
        _ = self.predictor.predict(data, time_ahead=time_ahead)
        horizon_percentage = 0.2
        test_df = self.predictor.test(horizon_percentage=horizon_percentage)
        self.assertTrue(test_df is not None, "test df should not be empty")

    def test_predict_works_correctly(self):
        data = pd.DataFrame.from_dict(
            {
                "y": [0.5, 1.5, 2.5],
                "ds": ["01-01-2010", "01-02-2010", "01-03-2010"],
            }
        )
        time_ahead = 3
        predictions = self.predictor.predict(data, time_ahead=time_ahead)
        self.assertEqual(
            time_ahead + len(data),
            len(predictions),
            "predictions should be of correct size")

    def test_train(self):
        self.predictor.train()

    def test_load(self):
        self.predictor.load()

    def test_create_model(self):
        model = self.predictor.create_model()
        self.assertIsInstance(model, Prophet)


if __name__ == "__main__":
    unittest.main()
