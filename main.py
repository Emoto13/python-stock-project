import numpy as np
import pandas as pd

from src.integrators import IntegratorFactory
from src.plotters import ComparisonPlotter
from src.predictors.prophet.wrapper import ProphetWrapper

api_key = "RJ94T0LY5VZAN6EK"
stock_symbol = "GOOGL"
predictor_name = "transformer"
# os.getenv("ALPHAVANTAGE_API_KEY")

if __name__ == "__main__":
    integrator = IntegratorFactory.get_integrator(integrator_name="alpha_vantage")
    integrator = integrator.with_credentials(api_key)
    data = integrator.get_data(stock_symbol="GOOGL", periodicity="weekly")

    # predictor_obj = PredictorFactory.create_predictor(predictor_name=predictor_name)
    # checkpoint = CheckpointCreator.create_checkpoint_path(predictor_name=predictor_name, stock_symbol=stock_symbol)
    # predictor = predictor_obj(dataframe=data,
    #                          checkpoint=checkpoint)
    # predictor.run_experiment(should_load=True, should_test=True, should_train=True)

    pr = ProphetWrapper(data=data, periodicity="weekly")
    pred = pr.run_experiment(should_test=True, time_ahead=52)

    data = data.reset_index()
    data["date"] = pd.to_datetime(data["date"])
    print(data, pred)

    ComparisonPlotter.plot(data, pred)
    # x, y = PreProcessor.prepare_data(data.price.values, 128)

    # print(x, y)
