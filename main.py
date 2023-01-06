from src.handlers import app
from src.integrators import IntegratorFactory
from src.plotters import ComparisonPlotter
from src.predictors import PredictorFactory
from src.predictors.prophet.wrapper import ProphetWrapper
from src.utils.path_creator import PathCreator

# os env
api_key = ""
integrator = "alpha_vantage"

stock_symbol = "GOOGL"
predictor_name = "transformer"
periodicity = "weekly"

if __name__ == "__main__":
    app.run()

    """integrator = IntegratorFactory.get_integrator(integrator_name="alpha_vantage")
    integrator = integrator.with_credentials(api_key)
    data = integrator.get_data(stock_symbol=stock_symbol, periodicity=periodicity)
    predictor_obj = PredictorFactory.create_predictor(predictor_name=predictor_name)
    predictor = predictor_obj(dataframe=data,
                              stock_symbol=stock_symbol,
                              periodicity=periodicity)
    predictor.create_model()
    # print(predictor)
    pred = predictor.run_experiment(should_test=False, time_ahead=52)
    #
    print(pred, pred["date"])
    data = data.reset_index()
    ##data["date"] = pd.to_datetime(data["date"])
    # print(data, pred)

    ComparisonPlotter.plot(data, pred)
# x, y = PreProcessor.prepare_data(data.price.values, 128)

# print(x, y)"""
