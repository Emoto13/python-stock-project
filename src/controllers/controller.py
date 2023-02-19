from threading import Thread

from src.config import Config
from src.integrators import IntegratorFactory
from src.models.internal.predict_request import PredictRequest
from src.plotters import ComparisonPlotter
from src.predictors import PredictorFactory
from src.preprocessors.scalers.scaler_factory import ScalerFactory
from src.preprocessors.smoothers import SmootherFactory
from src.utils import try_catch
from src.utils.path_creator import PathCreator


class Controller:
    @staticmethod
    @try_catch
    def predict(predict_request: PredictRequest):
        config = Config.get_config()

        integrator = IntegratorFactory.get_integrator(
            integrator_name=config.integrator
        )
        integrator = integrator.with_credentials(config.api_key)
        data = integrator.get_data(
            stock_symbol=predict_request.stock_symbol,
            periodicity=predict_request.periodicity)

        if predict_request.smoother_config is not None:
            smoother = SmootherFactory.create_smoother(
                predict_request.smoother_config.name
            )
            smoother.preprocess(
                **predict_request.smoother_config.additional_data
            )

            data.price = smoother.preprocess(
                data=data.price.values,
                **predict_request.smoother_config.additional_data)

        if predict_request.scaler_config is not None:
            scaler = ScalerFactory.create_scaler(
                predict_request.scaler_config.name
            )
            data.price = scaler.preprocess(
                data=data.price.values,
                **predict_request.scaler_config.additional_data)

        predictor_class = PredictorFactory.create_predictor(
            predict_request.model
        )
        predictor = predictor_class(
            dataframe=data,
            stock_symbol=predict_request.stock_symbol,
            periodicity=predict_request.periodicity)

        predictor.create_model(
            **predict_request.additional_model_parameters
        )
        experiment_result = predictor.run_experiment(
            should_train=predict_request.train_config.should_train,
            should_test=predict_request.test_config.should_test,
            should_load=predict_request.predict_config.should_load,
            time_ahead=predict_request.predict_config.time_ahead)

        if predict_request.predict_config.should_generate_plot:
            plot_path = PathCreator.create_plot_path(
                periodicity=predict_request.periodicity,
                stock_symbol=predict_request.stock_symbol,
                predictor_name=predict_request.model)

            ComparisonPlotter.plot(
                data.reset_index(), experiment_result.predictions, plot_path
            )

        return experiment_result

    @staticmethod
    @try_catch
    def predict_async(predict_request: PredictRequest):
        thread = Thread(target=Controller.predict, args=(predict_request,))
        thread.start()
        return {"message": "Response received successfully"}
