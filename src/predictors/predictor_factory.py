from src.predictors.prophet.wrapper import ProphetWrapper
from src.predictors.transformer.wrapper import TransformerWrapper

PREDICTORS = {
    "prophet": ProphetWrapper,
    "transformer": TransformerWrapper,
}


class PredictorFactory:
    @staticmethod
    def create_predictor(predictor_name=""):
        if predictor_name not in PREDICTORS:
            raise ValueError("Invalid predictor name")
        return PREDICTORS[predictor_name]
