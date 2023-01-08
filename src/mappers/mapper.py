from dacite import from_dict

from src.models.internal import PredictRequest, PreprocessorConfig, \
    TrainConfig, TestConfig, PredictConfig

from src.utils import try_catch


@try_catch
def to_predict_request(model, stock_symbol, raw_request=None):
    if raw_request is None:
        raw_request = {}

    data = {
        "model": model,
        "stock_symbol": stock_symbol,
        "scaler_config": object_or_none(
            raw_request, "scaler_config", PreprocessorConfig
        ),
        "smoother_config": object_or_none(
            raw_request, "smoother_config", PreprocessorConfig
        ),
        "train_config": object_or_none(
            raw_request, "train_config", TrainConfig
        ),
        "test_config": object_or_none(
            raw_request, "test_config", TestConfig
        ),
        "predict_config": object_or_none(
            raw_request, "predict_config", PredictConfig
        ),
        **raw_request,
    }

    return from_dict(data_class=PredictRequest, data=data)


def object_or_none(raw_request, key, object_type):
    if key not in raw_request:
        return None
    result = convert_to_object(object_type, raw_request[key])
    if isinstance(result, Exception):
        return None
    return result


@try_catch
def convert_to_object(object_type, data):
    return from_dict(data_class=object_type, data=data)
