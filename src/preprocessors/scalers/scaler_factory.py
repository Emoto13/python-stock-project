from src.preprocessors.scalers.min_max import MinMax

NAME_TO_SCALER = {
    "min_max": MinMax
}


class ScalerFactory:
    @staticmethod
    def create_scaler(scaler_name=""):
        if scaler_name not in NAME_TO_SCALER:
            raise ValueError("Scaler doesn't exist")
        return NAME_TO_SCALER[scaler_name]()
