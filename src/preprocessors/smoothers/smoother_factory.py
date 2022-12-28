from src.preprocessors.smoothers.moving_average import MovingAverage

NAME_TO_SMOOTHER = {
    'moving_average': MovingAverage,
}


class SmootherFactory:
    @staticmethod
    def create_smoother(smoother_name=""):
        if smoother_name not in NAME_TO_SMOOTHER:
            raise ValueError("Smoother doesn't exist")
        return NAME_TO_SMOOTHER[smoother_name]()
