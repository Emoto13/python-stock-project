from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    should_train: bool


@dataclass
class TestConfig:
    should_test: bool


@dataclass
class PredictConfig:
    should_generate_plot: bool
    time_ahead: int
    should_load: bool

@dataclass
class PreprocessorConfig:
    name: str
    additional_data: Optional[dict]


@dataclass
class PredictRequest:
    model: str
    stock_symbol: str
    periodicity: Optional[str]

    train_config: Optional[TrainConfig | None]
    test_config: Optional[TestConfig | None]
    predict_config: Optional[PredictConfig | None]

    smoother_config: Optional[PreprocessorConfig | None]
    scaler_config: Optional[PreprocessorConfig | None]

    additional_model_parameters: Optional[dict]

    def __post_init__(self):
        if self.periodicity is None:
            self.periodicity = "daily"

        if self.train_config is None:
            self.train_config = TrainConfig(should_train=False)

        if self.test_config is None:
            self.test_config = TestConfig(
                should_test=False,
            )

        if self.predict_config is None:
            self.predict_config = PredictConfig(
                time_ahead=30,
                should_generate_plot=False,
                should_load=True
            )

        if self.additional_model_parameters is None:
            self.additional_model_parameters = {}

