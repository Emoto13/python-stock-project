import os
import pandas as pd
import numpy as np
import tensorflow as tf
from src.models import BaseWrapper
from src.predictors.transformer.predictor import TransformerPredictor
from src.utils.path_creator import PathCreator
from .preprocessor import PreProcessor


class TransformerWrapper(BaseWrapper):
    def __init__(self, dataframe=None, sequence_len=128,
                 key_dimension=32, value_dimension=32, n_heads=16,
                 ff_dimension=256, filter_size=3,
                 dropout=0.1, epochs=50, batch_size=32,
                 train_test_split=0.1, validation_split=0.1,
                 checkpoint=None, symbol="undefined", periodicity="undefined"):
        super().__init__()
        self.model = None
        self.dataframe = dataframe
        self.sequence_len = sequence_len
        self.checkpoint = PathCreator.create_checkpoint_path(
            checkpoint=checkpoint,
            periodicity=periodicity,
            stock_symbol=symbol,
            predictor_name="transformer"
        )
        self.data, self.target = PreProcessor.prepare_data(
            self.dataframe.price.values,
            sequence_len
        )
        split_data = PreProcessor.split(
            self.data,
            self.target,
            train_test_split
        )
        self.train_data, self.train_target, \
            self.test_data, self.test_target = split_data

        # set model attributes
        self.key_dimension = key_dimension
        self.value_dimension = value_dimension
        self.n_heads = n_heads
        self.ff_dimension = ff_dimension
        self.filter_size = filter_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

    def create_model(self, **kwargs):
        model = TransformerPredictor(sequence_len=self.sequence_len,
                                     key_dimension=self.key_dimension,
                                     value_dimension=self.value_dimension,
                                     n_heads=self.n_heads,
                                     ff_dimension=self.ff_dimension,
                                     filter_size=self.filter_size,
                                     dropout=self.dropout,
                                     epochs=self.epochs,
                                     batch_size=self.batch_size,
                                     validation_split=self.validation_split,
                                     checkpoint=self.checkpoint,
                                     )
        return model

    def train(self):
        self.model.train(self.train_data, self.train_target)

    def test(self):
        self.model.test(self.test_data, self.test_target)

    def predict_once(self, data=None):
        prediction = self.model.predict(self.data)
        return prediction[0][0]

    def predict_ahead(self, data=None, time_ahead=30):
        time_steps = [i for i in range(1, time_ahead + 1)]
        output = []
        temp_predictions = list(data)
        # Insert dummy element
        temp_predictions.insert(0, 0)
        print(data)

        for i in range(time_ahead):
            model_input = np.array(temp_predictions[1:], dtype='float64')
            model_input = model_input.reshape([1, self.sequence_len, 1])
            prediction = self.model.predict(
                tf.convert_to_tensor(model_input, dtype='float64')
            )

            temp_predictions.append(prediction[0][0])
            temp_predictions = temp_predictions[1:]

            output.append(prediction[0].numpy()[0])
            print("Predicted date:", time_steps[i], output[-1])

        result_df = pd.DataFrame(
            dict(
                time_steps=pd.Series(time_steps),
                prediction=pd.Series(output)
            )
        )
        return result_df

    def run_experiment(self, should_load=False, should_train=False,
                       should_test=False, time_ahead=30):
        if should_load and os.path.exists(self.checkpoint):
            self.load()

        if should_train:
            self.train()

        if should_test:
            self.test()

        predictions = self.predict_ahead(
            self.dataframe.price.values[-self.sequence_len:],
            time_ahead=time_ahead)
        return predictions

    def load(self):
        self.model.load()
