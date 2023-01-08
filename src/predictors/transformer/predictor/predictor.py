from keras.layers import Concatenate, GlobalAveragePooling1D,\
    Dropout, Dense, Input
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint

from src.models import BasePredictor
from .transformer_encoder import TransformerEncoder
from .time_to_vector import Time2Vector


class TransformerPredictor(BasePredictor):
    def __init__(self, sequence_len=128, key_dimension=32,
                 value_dimension=32, n_heads=16, ff_dimension=256,
                 filter_size=3, dropout=0.1, epochs=50, batch_size=32,
                 validation_split=0.1, checkpoint=None):
        super().__init__()
        self.sequence_len = sequence_len
        self.key_dimension = key_dimension
        self.value_dimension = value_dimension
        self.n_heads = n_heads
        self.ff_dimension = ff_dimension
        self.filter_size = filter_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.checkpoint = checkpoint
        self.model = self.create_model()

    def create_model(self):
        """Construct model"""
        in_seq = Input(shape=(self.sequence_len, 1))
        data = Time2Vector()(in_seq)
        data = Concatenate(axis=-1)([in_seq, data])

        transformer_encoder = TransformerEncoder(d_k=self.key_dimension,
                                                 d_v=self.value_dimension,
                                                 n_heads=self.n_heads,
                                                 ff_dim=self.ff_dimension,
                                                 filter_size=self.filter_size,
                                                 dropout=self.dropout)
        data = transformer_encoder(data)
        data = transformer_encoder(data)
        data = transformer_encoder(data)
        data = GlobalAveragePooling1D(data_format='channels_first')(data)
        data = Dropout(self.dropout)(data)
        data = Dense(64, activation='relu')(data)
        data = Dropout(self.dropout)(data)
        out = Dense(1, activation='linear')(data)
        model = Model(inputs=in_seq, outputs=out)

        optimizer = Adam()
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape'])
        return model

    def train(self, x_train=None, y_train=None):
        cp_callback = ModelCheckpoint(
            filepath=self.checkpoint,
            save_best_only=True,
            save_weights_only=True,
            verbose=1)
        self.model.fit(x_train, y_train,
                       epochs=self.epochs, verbose=1,
                       batch_size=self.batch_size,
                       callbacks=[cp_callback],
                       validation_split=self.validation_split)

    def test(self, test_data=None, test_target=None):
        return self.model.evaluate(x=test_data, y=test_target, verbose=1)

    def predict(self, data=None):
        return self.model(data, training=False)

    def load(self):
        return self.model.load_weights(self.checkpoint)
