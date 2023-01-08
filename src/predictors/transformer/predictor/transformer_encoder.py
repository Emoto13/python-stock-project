from keras.layers import Layer, Dropout, LayerNormalization, Conv1D
from .multi_attention import MultiAttention


class TransformerEncoder(Layer):
    def __init__(self, d_k=32, d_v=32, n_heads=16, ff_dim=256, filter_size=3,
                 dropout=0.1):
        super().__init__()
        self.key_dimension = d_k
        self.value_dimension = d_v
        self.attention_heads_amount = n_heads
        self.ff_dim = ff_dim
        self.filter_size = filter_size
        self.dropout = dropout
        self.attention_heads = []
        self.multi_attention = MultiAttention(
            d_k=self.key_dimension,
            d_v=self.value_dimension,
            n_heads=self.attention_heads_amount,
            filter_size=self.filter_size)
        self.attention_dropout = Dropout(self.dropout)
        self.attention_normalization = None
        self.ff_conv_one_dimension_first = Conv1D(
            filters=self.ff_dim,
            kernel_size=1,
            activation='relu')
        self.ff_conv_one_dimension_second = Conv1D(
            filters=self.filter_size,
            kernel_size=1)
        self.ff_dropout = Dropout(self.dropout)
        self.ff_normalization = None

    def build(self, input_shape):
        self.attention_normalization = LayerNormalization(
            input_shape=input_shape, epsilon=1e-6)
        self.ff_normalization = LayerNormalization(
            input_shape=input_shape, epsilon=1e-6)

    def call(self, input_data):  # inputs = (in_seq, in_seq, in_seq)
        inputs = (input_data, input_data, input_data)
        attention_layer = self.multi_attention(inputs)
        attention_layer = self.attention_dropout(attention_layer)
        attention_layer = self.attention_normalization(
            inputs[0] + attention_layer
        )
        ff_layer = self.ff_conv_one_dimension_first(attention_layer)
        ff_layer = self.ff_conv_one_dimension_second(ff_layer)
        ff_layer = self.ff_dropout(ff_layer)
        ff_layer = self.ff_normalization(inputs[0] + ff_layer)
        return ff_layer
