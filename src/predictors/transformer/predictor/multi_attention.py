from tensorflow.keras.layers import Layer, Dense
import tensorflow as tf

from .single_attention import SingleAttention


class MultiAttention(Layer):
    def __init__(self, d_k=32, d_v=32, n_heads=16, filter_size=3):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.filter_size = filter_size
        self.attention_heads = [
                                   SingleAttention(d_k=self.d_k, d_v=self.d_v)
                               ] * self.n_heads
        self.linear = None

    def build(self, input_shape):
        self.linear = Dense(self.filter_size, input_shape=input_shape,
                            kernel_initializer='glorot_uniform',
                            bias_initializer='glorot_uniform')

    def call(self, inputs):
        attention = [
            self.attention_heads[i](inputs) for i in range(self.n_heads)
        ]
        combined_attention = tf.concat(attention, axis=-1)
        multi_linear = self.linear(combined_attention)
        return multi_linear
