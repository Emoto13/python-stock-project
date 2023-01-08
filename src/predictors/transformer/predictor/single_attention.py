from keras.layers import Layer, Dense
import tensorflow as tf
import numpy as np


class SingleAttention(Layer):
    def __init__(self, d_k=32, d_v=32):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.query = None
        self.key = None
        self.value = None

    def build(self, input_shape):
        self.query = Dense(self.d_k, input_shape=input_shape,
                           kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')
        self.key = Dense(self.d_k, input_shape=input_shape,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='glorot_uniform')
        self.value = Dense(self.d_v, input_shape=input_shape,
                           kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        query = self.query(inputs[0])
        key = self.key(inputs[1])

        attention_weights = tf.matmul(query, key, transpose_b=True)
        attention_weights = tf.map_fn(
          lambda x: x/np.sqrt(self.d_k), attention_weights)
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)

        value = self.value(inputs[2])
        attention_result = tf.matmul(attention_weights, value)
        return attention_result
