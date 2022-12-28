from tensorflow.keras.layers import Layer
import tensorflow as tf


class Time2Vector(Layer):
    def __init__(self, sequence_len=128):
        super().__init__()
        self.sequence_len = sequence_len
        self.weights_linear = self.add_weight(name='weight_linear',
                                              shape=(int(self.sequence_len),),
                                              initializer='uniform',
                                              trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                           shape=(int(self.sequence_len),),
                                           initializer='uniform',
                                           trainable=True)

        self.weights_periodic = self.add_weight(
            name='weight_periodic',
            shape=(int(self.sequence_len),),
            initializer='uniform',
            trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                             shape=(int(self.sequence_len),),
                                             initializer='uniform',
                                             trainable=True)

    def call(self, x):
        # Convert (batch, seq_len, 5) to (batch, seq_len)
        x = tf.squeeze(x, axis=(2,))
        time_linear = self.weights_linear * x + self.bias_linear
        time_linear = tf.expand_dims(time_linear, axis=-1)
        time_periodic = tf.math.sin(
          tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)
        return tf.concat([time_linear, time_periodic], axis=-1)
