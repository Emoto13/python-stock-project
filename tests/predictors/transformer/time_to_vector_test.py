import unittest

import tensorflow as tf

from src.predictors.transformer.predictor.time_to_vector import Time2Vector


class TestTime2Vector(unittest.TestCase):
    def setUp(self):
        super(TestTime2Vector, self).setUp()
        self.sequence_len = 10
        self.time_to_vector = Time2Vector(sequence_len=self.sequence_len)

    def test_call_model(self):
        input_data = tf.random.uniform((self.sequence_len, self.sequence_len, 1))
        result = self.time_to_vector(input_data)
        self.assertEqual((self.sequence_len, self.sequence_len, 2), result.shape)
        self.assertEqual(tf.dtypes.float32, result.dtype)

