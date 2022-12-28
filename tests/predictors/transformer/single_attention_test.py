import unittest

import tensorflow as tf

from src.predictors.transformer.predictor.single_attention import SingleAttention


class TestSingleAttention(unittest.TestCase):
    def setUp(self):
        super(TestSingleAttention, self).setUp()
        self.single_attention = SingleAttention()

    def test_call_model(self):
        input_data = tf.convert_to_tensor([[1.]], dtype=float)
        self.single_attention.build(input_data.shape)
        result = self.single_attention((input_data, input_data, input_data))
        self.assertEqual((1, 32), result.shape)
        self.assertEqual(tf.dtypes.float32, result.dtype)
