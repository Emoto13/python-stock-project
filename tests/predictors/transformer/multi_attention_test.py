import unittest

import tensorflow as tf

from src.predictors.transformer.predictor.multi_attention import MultiAttention


class TestMultiAttention(unittest.TestCase):
    def setUp(self):
        super(TestMultiAttention, self).setUp()
        self.filter_size = 5
        self.multi_attention = MultiAttention(filter_size=self.filter_size)

    def test_call_model(self):
        input_data = tf.convert_to_tensor([[1.]], dtype=float)
        self.multi_attention.build(input_data.shape)
        result = self.multi_attention((input_data, input_data, input_data))
        self.assertEqual((1, self.filter_size), result.shape)
        self.assertEqual(tf.dtypes.float32, result.dtype)

