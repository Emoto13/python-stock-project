import unittest

import tensorflow as tf

from src.predictors.transformer.predictor.transformer_encoder import TransformerEncoder


class TestTransformerEncoder(unittest.TestCase):
    def setUp(self):
        super(TestTransformerEncoder, self).setUp()
        self.encoder = TransformerEncoder()

    def test_call_model(self):
        shape = (10, 1, 1)
        input_data = tf.random.uniform(shape)
        result = self.encoder(input_data)
        self.assertEqual((shape[0], shape[1], shape[2] + 2), result.shape)
        self.assertEqual(tf.dtypes.float32, result.dtype)

