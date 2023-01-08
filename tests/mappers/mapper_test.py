import unittest
from dataclasses import dataclass

from src.mappers import to_predict_request
from src.mappers.mapper import convert_to_object, object_or_none


class TestMapper(unittest.TestCase):
    @dataclass
    class DummyCls:
        string_field: str
        int_field: int

    def test_to_predict_request_works_correctly(self):
        model = "some_model"
        stock_symbol = "GOOGL"
        request = to_predict_request(model, stock_symbol)
        expected_fields = {
            "model": model,
            "stock_symbol": stock_symbol,
            "scaler_config": None,
            "smoother_config": None,
            "train_config": None,
            "test_config": None,
            "predict_config": None,
            "additional_model_parameters": None,
        }

        for key, value in expected_fields.items():
            self.assertIn(key, request.__dict__)
            if value:
                self.assertEqual(value, request.__dict__[key])

    def test_convert_to_object_works_correctly(self):
        data = {
            "string_field": "value",
            "int_field": 1,
        }

        result = convert_to_object(TestMapper.DummyCls, data)
        self.assertIsInstance(result, TestMapper.DummyCls)
        self.assertEqual(data["string_field"], result.string_field)
        self.assertEqual(data["int_field"], result.int_field)

    def test_object_or_none_works_correctly(self):
        test_cases = [
            {
                "request": {},
                "key": "some_key",
                "cls": TestMapper.DummyCls,
                "expected": None,
            },
            {
                "request": {
                    "bad_key": None,
                },
                "key": "bad_key",
                "cls": TestMapper.DummyCls,
                "expected": None,
            },
            {
                "request": {
                    "dummy_cls": {
                        "int_field": 1,
                        "string_field": "value"
                    },
                },
                "key": "dummy_cls",
                "cls": TestMapper.DummyCls,
                "expected": TestMapper.DummyCls(int_field=1, string_field="value"),
            }
        ]

        for test_case in test_cases:
            result = object_or_none(test_case["request"], test_case["key"], test_case["cls"])
            self.assertEqual(test_case["expected"], result)