import unittest

from src.utils import try_catch


class TestTryCatch(unittest.TestCase):
    def test_try_catch_no_exception(self):
        @try_catch
        def test_func():
            return 1

        result = test_func()
        self.assertNotIsInstance(result, Exception)
        self.assertEqual(1, result)

    def test_try_catch_with_exception(self):
        msg = "Some error"

        @try_catch
        def test_func():
            raise Exception(msg)

        result = test_func()
        self.assertIsInstance(result, Exception)
        self.assertEqual(msg, str(result))
