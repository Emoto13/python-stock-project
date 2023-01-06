import unittest

from src.utils.path_creator import PathCreator


class TestPathCreator(unittest.TestCase):
    def test_create_checkpoint_path_work_correctly_for_none_checkpoint(self):
        periodicity = "daily"
        predictor_name = "name"
        stock_symbol = "GOOGL"
        checkpoint = f"checkpoints/{predictor_name}/{stock_symbol}/{periodicity}/checkpoint.ckpt"
        path = PathCreator.create_checkpoint_path(
            periodicity="daily",
            predictor_name="name",
            stock_symbol="GOOGL"
        )
        self.assertEqual(checkpoint, path)

    def test_create_checkpoint_path_work_correctly_for_some_checkpoint(self):
        checkpoint = "some_checkpoint.ckpt"
        path = PathCreator.create_checkpoint_path(
            checkpoint=checkpoint,
        )
        self.assertEqual(checkpoint, path)

    def test_create_plot_path_work_correctly_for_none_path(self):
        predictor_name = "name"
        stock_symbol = "GOOGL"
        periodicity = "daily"
        plot_path = f"plots/{predictor_name}/{stock_symbol}/{periodicity}/plot.jpg"
        path = PathCreator.create_plot_path(
            predictor_name=predictor_name,
            stock_symbol=stock_symbol,
            periodicity=periodicity,
        )
        self.assertEqual(plot_path, path)

    def test_create_plot_path_work_correctly_for_some_path(self):
        plot_path = f"plot.jpg"
        path = PathCreator.create_plot_path(
            plot_path=plot_path
        )
        self.assertEqual(plot_path, path)
