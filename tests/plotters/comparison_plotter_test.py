import os
import unittest
from unittest.mock import MagicMock

import pandas as pd
from matplotlib import pyplot as plt

from src.plotters import ComparisonPlotter


class TestComparisonPlotter(unittest.TestCase):
    def test_comparison_plotter_works_correctly(self):
        plotter = ComparisonPlotter()
        plt.savefig = MagicMock()
        os.makedirs = MagicMock()
        os.path.exists = MagicMock(return_value=False)
        plotter.plot(pd.DataFrame.from_dict(
            {
                "price": [],
                "date": []
            }
        ), pd.DataFrame.from_dict(
            {
                "price": [],
                "date": []
            }
        ))
        plt.savefig.assert_called_once()
        os.makedirs.assert_called_once()
        os.path.exists.assert_called_once()
