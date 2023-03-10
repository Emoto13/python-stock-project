import os
from matplotlib import pyplot as plt

from src.utils.path_creator import PathCreator


class ComparisonPlotter:
    @staticmethod
    def plot(original_data=None, predictions=None, plot_path=None):
        """
        :param original_data: pandas dataframe with date and price columns
        :param predictions: pandas dataframe with date and price columns
        :param plot_path: Optional - specifies path for the generated plot.
         If not provided will dynamically create one.
        :return: None
        """
        plot_path = PathCreator.create_plot_path(plot_path)
        dir_name = os.path.dirname(plot_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        plt.figure()
        for frame in [original_data, predictions]:
            plt.plot(frame['date'], frame['price'])

        plt.title('Stock price actual vs prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.savefig(plot_path)
