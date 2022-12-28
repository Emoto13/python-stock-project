class PathCreator:
    @staticmethod
    def create_checkpoint_path(checkpoint="", periodicity="undefined", predictor_name="undefined",
                               stock_symbol="undefined"):
        if checkpoint is not None and checkpoint.strip() != "":
            return checkpoint
        return f"checkpoints/{predictor_name}/{stock_symbol}/{periodicity}/{stock_symbol}.ckpt"

    @staticmethod
    def create_plot_path(plot_path="", predictor_name="undefined", stock_symbol="undefined"):
        if plot_path is not None and plot_path.strip() != "":
            return plot_path
        return f"plots/{predictor_name}/{stock_symbol}/{stock_symbol}.jpg"
