class PathCreator:
    @staticmethod
    def create_checkpoint_path(checkpoint="", periodicity="undefined",
                               predictor_name="undefined",
                               stock_symbol="undefined"):
        if checkpoint is not None and checkpoint.strip() != "":
            return checkpoint
        path = [
            "checkpoints",
            f"{predictor_name}",
            f"{stock_symbol}",
            f"{periodicity}",
            f"{stock_symbol}.ckpt"
        ]
        return "/".join(path)

    @staticmethod
    def create_plot_path(plot_path="", periodicity="undefined",
                         predictor_name="undefined",
                         stock_symbol="undefined"):
        if plot_path is not None and plot_path.strip() != "":
            return plot_path
        path = [
            "plots",
            f"{predictor_name}",
            f"{stock_symbol}",
            f"{periodicity}",
            f"{stock_symbol}.jpg"
        ]
        return "/".join(path)
