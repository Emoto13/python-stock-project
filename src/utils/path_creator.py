class PathCreator:

    @staticmethod
    def create_checkpoint_path(checkpoint=None, periodicity=None,
                               predictor_name=None,
                               stock_symbol=None):
        if is_segment_usable(checkpoint):
            return checkpoint
        raw_path = [
            "checkpoints",
            predictor_name,
            stock_symbol,
            periodicity,
            "checkpoint.ckpt"
        ]
        path = filter_empty_segments(raw_path)
        return "/".join(path)

    @staticmethod
    def create_plot_path(plot_path="", periodicity=None,
                         predictor_name=None,
                         stock_symbol=None,
                         mode=None):
        if is_segment_usable(plot_path):
            return plot_path
        raw_path = [
            "plots",
            predictor_name,
            mode,
            stock_symbol,
            periodicity,
            "plot.jpg"
        ]
        path = filter_empty_segments(raw_path)
        return "/".join(path)


def is_segment_usable(segment):
    if (segment is None or
            not isinstance(segment, str) or
            segment.strip() == ""):
        return False
    return True


def filter_empty_segments(path):
    filtered_path = filter(is_segment_usable, path)
    return list(filtered_path)
