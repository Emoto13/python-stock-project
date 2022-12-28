from flask import Flask

app = Flask(__name__)


class Handler:
    pass


@app.route("/predict")
def hello_world():
    """
        {
            model: string
            stock_symbol: string
            periodicity: string
            time_ahead: int
            should_generate_plot: bool
            smoother: string
            scaler: string
            additional_parameters: {}
        }

    :return:
    """
    return "Hello, World!"
