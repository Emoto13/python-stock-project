from threading import Thread

from flask import Flask, request, abort

from src.controllers.controller import Controller
from src.mappers import to_predict_request

app = Flask(__name__)


@app.route('/predict/<model>/<stock_symbol>', methods=['POST'])
def predict(model, stock_symbol):
    content = request.get_json(silent=True)

    mapped_request = to_predict_request(model, stock_symbol, content)
    if isinstance(mapped_request, Exception):
        abort(400, mapped_request)

    thread = Thread(target=Controller.predict, args=(mapped_request,))
    thread.start()
    return {"success": "success"}
    experiment_result = Controller.predict(mapped_request)
    if isinstance(experiment_result, Exception):
        abort(500, experiment_result)

    return experiment_result.build()
