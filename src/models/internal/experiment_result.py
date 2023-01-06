class ExperimentResult:
    def __init__(self, test_results=None, predictions=None):
        self.test_results = test_results
        self.predictions = predictions

    def with_test_results(self, test_results):
        self.test_results = test_results
        return self

    def with_predictions(self, predictions):
        self.predictions = predictions
        return self

    def build(self):
        return {
            "predictions": self.predictions.to_json(),
            "test_results": self.test_results.to_json(),
        }
