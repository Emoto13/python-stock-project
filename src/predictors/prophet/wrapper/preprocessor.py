class PreProcessor:
    @staticmethod
    def prepare_data(raw_data):
        data = raw_data.reset_index()
        data = data.rename(columns={"date": "ds", "price": "y"})
        return data
