import numpy as np


class PreProcessor:
    @staticmethod
    def prepare_data(raw_data, sequence_length):
        data, target = [], []
        for idx in range(sequence_length, len(raw_data)):
            start_idx = idx - sequence_length
            data_entry, target_entry = raw_data[start_idx:idx], raw_data[idx]
            data.append(data_entry)
            target.append(target_entry)
        return np.array(data), np.array(target)

    @staticmethod
    def split(data, target, train_test_split=0.1):
        data_index = int((1 - train_test_split) * len(data))
        target_index = int((1 - train_test_split) * len(target))
        train_data, test_data = data[:data_index], data[data_index:]
        train_target = target[:target_index]
        test_target = target[target_index:]
        return train_data, train_target, test_data, test_target
