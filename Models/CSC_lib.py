import pandas as pd


class CSC:
    def __init__(self):
        pass

    def load_sets(self, path='Data'):
        test_data = pd.read_csv('{!s}/test_data.csv'.format(path))
        train_answers = pd.read_csv('{!s}/train_answers.csv'.format(path))
        train_data = pd.read_csv('{!s}/train_data.csv'.format(path))
        return train_data, train_answers, test_data


