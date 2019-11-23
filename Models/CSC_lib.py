import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class CSC:
    def __init__(self):
        pass

    def load_sets(self, path='Data'):
        test_data = pd.read_csv('{!s}/test_data.csv'.format(path))
        train_answers = pd.read_csv('{!s}/train_answers.csv'.format(path))
        train_data = pd.read_csv('{!s}/train_data.csv'.format(path))
        return train_data, train_answers, test_data

    def return_sentence(self, df, sentence_id=1):
        return df[df['id'] == 'sentence_{!s}'.format(sentence_id)]

    def bag_of_words(self, text):
        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform(text)
        feature_names = vectorizer.get_feature_names()
        return bow #, feature_names