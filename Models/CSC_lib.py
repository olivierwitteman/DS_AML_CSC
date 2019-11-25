import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time
import numpy as np


class CSC:
    def __init__(self):
        self.PreProcess = PreProcess()

    def load_sets(self, path='Data'):
        test_data = pd.read_csv('{!s}/test_data.csv'.format(path))
        train_answers = pd.read_csv('{!s}/train_answers.csv'.format(path))
        train_data = pd.read_csv('{!s}/train_data.csv'.format(path))
        return train_data, train_answers, test_data

    def return_sentence(self, df, sentence_id=1):
        self.PreProcess.hello()
        return df[df['id'] == 'sentence_{!s}'.format(sentence_id)]

    def bag_of_words(self, text):
        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform(text)
        feature_names = vectorizer.get_feature_names()
        return bow #, feature_names

    def random_predictions(self, testset):
        options = ['A', 'B', 'C']
        answer, sentence = [], []
        for i in range(len(testset)):
            sentence.append(str(i+1))
            answer.append(options[int(np.random.rand() * 3)])
        return sentence, answer

    def export_predictions(self, sentence, answer, path='Data'):
        with open('{!s}/{!s}.csv'.format(path, time.time()), 'a') as a:
            a.write('id,answer\n')
            for i in range(len(sentence)):
                a.write('sentence_{!s},{!s}\n'.format(sentence[i], answer[i]))


class PreProcess:
    def __init__(self):
        pass 

    def hello(self):
        print("hello world:)")
