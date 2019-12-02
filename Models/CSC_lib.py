import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time
import numpy as np


class CSC:
    def __init__(self):
        """
        Creates an instance of the PreProcess class in this global class.
        """
        self.PreProcess = PreProcess()

    def load_sets(self, path='Data'):
        """
        :param path: Defaults to the Data folder from github.
        :return: Pandas DataFrame of training data, training answers and the test data.
        """
        test_data = pd.read_csv('{!s}/test_data.csv'.format(path))
        train_answers = pd.read_csv('{!s}/train_answers.csv'.format(path))
        train_data = pd.read_csv('{!s}/train_data.csv'.format(path))
        return train_data, train_answers, test_data

    def return_sentence(self, df, sentence_id=1):
        """
        :param df: Pandas DataFrame of the desired set.
        :param sentence_id: Integer of the desired sentence from the set.
        :return: Pandas DataFrame of the complete line (for the training data this would be ['id', 'FalseSent',
         'OptionA', 'OptionB', 'OptionC']).
        """
        return df[df['id'] == 'sentence_{!s}'.format(sentence_id)]

    def bag_of_words(self, text):
        """

        :param text: List of strings.
        :return: Bag of words.
        """
        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform(text)
        feature_names = vectorizer.get_feature_names()
        return bow #, feature_names

    def random_predictions(self, testset):
        """

        :param testset: Testset is used to identify the length of the required random prediction. It is not
         actually used for predictions, hence the name 'random_predictions'.
        :return: Random A, B or C for each sentence in the test set.
        """
        options = ['A', 'B', 'C']
        answer, sentence = [], []
        for i in range(len(testset)):
            sentence.append(str(i+1))
            answer.append(options[int(np.random.rand() * 3)])
        return sentence, answer

    def export_predictions(self, sentence, answer, path='Data'):
        """

        :param sentence: sentence #
        :param answer: option A, B or C
        :param path: Defaults to the Data folder from github.
        :return: nothing
        """
        with open('{!s}/{!s}.csv'.format(path, time.time()), 'a') as a:
            a.write('id,answer\n')
            for i in range(len(sentence)):
                a.write('sentence_{!s},{!s}\n'.format(sentence[i], answer[i]))


class PreProcess:
    def __init__(self):
        pass 

    def hello(self):
        print("hello world:)")
