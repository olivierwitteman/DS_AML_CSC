from Models import CSC_lib
import pandas as pd

CSC = CSC_lib.CSC()

X_train, y_train, X_test = CSC.load_sets()

# print(X_train.keys())
print(CSC.return_sentence(X_train, 4)[X_train.keys()[1]])

# print(CSC.bag_of_words(['jop is een een een mooie jongen', 'gideon ook is']))

# print(CSC.random_predictions(X_test))
sentence, answer = CSC.random_predictions(X_test)
CSC.export_predictions(sentence, answer)

