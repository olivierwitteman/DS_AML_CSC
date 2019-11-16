from Models import CSC_lib

CSC = CSC_lib.CSC()

X_train, y_train, X_test = CSC.load_sets()


print(X_train)
