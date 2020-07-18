from sklearn.linear_model import LogisticRegression
import Readfile as Rf
import numpy as np


def do_logistic_regression(ada_train, ada_test, rf_train, rf_test, svm_train, svm_test, y_train, y_test):
    x_train = [ada_train, rf_train, svm_train]
    x_train = np.array(x_train).transpose()

    x_test = [ada_test, rf_test, svm_test]
    x_test = np.array(x_test).transpose()

    logistic_regression = LogisticRegression()
    logistic_regression.fit(x_train, y_train)

    result = logistic_regression.score(x_test, y_test)
    print(result)
