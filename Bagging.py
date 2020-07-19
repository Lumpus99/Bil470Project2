from pandas import np
from sklearn.ensemble import BaggingClassifier

import Readfile as Rf


def do_Bagging():
    x_train, _, y_train, _, x_test, y_test = Rf.read_data()
    bagging = BaggingClassifier()
    bagging.fit(x_train, y_train)
    score = bagging.score(x_test, y_test)
    print(score)
    Rf.save_model("Bagging2", bagging)


def do_Bagging_with_k():
    bagging = BaggingClassifier()
    Rf.k_validation(bagging)
