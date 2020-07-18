from sklearn.linear_model import LogisticRegression
import Readfile as Rf
import numpy as np


def do_logistic_regression(ada, rf, svm):
    print("LR")
    x_train = np.array(ada, rf, svm)
    print(x_train)
    _, x_test, y_train, y_test = Rf.read_data()
    print("asad")
    random_forest = LogisticRegression()
    print("asad1")
    random_forest.fit(x_train, y_train)
    print("asad2")
    predictions = random_forest.predict(x_test)
    Rf.calc_success(predictions, y_test)
