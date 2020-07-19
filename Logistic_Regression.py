from sklearn.ensemble import RandomForestClassifier
import Readfile as Rf
import numpy as np


def do_logistic_regression(rf_predictions_train, rf_predictions_test,
                           svm_predictions_train, svm_predictions_test,
                           ada_predictions_train, ada_predictions_test,
                           y_train_blender, y_test):

    x_train = [rf_predictions_train, svm_predictions_train, ada_predictions_train]
    x_train = np.array(x_train).transpose()

    x_test = [rf_predictions_test, svm_predictions_test, ada_predictions_test]
    x_test = np.array(x_test).transpose()

    logistic_regression = RandomForestClassifier()
    logistic_regression.fit(x_train, y_train_blender)

    result = logistic_regression.score(x_test, y_test)
    print(result)
