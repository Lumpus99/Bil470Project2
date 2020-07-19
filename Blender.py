from sklearn.ensemble import RandomForestClassifier
import numpy as np


# random forest olarak kullaniliyor.Metod ismi degismedi.
def do_blender(rf_predictions_train, rf_predictions_test,
               knn_predictions_train, knn_predictions_test,
               bagging_predictions_train, bagging_predictions_test,
               y_train_blender, y_test):

    x_train = [rf_predictions_train, knn_predictions_train, bagging_predictions_train]
    x_train = np.array(x_train).transpose()

    x_test = [rf_predictions_test, knn_predictions_test, bagging_predictions_test]
    x_test = np.array(x_test).transpose()

    logistic_regression = RandomForestClassifier()
    logistic_regression.fit(x_train, y_train_blender)

    result = logistic_regression.score(x_test, y_test)
    print(result)
