from sklearn.ensemble import RandomForestClassifier
from Readfile import create_submission_file
import numpy as np


def do_blender(rf_predictions_train, rf_predictions_test,
               knn_predictions_train, knn_predictions_test,
               bagging_predictions_train, bagging_predictions_test,
               y_train_blender, y_test):
    x_train = [rf_predictions_train, knn_predictions_train, bagging_predictions_train]
    x_train = np.array(x_train).transpose()

    x_test = [rf_predictions_test, knn_predictions_test, bagging_predictions_test]
    x_test = np.array(x_test).transpose()

    random_forest = RandomForestClassifier()
    random_forest.fit(x_train, y_train_blender)

    result = random_forest.score(x_test, y_test)
    print(result)


def do_blender_final(rf_predictions_train, rf_predictions_test,
                     knn_predictions_train, knn_predictions_test,
                     bagging_predictions_train, bagging_predictions_test,
                     y_train_blender):
    x_train = [rf_predictions_train, knn_predictions_train, bagging_predictions_train]
    x_train = np.array(x_train).transpose()

    x_test = [rf_predictions_test, knn_predictions_test, bagging_predictions_test]
    x_test = np.array(x_test).transpose()

    random_forest = RandomForestClassifier()
    random_forest.fit(x_train, y_train_blender)

    result = random_forest.predict(x_test)
    print(result)
    create_submission_file(result)
