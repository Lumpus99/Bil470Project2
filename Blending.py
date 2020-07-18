import Random_Forest as Rf
import Adaboost as Ab
import svm
import Logistic_Regression as Lr
from Readfile import read_model, read_data


def do_blending():
    print("Reading Data:")
    x_train, x_test, y_train, y_test = read_data()

    print("Staring Random Forest:")
    rf_model = read_model("Random_Forest")
    rf_predictions_train = rf_model.predict(x_train)
    rf_predictions_test = rf_model.predict(x_test)

    print("Staring SVM:")
    svm_model = read_model("SVM")
    svm_predictions_train = svm_model.predict(x_train)
    svm_predictions_test = svm_model.predict(x_test)

    print("Staring AdaBoost:")
    ada_model = read_model("Ada")
    ada_predictions_train = ada_model.predict(x_train)
    ada_predictions_test = ada_model.predict(x_test)

    print("Staring Blending:")
    Lr.do_logistic_regression(ada_predictions_train, ada_predictions_test,
                              rf_predictions_train, rf_predictions_test,
                              svm_predictions_train, svm_predictions_test,
                              y_train, y_test)


def main():
    do_blending()


if __name__ == '__main__':
    main()
