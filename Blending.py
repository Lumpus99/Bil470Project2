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
    rf_predictions = rf_model.predict(x_test)

    print("Staring SVM:")
    svm_model = read_model("SVM")
    svm_predictions = svm_model.predict(x_test)

    print("Staring AdaBoost:")
    ada_model = read_model("Ada")
    ada_predictions = ada_model.predict(x_test)

    print("Staring Blending:")

    Lr.do_logistic_regression(ada_predictions, rf_predictions, svm_predictions, x_test, y_train, y_test)


def main():
    do_blending()


if __name__ == '__main__':
    main()

