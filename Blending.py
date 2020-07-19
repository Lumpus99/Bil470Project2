import Random_Forest as Rf
import Adaboost as Ab
import svm
import Logistic_Regression as Lr
from Readfile import read_model, read_data


def do_blending():
    print("Reading Data:")
    _, x_train_blender, _, y_train_blender, x_test, y_test = read_data()

    print("Staring Random Forest:")
    rf_model = read_model("Random_Forest2")
    rf_predictions_train = rf_model.predict(x_train_blender)
    print(rf_model.score(x_train_blender, y_train_blender))
    rf_predictions_test = rf_model.predict(x_test)

    print("Staring SVM:")
    svm_model = read_model("SVM2")
    svm_predictions_train = svm_model.predict(x_train_blender)
    print(svm_model.score(x_train_blender, y_train_blender))
    svm_predictions_test = svm_model.predict(x_test)

    print("Staring AdaBoost:")
    ada_model = read_model("Ada2")
    ada_predictions_train = ada_model.predict(x_train_blender)
    print(ada_model.score(x_train_blender, y_train_blender))
    ada_predictions_test = ada_model.predict(x_test)

    print("Staring Blending:")
    Lr.do_logistic_regression(rf_predictions_train, rf_predictions_test,
                              svm_predictions_train, svm_predictions_test,
                              ada_predictions_train, ada_predictions_test,
                              y_train_blender, y_test)


def main():
    do_blending()


if __name__ == '__main__':
    main()
