import Random_Forest as Rf
import Adaboost as Ab
import svm
import Logistic_Regression as Lr


def do_blending():
    print("Staring Random Forest:")
    rf_predictions = Rf.do_random_forest()

    print("Staring SVM:")
    svm_predictions = svm.do_svm()

    print("Staring AdaBoost:")
    ada_predictions = Ab.do_ada_boost()

    print("Staring Blending:")
    # Lr.do_logistic_regression(ada_predictions, rf_predictions, svm_predictions)


def main():
    do_blending()


if __name__ == '__main__':
    main()

