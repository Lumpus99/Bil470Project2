import Random_Forest as Rf
import Adaboost as Ab
import svm
import Blender as Lr
from Readfile import read_model, read_data


def do_blending():
    print("Reading Data:")
    _, x_train_blender, _, y_train_blender, x_test, y_test = read_data()

    print("Staring Random Forest:")
    rf_model = read_model("Random_Forest2")
    rf_predictions_train = rf_model.predict(x_train_blender)
    print(rf_model.score(x_train_blender, y_train_blender))
    rf_predictions_test = rf_model.predict(x_test)

    print("Staring KNN:")
    knn_model = read_model("Knn2")
    knn_predictions_train = knn_model.predict(x_train_blender)
    print(knn_model.score(x_train_blender, y_train_blender))
    knn_predictions_test = knn_model.predict(x_test)

    print("Staring Bagging:")
    bagging_model = read_model("Bagging2")
    bagging_predictions_train = bagging_model.predict(x_train_blender)
    print(bagging_model.score(x_train_blender, y_train_blender))
    bagging_predictions_test = bagging_model.predict(x_test)

    print("Staring Blending:")
    Lr.do_blender(rf_predictions_train, rf_predictions_test,
                  knn_predictions_train, knn_predictions_test,
                  bagging_predictions_train, bagging_predictions_test,
                  y_train_blender, y_test)


def main():
    do_blending()


if __name__ == '__main__':
    main()
