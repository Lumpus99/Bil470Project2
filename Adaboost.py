from sklearn.ensemble import AdaBoostClassifier
import Readfile as Rf


def do_ada_boost():
    x_train, x_test, y_train, y_test = Rf.read_data()
    ada_boost = AdaBoostClassifier()
    ada_boost.fit(x_train, y_train)
    predictions = ada_boost.predict(x_test)
    Rf.calc_success(predictions, y_test)
    Rf.save_model("Ada", ada_boost)
    return predictions
