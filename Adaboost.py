from sklearn.ensemble import AdaBoostClassifier
import Readfile as Rf


def do_ada_boost():
    x_train, x_test, y_train, y_test = Rf.read_data()
    random_forest = AdaBoostClassifier()
    random_forest.fit(x_train, y_train)
    predictions = random_forest.predict(x_test)
    Rf.calc_success(predictions, y_test)
    return predictions
