from sklearn.ensemble import RandomForestClassifier
import Readfile as Rf


def do_random_forest():
    x_train, x_test, y_train, y_test = Rf.read_data()
    random_forest = RandomForestClassifier()
    random_forest.fit(x_train, y_train)
    predictions = random_forest.predict(x_test)
    Rf.calc_success(predictions, y_test)
    Rf.save_model("Random_Forest", random_forest)
    return predictions
