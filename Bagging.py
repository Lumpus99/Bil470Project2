from sklearn.ensemble import BaggingClassifier
import Readfile as Rf


def do_Bagging():
    x_train, _, y_train, _, x_test, y_test = Rf.read_data()
    random_forest = BaggingClassifier()
    random_forest.fit(x_train, y_train)
    score = random_forest.score(x_test, y_test)
    print(score)
    Rf.save_model("Bagging2", random_forest)
