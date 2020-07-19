from sklearn.ensemble import RandomForestClassifier
import Readfile as Rf


def do_random_forest():
    x_train, x_test, y_train, y_test, _, _ = Rf.read_data()
    random_forest = RandomForestClassifier()
    random_forest.fit(x_train, y_train)
    score = random_forest.score(x_test, y_test)
    print(score)
    Rf.save_model("Random_Forest2", random_forest)
