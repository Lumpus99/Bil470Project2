from sklearn.ensemble import AdaBoostClassifier
import Readfile as Rf


def do_ada_boost():
    x_train, x_test, y_train, y_test, _, _ = Rf.read_data()
    ada_boost = AdaBoostClassifier()
    ada_boost.fit(x_train, y_train)
    score = ada_boost.score(x_test, y_test)
    print(score)
    Rf.save_model("Ada2", ada_boost)
