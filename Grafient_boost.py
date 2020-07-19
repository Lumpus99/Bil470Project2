from sklearn.ensemble import GradientBoostingClassifier
import Readfile as Rf


def do_gradient_boost():
    x_train, _, y_train, _, x_test, y_test = Rf.read_data()
    ada_boost = GradientBoostingClassifier()
    ada_boost.fit(x_train, y_train)
    score = ada_boost.score(x_test, y_test)
    print(score)
    Rf.save_model("Gradient2", ada_boost)
