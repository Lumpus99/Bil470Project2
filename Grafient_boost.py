from sklearn.ensemble import GradientBoostingClassifier
import Readfile as Rf


def do_gradient_boost():
    x_train, _, y_train, _, x_test, y_test = Rf.read_data()
    ada_boost = GradientBoostingClassifier()
    ada_boost.fit(x_train, y_train)
    score = ada_boost.score(x_test, y_test)
    print(score)
    Rf.save_model("Gradient2", ada_boost)


def do_gradient_boost_with_k():
    ada_boost = GradientBoostingClassifier()
    Rf.k_validation(ada_boost)

def do_gradient_boost_with_f1():
    ada_boost = GradientBoostingClassifier()
    Rf.f1_validation(ada_boost, 'micro')
    Rf.f1_validation(ada_boost, 'macro')
    Rf.f1_validation(ada_boost, 'weighted')
    Rf.f1_validation(ada_boost, None)