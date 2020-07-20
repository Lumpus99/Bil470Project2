
from sklearn.ensemble import AdaBoostClassifier

import Readfile as Rf


def do_ada_boost():
    x_train, _, y_train, _, x_test, y_test = Rf.read_data()
    ada_boost = AdaBoostClassifier()
    ada_boost.fit(x_train, y_train)
    score = ada_boost.score(x_test, y_test)
    print(score)
    Rf.save_model("Ada2", ada_boost)


def do_ada_boost_with_k():
    ada_boost = AdaBoostClassifier()
    Rf.k_validation(ada_boost)

def do_ada_boost_with_f1():
    ada_boost = AdaBoostClassifier()
    Rf.f1_validation(ada_boost,'micro')
    Rf.f1_validation(ada_boost, 'macro')
    Rf.f1_validation(ada_boost, 'weighted')
    Rf.f1_validation(ada_boost, None)