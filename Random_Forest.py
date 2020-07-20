from sklearn.ensemble import RandomForestClassifier


import Readfile as Rf



def do_random_forest():
    x_train, _, y_train, _, x_test, y_test = Rf.read_data()
    random_forest = RandomForestClassifier()
    random_forest.fit(x_train, y_train)
    score = random_forest.score(x_test, y_test)
    print(score)
    Rf.save_model("Random_Forest2", random_forest)


def do_final_rf():
    x_train, _, y_train, _, _ = Rf.read_data_final()
    random_forest = RandomForestClassifier()
    random_forest.fit(x_train, y_train)
    Rf.save_model("Random_Forest", random_forest)


def do_random_forest_with_k():
    random_forest = RandomForestClassifier()
    Rf.k_validation(random_forest)

def do_random_forest_with_f1():
    random_forest = RandomForestClassifier()
    Rf.f1_validation(random_forest, 'micro')
    Rf.f1_validation(random_forest, 'macro')
    Rf.f1_validation(random_forest, 'weighted')
    Rf.f1_validation(random_forest, None)