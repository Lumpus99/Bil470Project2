from sklearn.neighbors import KNeighborsClassifier


import Readfile as Rf


def do_Knn():
    x_train, _, y_train, _, x_test, y_test = Rf.read_data()
    random_forest = KNeighborsClassifier(n_neighbors=5)
    random_forest.fit(x_train, y_train)
    score = random_forest.score(x_test, y_test)
    print(score)
    Rf.save_model("Knn2", random_forest)
