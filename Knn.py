from sklearn.neighbors import KNeighborsClassifier

import Readfile as Rf


def do_Knn():
    x_train, _, y_train, _, x_test, y_test = Rf.read_data()
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    score = knn.score(x_test, y_test)
    print(score)
    Rf.save_model("Knn2", knn)


def do_final_Knn():
    x_train, _, y_train, _, _ = Rf.read_data_final()
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    Rf.save_model("Knn", knn)
