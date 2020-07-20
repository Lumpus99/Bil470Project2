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


def do_Knn_with_k():
    print("K=4")
    knn = KNeighborsClassifier(n_neighbors=4)
    Rf.k_validation(knn)
    print("K=5")
    knn = KNeighborsClassifier(n_neighbors=5)
    Rf.k_validation(knn)
    print("K=10")
    knn = KNeighborsClassifier(n_neighbors=10)
    Rf.k_validation(knn)

def do_Knn_with_f1():
    print("K=4")
    knn = KNeighborsClassifier(n_neighbors=4)
    Rf.f1_validation(knn, 'micro')
    Rf.f1_validation(knn, 'macro')
    Rf.f1_validation(knn, 'weighted')
    print("K=5")
    knn = KNeighborsClassifier(n_neighbors=5)
    Rf.f1_validation(knn, 'micro')
    Rf.f1_validation(knn, 'macro')
    Rf.f1_validation(knn, 'weighted')
    print("K=10")
    knn = KNeighborsClassifier(n_neighbors=10)
    Rf.f1_validation(knn, 'micro')
    Rf.f1_validation(knn, 'macro')
    Rf.f1_validation(knn, 'weighted')