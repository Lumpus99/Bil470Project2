import Readfile as Rf
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def do_svm():
    x_train, x_test, y_train, y_test, _, _ = Rf.read_data()

    scaler = StandardScaler()
    x_train = scaler.fit(x_train).transform(x_train)

    svc = SVC(kernel='linear', max_iter=1000)
    svc.fit(x_train, y_train)
    score = svc.score(x_test, y_test)
    print(score)
    Rf.save_model("SVM2", svc)
