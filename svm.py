import Readfile as Rf
from sklearn.svm import SVC


x_train, x_test, y_train, y_test = Rf.read_data()
svc = SVC(kernel='linear', cache_size=7000, max_iter=1000)
print("done1")
svc.fit(x_train, y_train)
predictions = svc.predict(x_test)
print(predictions)
Rf.calc_success(predictions, y_test)
