from sklearn.linear_model import LogisticRegression
import Readfile as Rf

x_train, x_test, y_train, y_test = Rf.read_data()
random_forest = LogisticRegression()
random_forest.fit(x_train, y_train)
predictions = random_forest.predict(x_test)
Rf.calc_success(predictions, y_test)
