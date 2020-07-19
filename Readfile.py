import pandas as pd
from sklearn import preprocessing
from pandas import np
from sklearn.model_selection import train_test_split, KFold
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler


def read_data():
    cols = tuple(range(1, 39))

    train_target = pd.read_csv('data/train_labels.csv')['damage_grade']
    train_data = np.genfromtxt('data/train_values.csv', delimiter=",", dtype="|a20", skip_header=1, usecols=cols)
    train_target = np.array(train_target)

    le = preprocessing.LabelEncoder()

    for i in range(38):
        train_data[:, i] = le.fit_transform(train_data[:, i])

    x_train, x_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.20, random_state=42)
    x_train_layer2, x_train_blender, y_train_layer2, y_train_blender = \
        train_test_split(x_train, y_train, test_size=0.20, random_state=42)

    return x_train_layer2, x_train_blender, y_train_layer2, y_train_blender, x_test, y_test


def save_model(model_name, model):
    filename = 'Models/{}_model.sav'.format(model_name)
    file = open(filename, 'wb')
    pickle.dump(model, file)
    file.close()


def read_model(model_name):
    filename = 'Models/{}_model.sav'.format(model_name)
    file = open(filename, 'rb')
    loaded_model = pickle.load(file)
    file.close()
    return loaded_model


def k_validation(the_model):
    x, _, y, _, _, _ = read_data()
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x)
    scores = []
    cv = KFold(n_splits=10, random_state=42, shuffle=False)

    for train_index, test_index in cv.split(x_train):
        #print("Train Index: ", train_index, "\n") print("Test Index: ", test_index)
        X_train, X_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[
            test_index]
        the_model.fit(X_train, y_train)
        scores.append(the_model.score(X_test, y_test))
    print(the_model.__str__(), " with score ", np.mean(scores))
