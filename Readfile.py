import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR


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

def k_validation():
    x_train, _, y_train, _, _, _ = read_data()
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x_train)
    scores = []
    best_svr = SVR(kernel='rbf')
    cv = KFold(n_splits=20, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(x_train):
        print("Train Index: ", train_index, "\n")
        print("Test Index: ", test_index)
        X_train, X_test, y_train, y_test = x_train[train_index], x_train[test_index], y_train[train_index], y_train[
            test_index]
        best_svr.fit(X_train, y_train)
        scores.append(best_svr.score(X_test, y_test))
    return np.mean(scores)