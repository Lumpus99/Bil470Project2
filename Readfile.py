import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split


def read_data():
    cols = tuple(range(1, 39))

    train_target = pd.read_csv('data/train_labels.csv')['damage_grade']
    train_data = np.genfromtxt('data/train_values.csv', delimiter=",", dtype="|a20", skip_header=1, usecols=cols)
    train_target = np.array(train_target)

    le = preprocessing.LabelEncoder()

    for i in range(38):
        train_data[:, i] = le.fit_transform(train_data[:, i])

    x_train, x_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.20, random_state=42)
    return x_train, x_test, y_train, y_test


def calc_success(predictions, y_test):
    correct = 0
    wrong = 0

    for pred, y_test in zip(predictions, y_test):
        print("pred: ", pred, "real:", y_test)
        if pred == y_test:
            correct += 1
        else:
            wrong += 1
    print(correct / (correct + wrong))
