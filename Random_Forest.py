from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import preprocessing
import numpy as np

train_target = pd.read_csv('data/train_labels.csv')['damage_grade']
real_values = pd.read_csv('data/submission_format.csv')['damage_grade'].to_numpy()

le = preprocessing.LabelEncoder()

cols = tuple(range(1, 39))

train_data = np.genfromtxt('data/train_values.csv', delimiter=",", dtype="|a20", skip_header=1, usecols=cols)
test_values = np.genfromtxt('data/test_values.csv', delimiter=",", dtype="|a20", skip_header=1, usecols=cols)

print(type(train_data), train_data)

# 39 = attribute number
for i in range(38):
    train_data[:, i] = le.fit_transform(train_data[:, i])

for i in range(38):
    test_values[:, i] = le.fit_transform(test_values[:, i])

test_values = test_values
test_values = np.array(test_values)


rf = RandomForestClassifier()

rf.fit(train_data, train_target)

predictions = rf.predict(test_values)

print("pred: ", len(predictions), "real:", len(real_values))

correct = 0
wrong = 0

for pred, real in zip(predictions, real_values):
    print("pred: ", pred, "real:", real)
    if pred == real:
        correct += 1
    else:
        wrong += 1
print(correct/(correct+wrong))


