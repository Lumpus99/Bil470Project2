import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.svm import SVC


train_target = pd.read_csv('data/train_labels.csv')['damage_grade']
cols = tuple(range(1, 39))

train_data = np.genfromtxt('data/train_values.csv', delimiter=",", dtype="|a20", skip_header=1, usecols=cols)
test_values = np.genfromtxt('data/test_values.csv', delimiter=",", dtype="|a20", skip_header=1, usecols=cols)
le = preprocessing.LabelEncoder()

for i in range(38):
    train_data[:, i] = le.fit_transform(train_data[:, i])

for i in range(38):
    test_values[:, i] = le.fit_transform(test_values[:, i])
test_values = test_values
test_values = np.array(test_values)

svc = SVC(kernel='linear')
print(type(train_data), train_data)
# Makineyi eğitiyoruz
svc.fit(train_data, train_target)

print(svc.predict(test_values))
