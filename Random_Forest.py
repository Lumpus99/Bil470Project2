from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split


train_target = pd.read_csv('data/train_labels.csv')['damage_grade']
##real_values = pd.read_csv('data/submission_format.csv')['damage_grade'].to_numpy()

le = preprocessing.LabelEncoder()

cols = tuple(range(1, 39))

train_data = np.genfromtxt('data/train_values.csv', delimiter=",", dtype="|a20", skip_header=1, usecols=cols)
train_target=np.array(train_target)

# 39 = attribute number
for i in range(38):
    train_data[:, i] = le.fit_transform(train_data[:, i])
#test_values = np.genfromtxt('data/test_values.csv', delimiter=",", dtype="|a20", skip_header=1, usecols=cols)

print(type(train_data), train_data)

X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.20, random_state=42)
##for i in range(38):
    ##test_values[:, i] = le.fit_transform(test_values[:, i])

#test_values = test_values
#test_values = np.array(test_values)


rf = RandomForestClassifier()

rf.fit(X_train,y_train)

predictions = rf.predict(X_test)

##print("pred: ", len(predictions), "real:", len(y_test))
correct = 0
wrong = 0

for pred, y_test in zip(predictions, y_test):
    print("pred: ", pred, "real:",y_test)
    if pred == y_test:
        correct += 1
    else:
        wrong += 1
print(correct/(correct+wrong))

