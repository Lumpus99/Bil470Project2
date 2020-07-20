from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
train_values = pd.read_csv('data/train_values.csv', index_col='building_id')
train_labels = pd.read_csv('data/train_labels.csv', index_col='building_id')

print(len(train_values))
print(train_values.dtypes)
print(train_values.columns)
size=len(train_values.dtypes)
print(train_values.dtypes)
print(len(train_values))

(train_labels.damage_grade
             .value_counts()
             .sort_index()
             .plot.bar(title="Number of Buildings with Each Damage Grade"))
liste=[]

for x in range (size):
    column_name=train_values.columns[x]
    farkli_eleman=train_values[column_name].unique()
    size=len(farkli_eleman)
    liste.append([column_name,size])
print(liste)

for x in range(len(liste)):
    print(liste[x])

print(len(liste))

(train_labels.damage_grade
             .value_counts()
             .sort_index()
             .plot.bar(title="Number of Buildings with Each Damage Grade"))
train_labels.info()
train_values.describe()
train_values.describe(include=np.object)
selected_features = ['foundation_type',
                     'age',
                     'roof_type',
                     'ground_floor_type',
                     'other_floor_type',
                     'position',
                     'plan_configuration',
                     'area_percentage',
                     'height_percentage',
                     'count_floors_pre_eq',
                     'count_families',
                     'land_surface_condition',]

train_values_subset = train_values[selected_features]
sns.pairplot(train_values_subset.join(train_labels),
             hue='damage_grade')
(train_values.roof_type
             .value_counts()
             .sort_index()
             .plot.bar(title="Number of Buildings with Root Type"))
(train_values.geo_level_2_id
             .value_counts()
             .sort_index()
             .plot.bar(title="Number of Buildings with Each Damage Grade"))
(train_values.age
             .value_counts()
             .sort_index()
             .plot.bar(title="Number of Buildings with Each Damage Grade"))


