import pandas as pd
from itertools import product
import numpy as np
import collections

df_data = pd.read_csv('data/ios_training_steps.csv')
df_mark = pd.read_csv('data/ios_training_mark.csv')
print(df_mark)
# create classifier
classifier = product([0, 20, 40], repeat=3)
classifier_list = []
for item in classifier:
    classifier_list.append(item)
print(classifier_list)

# transfer original data into feature vector and calculate it's distance from classifier
labels = []
for row_index, row in df_mark.iterrows():
    # feature_v = []
    f1 = row[1:480].sum()
    f2 = row[481:960].sum()
    f3 = row[961:].sum()
    feature_v = [f1, f2, f3]
    dist_vector = []
    for i in classifier_list:
        dist = np.linalg.norm(np.array(feature_v) - np.array(i))
        dist_vector.append(dist)
    label = dist_vector.index(min(dist_vector))
    labels.append(label)

print(labels)
counter = collections.Counter(labels)
print(counter)
df_data['label'] = pd.Series(labels, index=df_data.index)
print(df_data)
# df_data.to_csv('data/ios_training_labels.csv', index=False)
