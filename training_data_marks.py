import pandas as pd
import numpy as np

df = pd.read_csv("data/ios_training_steps.csv")
# print(df)

exercise_mark = []
for rowindex, row in df.iterrows():
    mark0 = [0]*1439
    for column_index in range(1,1430):
        if row[column_index] != np.nan and row[column_index+9] != np.nan:
            detal_steps = row[column_index+9] - row[column_index]
            if detal_steps >= 1000:
                for j in range(column_index, column_index+10):
                    mark0[j] = 1

    date = row[0]
    mark0.insert(0, date)
    # print(row.values.tolist())
    # print(mark0)
    exercise_mark.append(mark0)


columns = list(range(1, 1440))
columns.insert(0, "date")
df_mark = pd.DataFrame(data=exercise_mark, columns=columns)
df_mark.to_csv("data/ios_training_mark.csv", index=False)
print(df_mark)


