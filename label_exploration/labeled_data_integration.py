import pandas as pd
import os
from label_exploration.lable_a import dataframe_completion


data_path = "/Users/Q-Q/data set/step dataset/detailed_steps/"

# columns = list(range(1440))
# columns.insert(0, "user_id")
# columns.insert(1, "date")
# columns.insert(2, "type")
# columns.insert(3, "steps")
# print(columns)
df_integrated = pd.DataFrame()

for dirpath, dirname, files in os.walk(data_path):
    total_row = 0
    for file in files:
        if ".csv" in file:
            file_path = os.path.join(dirpath, file)
            user_id = file[0:-4]
            print(file_path)
            # print(user_id)
            try:
                df = pd.read_csv(file_path)
                df_ios = df.loc[df['type'] == "ios"]
                num = df_ios.shape[0]
                df_ios.insert(1, column='user_id', value=[user_id]*num)
                # print(df_ios)
                df_integrated = df_integrated.append(df_ios.iloc[:, 1:], ignore_index=True)
                # print(df_integrated)
                total_row += num
                if total_row > 5000:
                    break
            except:
                continue

print(df_integrated)
df_com = dataframe_completion(df_integrated)
print(df_com)
df_com.to_csv("ios_completion.csv", index=False)





