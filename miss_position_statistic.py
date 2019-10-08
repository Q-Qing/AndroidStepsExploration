import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def effective_data_frequency_cal(df):
    df = df.fillna(-1)
    freq_list = [0] * 1440
    rows = df.shape[0]
    android_freq = 0
    ios_freq = 0
    for row_index, row in df.iterrows():
        # print(row)
        if row['type'] == 'android':
            android_freq += 1
            for i in range(1440):
                if row[i + 4] != -1:
                    freq_list[i] = freq_list[i] + 1
        else:
            ios_freq += 1
        # for i in range(1440):
        #     if row[i+4] != -1:
        #         freq_list[i] = freq_list[i] + 1
        freq_list_np = np.array(freq_list)

    return freq_list_np, android_freq
    # return freq_list, rows, android_freq, ios_freq


path = '/Users/Q-Q/data set/step dataset/detailed_steps/'
file_path = '/Users/Q-Q/data set/step dataset/detailed_steps/03474.csv'
final_list = np.array([0] * 1440)
total_android = 0
"""
for dirpath, dirnames, files in os.walk(path):
    for file_name in files:
        file_path = os.path.join(dirpath, file_name)
        print(file_path)
        if ".csv" in file_name:
            try:
                # user_id = file_name[0:-4]
                df_data = pd.read_csv(file_path)
                print(df_data)
                android_freq_list,  android_rows = effective_data_frequency_cal(df_data)
                print(android_freq_list)
                final_list = android_freq_list + final_list
                total_android += android_rows
                print(final_list)
                print(total_android)
                # print(rows)
                # print(android_rows)
                # print(ios_freq)

            except:
                continue
"""
df = pd.read_csv(file_path)
final_list, total_android = effective_data_frequency_cal(df)
plt.subplots()
plt.plot(final_list)
plt.title("android:"+str(total_android))
plt.show()
