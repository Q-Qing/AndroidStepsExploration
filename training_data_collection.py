"""
Collecting ios data from original files.
Integrate date into a standard format and fill the missing data with null.
Write the processed data into csv file.
"""
import os
import pandas as pd
import numpy as np

# root_path = '/Users/Q-Q/data set/step dataset/847471356/steps'


def create_csv_for_user(root_path, result_path, user_id):

    columns = list(range(1440))
    columns.insert(0, "date")
    columns.insert(1, "type")
    columns.insert(2, "steps")
    # print(columns)
    df_ios_steps = pd.DataFrame(columns=columns)
    row = 0
    for dirpath, dirnames, files in os.walk(root_path):
        for file_name in files:
            file_path = os.path.join(dirpath, file_name)
            if ".txt" in file_path:
                print(file_path)
                try:
                    df = pd.read_csv(file_path, sep=' ', header=None)
                    df.columns = ['timestamp', 'steps']
                    df['date'] = pd.to_datetime(df['timestamp'], unit='s') + pd.to_timedelta(8, 'h')
                    df['minutes'] = df['date'].dt.minute + df['date'].dt.hour * 60
                    # print(df)
                    if "android" in file_name:
                        df_ios_steps.loc[row, 'type'] = 'android'
                    else:
                        df_ios_steps.loc[row, 'type'] = 'ios'
                    date = df.loc[df.index[0], 'date'].strftime("%Y/%m/%d")
                    df_ios_steps.loc[row, 'date'] = date
                    df_ios_steps.loc[row, 'steps'] = df.loc[df.index[-1], 'steps']
                    for subrow_index, subrow in df.iterrows():
                        column_name = subrow['minutes']
                        df_ios_steps.loc[df.index[row], column_name] = subrow['steps']
                    print(df_ios_steps)
                    row += 1
                except:
                    continue

    print(df_ios_steps)
    df_ios_steps.to_csv(result_path+user_id+".csv")