import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from itertools import product

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)

"""
df = pd.read_csv("/Users/Q-Q/data set/step dataset/detailed_steps/902162251.csv")
df['date'] = pd.to_datetime(df['date'])
# df.sort_values('date', inplace=True)
print(df)
"""

"""
# processing android data 
df_android = pd.read_csv("/Users/Q-Q/data set/step dataset/detailed_steps/63029.csv")
# print(df_android)
for index, row in df_android.iterrows():
    last_value = -1
    steps_list = []
    steps = 0
    for i in range(4, 1444):
        if not np.isnan(row[i]):
            if last_value == -1:
                df_android.loc[index, i] = 0
            elif row[i] >= last_value:
                steps += (row[i] - last_value)
            else:
                steps += row[i]

            df_android.iloc[index, i] = steps
            last_value = row[i]
            steps_list.append(steps)
    # print(steps_list)

    df_android.loc[index, 'steps'] = steps

print(df_android)
"""


def dataframe_completion(df_input):
    df_copy = df_input.copy()
    df_copy.iloc[:, 4:] = df_copy.iloc[:, 4:].interpolate(axis=1)
    df_copy = df_copy.fillna(0)
    drop_index = []
    for index, row in df_copy.iterrows():
        total_steps = row['steps']
        steps = row[4:].values.tolist()
        print(index)
        normalization_list = np.array(steps) / total_steps
        # print(normalization_list)
        df_copy.iloc[index, 4:] = normalization_list
        last_value = 0
        for i in range(1440):
            if last_value <= normalization_list[i] <= 1:
                last_value = normalization_list[i]
            else:
                # print("4 should be droped")
                drop_index.append(index)
                break
    # print(drop_index)
    df_copy = df_copy.drop(drop_index)
    return df_copy


def plot_completion(df_com, num):

    subdf = df_com.iloc[0:num, :]
    fig, ax = plt.subplots(figsize=(16, 8))
    for index, row in subdf.iterrows():
        total_steps = row['steps']
        date = row['date']
        type = row['type']
        x = list(range(1440))
        y = row[4:].values.tolist()
        # print(y)
        line = ax.plot(x, y, label=str(date)+":"+str(total_steps)+" "+type)
        ax.set_xlabel("minutes")
        ax.set_ylabel("rate")
        ax.legend()

    plt.show()


def mean_completion_cal(df_com):
    num_row = df_com.shape[0]
    sub_df = df_com.iloc[:, 4:]
    mean_list = np.array(sub_df.sum(axis=0).values.tolist())
    mean_list = mean_list/num_row
    fig, ax = plt.subplots(figsize=(16, 8))
    x = list(range(1440))
    y = mean_list
    line = ax.plot(x, y, label="mean and num="+str(num_row))
    ax.set_xlabel("minutes")
    ax.set_ylabel("rate")
    ax.legend(loc=2)
    plt.show()
    return mean_list


def up_state_statistic(df_com):
    """
    calculate up state of each curve
    :param df_com: data frame of completion
    :return:
    """
    df_inner = df_com.iloc[:, 4:]
    df2 = df_com.iloc[:, 4:].diff(axis=1)
    df2 = df2.fillna(0)
    up_state_total = []
    up_state_time = []
    up_state_range = []
    up_state_duration = []
    print(df2)
    row_index = 0
    # find everyday's up state
    for index, row in df2.iterrows():
        up_state = []
        # find consecutive non-zero difference
        for i in range(1440-1):
            if row[i] == 0 and row[i+1] != 0:
                start_index = i+1
            if row[i] != 0 and row[i+1] == 0:
                end_index = i
                up_state.append(start_index)
                up_state.append(end_index)
        """
        print("first step: find consecutive increment")
        print(up_state)
        print(df_com.iloc[index, up_state].values.tolist())
        """
        # merge adjacent up states into one
        up_state2 = up_state.copy()
        for index1 in np.arange(1, (len(up_state) - 2), 2):
            end_point = up_state[index1]
            next_start_point = up_state[index1 + 1]
            if (next_start_point - end_point) < 30:
                up_state2.remove(end_point)
                up_state2.remove(next_start_point)
        """
        print("second step: combine adjacent up state")
        print(up_state2)
        print(df_com.iloc[index, up_state2].values.tolist())
        """
        # delete up state if the range of up state is less than 0.1
        up_state3 = up_state2.copy()
        print(index)
        for j in np.arange(0, len(up_state2), 2):
            if df_inner.iloc[row_index, up_state2[j+1]] - df_inner.iloc[row_index, up_state2[j]] < 0.1:
                up_state3.remove(up_state2[j])
                up_state3.remove(up_state2[j+1])
        """
        print("third steps: delete small up state")
        print(up_state3)
        print(df_com.iloc[index, up_state3].values.tolist())
        """
        # up state statistics
        for m in np.arange(0, len(up_state3), 2):
            time = round((up_state3[m+1] + up_state3[m])/120)
            duration = round((up_state3[m+1] - up_state3[m])/30)
            up_range = round(df_inner.iloc[row_index, up_state3[m+1]] - df_inner.iloc[row_index, up_state3[m]], 1)
            up_state_time.append(time)
            up_state_range.append(up_range)
            up_state_duration.append(duration)
        up_state_total.append(up_state3)

        row_index += 1
    up_state_duration = np.array(up_state_duration)*0.5
    return up_state_total, up_state_time, up_state_range, up_state_duration


def init_classifier(com_list, up_point_list):
    """
    initialization of classifier
    :param com_list: a list of completion
    :param up_point_list: a list of time point
    :return final_classifier_list: two-dimensional list. Each sublist represents a classifier in detail.
    :return classifier_list: two-dimensional list. Each sublist represents a classifier in brief.
    """
    classifier_list = []
    list_len = len(up_point_list)
    for item in product(com_list, repeat=list_len):
        add_flag = True
        last_value = 0
        for i in range(list_len):
            if last_value <= item[i] <= 1:
                last_value = item[i]
            else:
                add_flag = False
                break
        if add_flag:
            classifier_list.append(list(item))

    final_classifier_list = []
    for classifier in classifier_list:
        final_classifier = np.array([0.0]*1440)
        # set the last hour of the day is always 1
        final_classifier[1380:] = 1
        for j in range(list_len):
            start_index = round(up_point_list[j]*60)
            if j == list_len - 1:
                end_index = 1380
            else:
                end_index = round(up_point_list[j+1]*60)
            final_classifier[start_index:end_index] = classifier[j]

        final_classifier_list.append(final_classifier)

    return final_classifier_list, classifier_list


def label_completion(df_com, classifier_center):
    """
    calculate Euclidean distance between sample and classifier_center
    :param df_com: data frame of completion
    :param classifier_center: list of initialized classifier
    :return labels: a list of labels which are response to samples
    """
    labels = []
    for index, row in df_com.iterrows():
        dist = []
        for one_classifier in classifier_center:
            sample = np.array([row[4:].values.tolist()])
            distance = np.linalg.norm(sample - one_classifier)
            dist.append(distance)
        print(dist)
        sample_label = dist.index(min(dist))
        labels.append(sample_label)
    return labels


# test part
# df_completion = dataframe_completion(df)
# print(df_completion)
# plot_completion(df_completion, num=5)
# m_list = mean_completion_cal(df_completion)
# total_list, time_list, range_list, duration_list = up_state_statistic(df_completion)
# print(total_list)
# print(Counter(time_list))
# print(Counter(range_list))
# print(Counter(duration_list))
# final_classifier_result, classifier_result = init_classifier(com_list=
# [0, 0.25, 0.5, 0.75, 1], up_point_list=[8, 12, 16, 20])
final_classifier_result, classifier_result = init_classifier(com_list=
[0, 0.25, 0.5, 0.75, 1], up_point_list=[8, 20])
print(len(classifier_result))
print(classifier_result[11])
print(classifier_result[13])
print(classifier_result[8])
print(classifier_result[0])
print(classifier_result[7])
# for i in final_classifier_result:
#     print(i.tolist())
# print(final_classifier_result)
# df_completion = pd.read_csv("/Users/Q-Q/Downloads/ios_completion.csv")
# df_com = df_completion.iloc[0:5000, :]
# df_com.to_csv("ios_completion.csv", index=False)
# labelss = label_completion(df_com, final_classifier_result)
# print(labelss)
# print(Counter(labelss))
# se_labels = pd.Series(labelss)
# se_labels.to_csv("labeled_data.csv", index=False, header="category")
