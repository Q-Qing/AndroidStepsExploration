import pandas as pd
import numpy as np
from math import ceil
from torch.utils.data import TensorDataset, DataLoader
import torch


def data_loader(path, training_ratio, batch_size, data_dim, fillna=True, scale_flag=True):
    df = pd.read_csv(path)
    if scale_flag:
        data_after_scale = []
        for row_index, row in df.iterrows():
            step_row = []
            max_steps = row[1:1440].max()
            for i in range(1, 1440):
                if row[i] != np.nan:
                    step_after_scale = row[i]/max_steps
                    step_row.append(step_after_scale)
                else:
                    step_row.append(np.nan)
            data_after_scale.append(step_row)
        df.iloc[:, 1:1440] = data_after_scale
        print("after scaling:", df)
    if fillna:
        df = df.fillna(-1)
    print("after fill nan value:", df)

    # split training dataset and testing dataset
    rows = df.shape[0]
    df_training = df.iloc[0:ceil(rows * training_ratio), :]
    df_testing = df.iloc[ceil(rows * training_ratio):, :]
    # transfer data frame to numpy arrays
    train_data = df_training.iloc[:, 1:1440].to_numpy()
    train_labels = df_training.iloc[:, 1440].to_numpy()
    test_data = df_testing.iloc[:, 1:1440].to_numpy()
    test_labels = df_testing.iloc[:, 1440].to_numpy()

    if data_dim == 3:
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)

    train_data = TensorDataset(torch.from_numpy(train_data).double(), torch.from_numpy(train_labels))
    test_data = TensorDataset(torch.from_numpy(test_data).double(), torch.from_numpy(test_labels))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return train_loader, test_loader



