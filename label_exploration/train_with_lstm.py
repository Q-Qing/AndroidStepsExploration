import pandas as pd
import numpy as np
from math import ceil
from torch.utils.data import TensorDataset, DataLoader
import torch
from traning_model_lstm import LstmModel
from traning_model_lstm import model_train,model_test
import os


data = pd.read_csv("ios_completion.csv")
labels = pd.read_csv("labeled_data.csv")
# print(labels)
# labels_list = labels.values.tolist()
# print(labels_list)
keep_index = []
for index, row in labels.iterrows():
    # print(row[0])
    if (row[0] == 0) or (row[0] == 49):
        keep_index.append(index)
        # print(keep_index)

print(len(keep_index))

data = data.iloc[keep_index]
labels = labels.iloc[keep_index]
labels = labels.replace(49, 16)


print(data)
print(labels)

batch_size = 100

train_data = data.iloc[0:1000, 4:].to_numpy()
train_labels = labels.iloc[0:1000].to_numpy()
test_data = data.iloc[1000:, 4:].to_numpy()
test_labels = labels.iloc[1000:].to_numpy()


train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)

train_data = TensorDataset(torch.from_numpy(train_data).double(), torch.from_numpy(train_labels).squeeze())
test_data = TensorDataset(torch.from_numpy(test_data).double(), torch.from_numpy(test_labels).squeeze())

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

input_size = 1
output_size = 40
hidden_dim = 100
n_layers = 2

model = LstmModel(input_size, output_size, hidden_dim, n_layers)
model.to(device)
model.double()

criterion = torch.nn.CrossEntropyLoss()
model_train(train_loader, model, n_epochs=20, lr=0.005)
model_test(test_loader, model, criterion, device)



