from torch import nn
import torch.nn.functional as F
import torch
import pandas as pd
from math import ceil
from torch.utils.data import TensorDataset, DataLoader
import time
import numpy as np
from dataloader import data_loader


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()

        self.fc1 = nn.Linear(1439, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 27)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # print(x)
        out = self.softmax(x)
        # print(out)
        return out


def model_train(train_loader, model, n_epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Starting Training of RNN model")
    for epoch in range(1, n_epochs + 1):
        avg_loss = 0.
        counter = 0
        for inputs, label in train_loader:
            counter += 1
            model.zero_grad()
            output = model(inputs.double())
            loss = criterion(output.squeeze(), label.long())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, n_epochs, avg_loss / len(train_loader)))


def model_test(test_loader, model, criterion, device):
    test_losses = []
    num_correct = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        test_loss = criterion(output.squeeze(), labels.long())
        test_losses.append(test_loss.item())
        result = np.argmax(output.detach().numpy(), axis=1)
        print("original output:", output)
        print("prediction:", result)
        print("real labels:", labels)
        for i in range(len(result)):
            if result[i] == labels[i]:
                num_correct += 1

    test_acc = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.3f}%".format(test_acc * 100))

"""
# load data
train_loader, test_loader = data_loader(path='data/ios_training_labels.csv', training_ratio=0.8, batch_size=100,
                                        data_dim=2)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = MLPModel()
model.to(device)
model.double()

criterion = nn.CrossEntropyLoss()
model_train(train_loader, model, n_epochs=20, lr=0.005)
model_test(test_loader, model, criterion, device)
"""

