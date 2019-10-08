from torch import nn
import torch
import pandas as pd
from math import ceil
from torch.utils.data import TensorDataset, DataLoader
import time
import numpy as np
from dataloader import data_loader


class LstmModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, drop_prob=0.0):
        super(LstmModel, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # RNN Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.lstm(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        # print("out put of rnn:", out.size())
        out = out[:, -1, :]
        out = out.contiguous().view(-1, self.hidden_dim)
        # print("out put of rnn2:", out.size())
        out = self.dropout(out)
        # print("after drop out:", out.size())
        out = self.fc(out)
        # print("after full connection:", out.size())
        out = self.softmax(out)
        # print("after softmax:", out.size())

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        is_cuda = torch.cuda.is_available()

        if is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        # hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim, dtype=torch.double)
        return hidden


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
            output, h = model(inputs.double())
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
        output, h = model(inputs)
        test_loss = criterion(output.squeeze(), labels.long())
        test_losses.append(test_loss.item())
        result = np.argmax(output.detach().numpy(), axis=1)
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
                                        data_dim=3)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

input_size = 1
output_size = 27
hidden_dim = 2
n_layers = 1

model = LstmModel(input_size, output_size, hidden_dim, n_layers)
model.to(device)
model.double()

criterion = nn.CrossEntropyLoss()
model_train(train_loader, model, n_epochs=20, lr=0.05)
model_test(test_loader, model, criterion, device)
"""










