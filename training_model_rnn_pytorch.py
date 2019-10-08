from torch import nn
import torch
import pandas as pd
from math import ceil
from torch.utils.data import TensorDataset, DataLoader
import time
import numpy as np

df = pd.read_csv("data/ios_training_labels.csv")
df = df.fillna(0)
# shuffle the rows of data frame
df = df.sample(frac=1).reset_index(drop=True)
print(df)
# split training dataset and testing dataset
rows = df.shape[0]
training_size = 0.75
df_training = df.iloc[0:ceil(rows*training_size), :]
df_testing = df.iloc[ceil(rows*training_size):, :]
# transfer data frame to numpy arrays
train_data = df_training.iloc[:, 1:1440].to_numpy()
train_labels = df_training.iloc[:, 1440].to_numpy()
test_data = df_testing.iloc[:, 1:1440].to_numpy()
test_labels = df_testing.iloc[:, 1440].to_numpy()
train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)


def one_hot_encode(lables, batch_size, output_size):
    features = np.zeros((batch_size, output_size), dtype=np.float32)
    i = 0
    for item in lables:
        features[i, item] = 1
        i += 1
    return features


# print(train_labels)
# train_labels = one_hot_encode(train_labels, train_labels.shape[0], 27)
# print(train_labels.shape)

train_data = TensorDataset(torch.from_numpy(train_data).double(), torch.from_numpy(train_labels))
test_data = TensorDataset(torch.from_numpy(test_data).double(), torch.from_numpy(test_labels))

batch_size = 100
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)


class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, drop_prob=0.2):
        super(RNNModel, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

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

        # out = out.view(batch_size, -1)
        # print(out.size())
        # out = out[:, -1]
        # print(out.size())

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim, dtype=torch.double)
        return hidden


# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

input_size = 1
output_size = 27
hidden_dim = 24
n_layers = 1


model = RNNModel(input_size, output_size, hidden_dim, n_layers)
model.to(device)
model.double()

lr = 0.005
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
n_epochs = 10

# for epoch in range(1, n_epochs + 1):
#     optimizer.zero_grad()  # Clears existing gradients from previous epoch
#     train_data = (torch.tensor(train_data, dtype=torch.double))
#     train_data.to(device)
#     print(train_data.size(0), train_data.size(1))
#     output, hidden = model(train_data)
#     print(output.size(0), output.size(1))
#     loss = criterion(output, train_labels.view(-1))
#     loss.backward()  # Does backpropagation and calculates gradients
#     optimizer.step()  # Updates the weights accordingly

model.train()
print("Starting Training of RNN model")
epoch_times = []
# Start training loop
for epoch in range(1, n_epochs+1):
    start_time = time.clock()
    h = model.init_hidden(batch_size)
    avg_loss = 0.
    counter = 0
    for inputs, label in train_loader:
        counter += 1
        # h = h.data
        # print(h)
        model.zero_grad()
        output, h = model(inputs.double())
        # print(output)
        # print(label)
        loss = criterion(output.squeeze(), label.long())
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        if counter % 200 == 0:
            print(
                "Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader),
                                                                                     avg_loss / counter))
    current_time = time.clock()
    print("Epoch {}/{} Done, Total Loss: {}".format(epoch, n_epochs, avg_loss / len(train_loader)))
    print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
    epoch_times.append(current_time - start_time)
print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

test_losses = []
num_correct = 0
h = model.init_hidden(batch_size)

model.eval()
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
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

test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc*100))



