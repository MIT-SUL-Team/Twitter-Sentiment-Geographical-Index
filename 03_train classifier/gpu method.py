# -*- coding: utf-8 -*-
# @File       : gpu method.py
# @Author     : Yuchen Chai
# @Date       : 2022/9/14 9:54
# @Description:

import os
import json
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *
# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# Please adjust the following parameters before executing the code
# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

# Please specify the location you put the source data
DIR_INPUT_TEXT = ''
DIR_INPUT_EMBEDDING = ''

# Please specify the location you store the output data
DIR_OUTPUT = ''

# Neural network parameters
PARA_BATCH_SIZE = 100
PARA_NUM_EPOCHS = 5


def read_in_data(text_path, embed_path):
    df = pd.read_csv(text_path)

    print("Preparing training and test sets")
    with open(os.path.join(text_path, "train_ids.txt"), 'r') as fp:
        train_ids = json.load(fp)
    with open(os.path.join(text_path, "test_ids.txt"), 'r') as fp:
        test_ids = json.load(fp)

    embeddings = torch.load(embed_path)

    return df, train_ids, test_ids, embeddings


# Read in training dataset
df, train_ids, test_ids, embeddings = read_in_data(
    text_path=os.path.join(DIR_INPUT_TEXT, 'processed_training_sentiment140.csv'),
    embed_path=os.path.join(DIR_INPUT_EMBEDDING, 'embeddings_training_sentiment140_stsb-xlm-r-multilingual_52.pkl'))

# Create and Train Model
train_df = df.loc[train_ids, :]
X_train = embeddings[train_ids, :]
y_train = train_df['label'].values
test_df = df.loc[test_ids, :]
X_test = embeddings[test_ids, :]
y_test = test_df['label'].values


# ===== ===== ===== ===== ===== ===== =====
# Training device
# ===== ===== ===== ===== ===== ===== =====
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# ===== ===== ===== ===== ===== ===== =====
# Data loader
# ===== ===== ===== ===== ===== ===== =====
train_data_dl = TensorDataset(torch.FloatTensor(X_train),
                              torch.LongTensor(y_train))
train_data_sampler = RandomSampler(train_data_dl)
train_data_loader = DataLoader(train_data_dl, sampler=train_data_sampler, batch_size=PARA_BATCH_SIZE)
test_data_dl = TensorDataset(torch.FloatTensor(X_test),
                             torch.LongTensor(y_test))
test_data_sampler = RandomSampler(test_data_dl)
test_data_loader = DataLoader(test_data_dl, sampler=test_data_sampler, batch_size=PARA_BATCH_SIZE)

# ===== ===== ===== ===== ===== ===== =====
# Network
# ===== ===== ===== ===== ===== ===== =====
classes = ('Negative', 'Positive')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense1 = nn.Linear(768, 512)
        self.b1 = nn.BatchNorm1d(512)
        self.m1 = nn.SiLU()
        self.d1 = nn.Dropout(p=0.5)
        self.dense2 = nn.Linear(512, 256)
        self.b2 = nn.BatchNorm1d(256)
        self.m2 = nn.SiLU()
        self.d2 = nn.Dropout(p=0.5)
        self.dense3 = nn.Linear(256, 64)
        self.b3 = nn.BatchNorm1d(64)
        self.m3 = nn.SiLU()
        self.dense4 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.m1(self.dense1(x))
        x = self.m2(self.dense2(x))
        x = self.m3(self.dense3(x))
        x = self.dense4(x)
        return x


# ===== ===== ===== ===== ===== ===== =====
# Evaluation
# ===== ===== ===== ===== ===== ===== =====
def get_test(p_model):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_data_loader:
            inputs, labels = data
            outputs = p_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test accuracy: {correct / total}')
    return correct / total


def get_train(p_model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_data_loader:
            inputs, labels = data
            outputs = p_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Train accuracy: {correct / total}')
    return correct / total

# ===== ===== ===== ===== ===== ===== =====
# Training
# ===== ===== ===== ===== ===== ===== =====
model = Net()
loss_record = []
best_test = 0
best_state = None

for round, lr in enumerate([0.001, 0.0001]):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(0,30):
        running_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):

            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i%2000==1999:
                print(f'[{round*30 + epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0


        temp_accuracy = get_test(model)
        loss_record.append({
                    "Iteration": (round*30 + epoch) * 12800 + 12799 ,
                    "Type": "Test",
                    "Accuracy": temp_accuracy
                })
        if temp_accuracy > best_test:
            best_test = temp_accuracy
            best_state = model.state_dict()
        temp_accuracy = get_train(model)
        loss_record.append({
                    "Iteration": (round*30 + epoch) * 12800 + 12799 ,
                    "Type": "Train",
                    "Accuracy": temp_accuracy
                })

print('Finished Training')

# ===== ===== ===== ===== ===== ===== =====
# Save model
# ===== ===== ===== ===== ===== ===== =====
torch.save(best_state, os.path.join(DIR_OUTPUT, "model_weight.pt"))
