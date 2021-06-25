#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
	Code Writer: Muhammad Shahid

    Discription: This file is used to train a fully connected neural network for Review score prediction from BERT embeddings

    Usage Example: (change the file path)
		$ python3

    LICENSE:
	This project is licensed under the terms of the MIT license.
	This project incorporates material from the projects listed below (collectively, "Third Party Code").
	This Third Party Code is licensed to you under their original license terms.
	We reserves all other rights not expressly granted, whether by implication, estoppel or otherwise.
	The software can be freely used for any non-commercial applications.
"""
from pathlib import Path
from sklearn.model_selection import train_test_split
import scipy.io as sio
import math

import torch
from torch import nn
from torch.utils.data import DataLoader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_epochs = 400
batch_size = 256

def read_AmzonReview_Mat(FilePath):
    File = Path(FilePath)
    NLPMat = sio.loadmat(File)
    ReviewFeature  = NLPMat['BertFeaturs']
    ReviewScore = NLPMat['ReviewScore']
    TotalVote   = NLPMat['TotalVote']
    return ReviewFeature, ReviewScore, TotalVote  #ReviewText, ReviewScore, TotalVote

MatFile = './Data/BertAmazonElectonicsNlp.mat'
ReviewFeature, ReviewScore, TotalVote = read_AmzonReview_Mat(MatFile)

train_ReviewFeatr, val_ReviewFeatr, train_ReviewScore, val_ReviewScore,train_TotalVote, val_TotalVote= \
    train_test_split(ReviewFeature, ReviewScore.transpose(), TotalVote.transpose(), test_size=.1)

class AmazonReviewDataset(torch.utils.data.Dataset):
    def __init__(self, BertFeature, ReviewScore,TotalVote):
        self.BertFeature = BertFeature
        self.ReviewScore = ReviewScore
        self.TotalVote   = TotalVote

    def __getitem__(self, idx):
        item = []
        BertFeature = torch.tensor(self.BertFeature[idx,:])#{key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        ReviewScore = torch.tensor(self.ReviewScore[idx,0],dtype= torch.int64)
        TotalVote   = torch.tensor(self.TotalVote[idx,0],dtype= torch.int64)
        return BertFeature,ReviewScore,TotalVote

    def __len__(self):
        return len(self.ReviewScore)
train_dataset = AmazonReviewDataset(train_ReviewFeatr, train_ReviewScore,train_TotalVote)
val_dataset = AmazonReviewDataset(val_ReviewFeatr, val_ReviewScore,val_TotalVote)

BertFeature,ReviewScore,TotalVote   = train_dataset[0] # Accessing First Batch
##########

'''
Fully connected Neural Network for Review Score prediction
'''

# Defining the Model
class Bert_fc(nn.Module):

    def __init__(self, num_classes, p=0.2):
        super(Bert_fc, self).__init__()
        self.linear1 = nn.Linear(768, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 512)
        #self.linear4 = nn.Linear(512, 256)
        self.drop_layer = nn.Dropout(p=p)
        self.linear4 = nn.Linear(512, num_classes)
        self.modelName = 'BertRegressionClassifier'

    def forward(self, features):
        # sequence_output has the following shape: (batch_size, sequence_length, 768) sequence_output[:, 0, :]
        linear1_output = self.linear1(features)
        linear2_output = self.linear2(linear1_output)
        linear3_output = self.linear3(linear2_output)
        linear4_output = self.linear4(linear3_output)
        return linear4_output

#######
Bert_net  = Bert_fc(num_classes=5)
criterion = nn.CrossEntropyLoss()

Bert_net.to(device)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
optim = torch.optim.Adam(Bert_net.parameters(),lr=5e-5)

total_train_samples = len(train_dataset)
total_val_samples = len(val_dataset)
n_iterations = math.ceil(total_train_samples / batch_size)
print(total_train_samples, n_iterations)
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i,Bert in enumerate(train_loader):
        optim.zero_grad()
        features = Bert[0]
        labels  = Bert[1]
        score  = Bert[2]
        features = features.to(device)
        labels = labels.to(device)
        outputs = Bert_net(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()
        if (i + 1) % 15 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
        if (i + 1) % 15 == 0:
            Bert_net.eval()
            # Check performance of trained model for training data
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for batch in train_loader:
                    features = Bert[0]
                    labels = Bert[1]
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = Bert_net(features)  # , labels=labels)
                    _, predicted = torch.max(outputs, 1)
                    # predicted = torch.argmax(outputs[-1, :]).item()
                    n_samples += labels.size(0)
                    n_correct += (predicted == labels).sum().item()
                acc = 100.0 * n_correct / n_samples
                print(f'Accuracy of the network on the {total_train_samples}  training samples: {acc} %')
    if (epoch + 1) % 5 == 0:
        Bert_net.eval()
        # Check performance of trained model for test data
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for batch in valid_loader:
                features = Bert[0]
                labels = Bert[1]
                features = features.to(device)
                labels = labels.to(device)
                outputs = Bert_net(features)  # , labels=labels)
                # outputs = model(input_ids)
                # max returns (value ,index)
                # _, predicted = torch.max(outputs.logits, 1)#_, predicted = torch.max(outputs.data, 1)
                _, predicted = torch.max(outputs, 1)
                # predicted = torch.argmax(outputs[-1, :]).item()
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()
            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network on the {total_val_samples}  test samples: {acc} %')
    if epoch % 40 == 0:
        FILE = "./Data/model_"+str(epoch) +".pth"
        torch.save(Bert_net, FILE)

