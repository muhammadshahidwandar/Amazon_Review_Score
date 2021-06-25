#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
	Code Writer: Muhammad Shahid

    Discription: This file is used to tokenize, feature embedding extractions for custom text data in JSON format using pre-trained
    english language BERT Model. Finally, the  feature alonge with other desired annotation fields are saved in mat file.

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
import json
from scipy.io import savemat
import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import DistilBertTokenizerFast
from transformers import BertModel





tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def read_Custom_JSON(FilePath):
    File = Path(FilePath)
    ReviewText = []
    ReviewScore = []
    TotalVote = []
    totalSample = 5000
    #fields = {'reviewText': ('rv', review_txt), 'overall': ('s', review), 'vote': ('v', totalvote)}
    i = 0
    for line in open(File, 'r'):  # Electronics_5.json' './Data/Electronics_5.json'
        lin_dictnry = json.loads(line)
        if i < totalSample:
            if 'reviewText' in lin_dictnry and 'vote' in lin_dictnry and 'overall' in lin_dictnry:
                ReviewText.append(lin_dictnry['reviewText'])
                ReviewScore.append(int(lin_dictnry['overall'] - 1))
                totalVote = lin_dictnry['vote'].split(',')
                if(len(totalVote)<2):
                    votes = int(totalVote[0])
                else:
                    votes = int(totalVote[0])*1000+int(totalVote[1])
                TotalVote.append(votes)
                #print(i)
                i = i + 1
        else :
            break
    return ReviewText, ReviewScore, TotalVote


ReviewText, ReviewScore,TotalVote = read_Custom_JSON('./Data/Electronics_5.json')

train_encodings = tokenizer(ReviewText, truncation=True, padding=True)#

class AmazonReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = AmazonReviewDataset(train_encodings, ReviewScore)

'''
Pre_Trained BERT Module
'''
# Defining the Model
class Bert_Embedding(nn.Module):

    def __init__(self):
        super(Bert_Embedding, self).__init__()
        self.Bert = BertModel.from_pretrained('bert-base-uncased')
        self.Bert.eval()
        self.modelName = 'BertFeatureEmmbeding'

    def forward(self, ids, mask):
        sequence_output = self.Bert(ids, attention_mask = mask)
        features = sequence_output.last_hidden_state
        result = features[:, 0, :].view(-1, 768)
        return result

#######
Bert_net  = Bert_Embedding()

#batch = train_dataset[0]
Bert_net.eval()
Bert_net.to(device)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
num_epochs = 1

total_samples = len(train_dataset)
n_iterations = math.ceil(total_samples / 8)
print(total_samples, n_iterations)
BertFeatures= []
for i,batch in enumerate(train_loader): #for i,data in enumerate(train_loader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    outputs = Bert_net(input_ids, mask=attention_mask)
    BertFeatures.append(outputs[0,:].detach().numpy())
    if i%100==0:
        print('the iteration is=',i)
data = {}
data['BertFeaturs'] = BertFeatures
data['ReviewScore'] = ReviewScore
data['TotalVote'] = TotalVote
savemat('./Data/BertAmazonElectonicsNlp.mat', data)
print('Bert Feature Embbeding are Saved')



