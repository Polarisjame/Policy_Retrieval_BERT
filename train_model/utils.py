from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.nn.functional import pairwise_distance,relu
import pandas as pd


def fill_body(data):
    if pd.isnull(data['POLICY_BODY']):
        return data['POLICY_TITLE']
    else:
        return data['POLICY_BODY']


def row2vec(row, title, category_list):
    texts = row[title]
    if texts is np.nan:
        return -1
    return category_list.index(texts)


def change2vec(data):
    category_title = ['POLICY_GRADE', 'POLICY_SOURCE', 'POLICY_TYPE', 'PROVINCE', 'PUB_AGENCY_FULLNAME']
    category_list = []
    with open(r'./data/category_index.txt', encoding="utf-8") as f:
        for ind, line in enumerate(f):
            line = line[:-1]
            if ind == 0:
                category_list.append(line.split())
            else:
                line = line[1: -1]
                line = [a.strip()[1:] for a in line.split('\', ')]
                line[-1] = line[-1][:-1]
                category_list.append(line)
    for ind, title in enumerate(category_title):
        data[title] = data.apply(row2vec, args=(title, category_list[ind]), axis=1)


def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def get_bert_cuda(tokenize):
    input_ids = get_cuda(tokenize.data['input_ids'])
    token_type_ids = get_cuda(tokenize.data['token_type_ids'])
    attention_mask = get_cuda(tokenize.data['attention_mask'])
    return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}


def logging(s):
    print(datetime.now(), s)


def distance(a,b):
    return torch.sqrt(torch.sum(torch.pow(a-b,2)))


class triplet_loss(nn.Module):
    def __init__(self, margin=1):
        super(triplet_loss,self).__init__()
        self.margin = margin
    
    def forward(self,ancor,positive,negative):
        pos_dis = pairwise_distance(ancor,positive)
        neg_dis = pairwise_distance(ancor,negative)
        loss = relu(pos_dis - neg_dis + self.margin)
        return loss.sum()


def get_tensor_distance(a,b):
    return pairwise_distance(a, b)