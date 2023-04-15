from torch.utils.data import Dataset
import torch
from model import BERT
import pandas as pd


class BERT_Dataset(Dataset):

    def __init__(self, data2vec, triplets):

        data_type = {
            'POLICY_TITLE': 'continuousvalue',
            'POLICY_BODY': 'continuousvalue',
            'POLICY_GRADE': 'text',
            'PUB_AGENCY_FULLNAME': 'text',
            # 'PUB_TIME': 'text',
            'POLICY_TYPE': 'text',
            'PROVINCE': 'text',
            'POLICY_SOURCE': 'text',
            # 'UPDATE_DATE': 'text'
        }

        self.trip_dicts = []  # list:[trip_attribute0:list,...]
        for ind, trip in triplets.iterrows():
            single_trip_attribute = []  # list:[[title,body,grade,...],{},{}]
            for ind,index in enumerate(trip):
                policy = data2vec.loc[index]
                if ind == 0:
                    single_trip_attribute.append(policy['POLICY_TITLE'])
                else:
                    title = policy['POLICY_TITLE']
                    body = policy['POLICY_BODY']

                    attribute = []
                    for tite,typea in data_type.items():
                        if typea == 'test':
                            attr = policy[tite]
                            if pd.isna(attr):
                                attr = '[UNK]'
                            attribute.append(attr)
                    attribute = ','.join(attribute)
                    single_trip_attribute.append(body)
            self.trip_dicts.append(single_trip_attribute)

    def __getitem__(self, index):
        return self.trip_dicts[index]

    def __len__(self):
        return len(self.trip_dicts)


class collater():
    def __init__(self, opt):
        self.tokenize_model_name = opt.tokenize_model_name
        self.bert = BERT(self.tokenize_model_name)

    def __call__(self, examples):
        query_tokenize = []
        positive_tokenize = []
        negative_tokenize = []
        for single_trip_attribute in examples:
            for ind,policy in enumerate(single_trip_attribute):
                if len(policy) > 512:
                    policy = policy[:512]
                if ind == 0:
                    query_tokenize.append(policy)
                if ind == 1:
                    positive_tokenize.append(policy)
                if ind == 2:
                    negative_tokenize.append(policy)
        query_out = None
        positive_out = None
        negative_out = None
        outs_tokenize = [query_tokenize,positive_tokenize,negative_tokenize]
        for ind, out_tokenize in enumerate(outs_tokenize):
            body_inputs = self.bert.tokenizer(out_tokenize)
            if ind == 0:
                query_out = body_inputs
            if ind == 1:
                positive_out = body_inputs
            if ind == 2:
                negative_out = body_inputs
        # targets = torch.tensor(tags)
        return query_out, positive_out, negative_out

