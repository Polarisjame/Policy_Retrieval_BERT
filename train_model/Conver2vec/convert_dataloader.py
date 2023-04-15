import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from model import BERT

class Conver_Dataset(Dataset):
    def __init__(self, data_vec: pd.DataFrame, opt):

        data_type = {
            'POLICY_TITLE': 'continuousvalue',
            'POLICY_BODY': 'continuousvalue',
            'POLICY_GRADE': 'text',
            'PUB_AGENCY_FULLNAME': 'text',
            'PUB_TIME': 'text',
            'POLICY_TYPE': 'text',
            'PROVINCE': 'text',
            'POLICY_SOURCE': 'text',
            'UPDATE_DATE': 'text'
        }

        self.policies = []  # list:[trip_attribute0:list,...]
        for ind, policy in data_vec.iterrows():
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
                if opt.index_type == 'Body':
                    self.policies.append(body)
                elif opt.index_type == 'Title':
                    self.policies.append('[SEP]'.join([title,attribute]))
                else:
                    assert 1==2, 'Please provide correct index_type'

    def __getitem__(self, index):
        return self.policies[index]

    def __len__(self):
        return len(self.policies)


class collater():
    def __init__(self, opt):
        self.tokenize_model_name = opt.tokenize_model_name
        self.bert = BERT(self.tokenize_model_name)

    def __call__(self, body_inputs):
        '''
        :param examples: list:b_z*[title,body,...]
        :return: list:[b_z*title_inputs, b_z*body_inputs, b_z*query]
        '''
        body_inputs = self.bert.tokenizer(body_inputs)
        return body_inputs
