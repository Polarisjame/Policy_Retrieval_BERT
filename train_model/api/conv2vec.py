import os
import sys
from argparse import ArgumentParser
import torch
import json

sys.path.append("../")
from model import BERT_MLP, BERT
os.chdir("../")

parser = ArgumentParser()
opt = parser.parse_args()
with open('commandline_args.txt', 'r') as f:
    opt.__dict__ = json.load(f)
bert_model_path = './checkpoint/BERT_DoubleMLP_best.pt'

model = BERT_MLP(opt)
chkpt = torch.load(bert_model_path, map_location=torch.device('cpu'))
model.load_state_dict(chkpt['checkpoint'])
bert_tokenize = BERT(opt.tokenize_model_name)


# tokenize = BertTokenizer.from_pretrained(opt.tokenize_model_name)

def convert2vector(text):
    with torch.no_grad():
        input_tokenize = bert_tokenize.tokenizer(text)
        out_tensor = model(input_tokenize)
        return out_tensor.numpy()
