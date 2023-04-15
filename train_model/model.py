import torch
from torch import nn
from transformers import BertTokenizer, BertModel


class BERT():

    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer_model = BertTokenizer.from_pretrained(model_name)

    def tokenizer(self, text):
        title_inputs = self.tokenizer_model(text,
                                            padding='max_length',
                                            truncation=True,
                                            max_length=512,
                                            return_tensors="pt")
        return title_inputs


class BERT_MLP(nn.Module):

    def __init__(self, opt):
        super(BERT_MLP, self).__init__()
        self.bert = BertModel.from_pretrained(opt.encode_model_name)
        if not opt.use_bert_finetune:
            for parameters in self.bert.parameters():
                parameters.requires_grad = False
        # if opt.dropout > 0:
        #     self.dropout = nn.Dropout(opt.dropout)
        # self.MLP = nn.Sequential(
        #     nn.Linear(768, opt.model_output_dim),
        #     nn.BatchNorm1d(opt.model_output_dim),
        #     nn.ReLU(),
        #     self.dropout,
        # )
        # self.dropout_rate = opt.dropout
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, triplet):
        body_tokenize = triplet
        body_output = self.bert(**body_tokenize, return_dict=True)
        output = body_output.pooler_output
        # origin = output.to(torch.float16)
        # output = self.MLP(output)
        # if self.dropout_rate > 0:
        #     output = self.dropout(output)
        return output

