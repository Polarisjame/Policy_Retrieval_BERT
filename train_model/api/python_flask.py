import numpy as np
from flask import Flask, request
import os
import sys
from argparse import ArgumentParser
import torch
import json
import time

sys.path.append("../")
from model import BERT_MLP, BERT

os.chdir("../")


def cal_l2_distances_no_loop(a, bs):
    """
    计算L2距离，通过矩阵运算
    :return:
    """

    first = np.sum(np.square(a), axis=1)
    second = np.sum(np.square(bs), axis=1).T
    # 注意这里的np.dot(test_X, train_X.T)中的test_X, train_X位置是和公式中的顺序保持一致的
    three = -2 * np.dot(a, bs.T)

    dists = np.sqrt(first + second + three)
    return dists


app = Flask(__name__)
parser = ArgumentParser()
opt = parser.parse_args()
with open('commandline_args.txt', 'r') as f:
    opt.__dict__ = json.load(f)
bert_model_path = './checkpoint/BERT_DoubleMLP_best.pt'

rates = np.array([0.7, 0.3])
grades = {
    '国家级': 1,
    '省级': 2,
    '市级': 3,
    '区县级': 4,
    '': 4
}
data_type = {
    'POLICY_GRADE': 'text',
    'PUB_AGENCY_FULLNAME': 'text',
    'POLICY_TYPE': 'text',
    'PROVINCE': 'text',
    'POLICY_SOURCE': 'text',
}

model = BERT_MLP(opt)
chkpt = torch.load(bert_model_path, map_location=torch.device('cpu'))
model.load_state_dict(chkpt['checkpoint'])
bert_tokenize = BERT(opt.tokenize_model_name)


@app.route('/json', methods=['POST'])
def data_process():
    data = request.get_data(as_text=True)
    json_obj = json.loads(data)
    policys = []
    user = ''
    policy_grade = []
    n = len(json_obj['Policy'])
    for obj in json_obj['Policy']:
        policy = obj['POLICY_TITLE']
        for attribute in data_type.keys():
            policy += '[SEP]' + obj[attribute]
        policy_grade.append(grades[obj['POLICY_GRADE']])
        policys.append(policy)
    policy_grade = np.array(policy_grade)
    # with torch.no_grad():
    #     input_tokenize = bert_tokenize.tokenizer(policys)
    #     policy_out_tensor = model(input_tokenize)
    #     policy_out_tensor = np.array(policy_out_tensor)
    ind = 0
    for key, item in json_obj['User'].items():
        if not ind:
            user += item
            ind += 1
        else:
            user += '[SEP]' + item
    policys.append(user)
    a = time.time()
    with torch.no_grad():
        input_tokenize = bert_tokenize.tokenizer(policys)
        out_tensor = model(input_tokenize)
        out_tensor = np.array(out_tensor)
        policy_out_tensor = out_tensor[:-1]
        user_out_tensor = out_tensor[-1].reshape([1, -1])
    b = time.time()
    print(b-a)
    vec_dis = cal_l2_distances_no_loop(user_out_tensor, policy_out_tensor)
    grade_dis = (policy_grade - np.min(policy_grade)) / (np.max(policy_grade) - np.min(policy_grade))
    final_dis = rates[0] * vec_dis + rates[1] * grade_dis
    final_dis.resize(n)
    final_dis = final_dis.tolist()
    return json.dumps({'res': final_dis})


if __name__ == '__main__':
    app.run()
