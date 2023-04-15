import pandas as pd
import numpy as np
import torch
from utils import *
from read_funcs import *
import cupy as cp


batch_size = 64

data_ori = pd.read_csv(r"./data/policyinfo_new.tsv", encoding='gb18030', sep='\t')
# data_ori = read_tsv(r"./data/policyinfo_new.tsv")
# print(data_ori.isnull().sum())
del_title = ['PUB_AGENCY_ID', 'PUB_NUMBER', 'CITY', 'PUB_AGENCY']
data_drop_t = data_ori.drop(del_title, axis=1, inplace=False)
data_drop_t['POLICY_BODY'] = data_drop_t.apply(fill_body, axis=1)
# print(data_drop_t.isnull().sum())

# index = list(range(data_drop_t.shape[0]))
# data_drop_t['index'] = index
# data_drop_t.to_json(r'./results/data_json.json', orient='records')

category_title = ['POLICY_SOURCE', 'POLICY_TYPE', 'PROVINCE', 'PUB_AGENCY_FULLNAME']
data_drop_t['POLICY_GRADE'] = pd.Categorical(data_drop_t['POLICY_GRADE'], categories=['国家级', '省级', '市级', '区县级']).codes
data_drop_t['PUB_TIME'] = pd.to_datetime(data_drop_t['PUB_TIME']).apply(lambda x: x.value/1e14)
data_drop_t['UPDATE_DATE'] = pd.to_datetime(data_drop_t['UPDATE_DATE']).apply(lambda x: x.value/1e14)
# np.set_printoptions(threshold=np.inf)
# pd.set_option('display.max_rows', None)
data_size = {
    'POLICY_TITLE': 768,
    'POLICY_GRADE': 1,
    'PUB_AGENCY_FULLNAME': 1,
    'PUB_TIME': 1,
    'POLICY_TYPE': 1,
    'POLICY_BODY': 768,
    'PROVINCE': 1,
    'POLICY_SOURCE': 1,
    'UPDATE_DATE': 1
}
dim_pos = np.array([768, 769, 770, 771, 1541, 1540, 1542])
data_type = {
    'POLICY_TITLE': 'continuousvalue',
    'POLICY_GRADE': 'text',
    'PUB_AGENCY_FULLNAME': 'text',
    'PUB_TIME': 'text',
    'POLICY_TYPE': 'text',
    'POLICY_BODY': 'continuousvalue',
    'PROVINCE': 'text',
    'POLICY_SOURCE': 'text',
    'UPDATE_DATE': 'text'
}

creat_category(category_title, data_drop_t)
# #
device = 'cuda' if torch.cuda.is_available() else 'cpu'
word2vec(data_drop_t, batch_size, device, data_size, data_type)

node_type, attribute2num, vertex_table = create_vertex(data_drop_t, data_type, to_file=True, sampled=False)
# node_type 属性名: 属性类型编号
# attribute2num 属性值：节点编号
# vertex_table 节点编号：属性类型编号

adjacency_list = create_graph(data_drop_t, node_type, attribute2num, vertex_table, to_file=True, sampled=False)

# cal distance
# fileroot = r'./results/DBLPdata.txt'
# length = 161150
# dblp_data = read_DBLP(fileroot, length)
# dblp_data[dblp_data == np.inf] = 0
# dblp_data[dblp_data == -np.inf] = 0
# dblp_data[dblp_data == -1] = 0
# dblp_data = dblp_data.astype(np.float16)
# dblp_data[:, dim_pos] = normalization(dblp_data[:, dim_pos])
# distance_matrix = cal_l2_distances_no_loops(dblp_data)
# np.savetxt("./results/distance_matrix.csv", distance_matrix, delimiter=",")
