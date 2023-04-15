from time import sleep
import cupy as cp
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

model_path = 'clue/roberta_chinese_base'
tokenizer = BertTokenizer.from_pretrained(model_path)
bertmodel = BertModel.from_pretrained(model_path)


def coffate_fn(examples):
    sents = []
    tags = []
    for itemname in examples:
        # itemname = re.sub('[^\u4e00-\u9fa5]+','',itemname)
        sents.append(itemname)
    tokenized_inputs = tokenizer(sents,
                                 truncation=True,
                                 padding=True,
                                 # return_offsets_mapping=True,
                                 is_split_into_words=False,
                                 max_length=100,
                                 return_tensors="pt")
    # targets = torch.tensor(tags)
    return tokenized_inputs


def fill_body(data):
    if pd.isnull(data['POLICY_BODY']):
        return data['POLICY_TITLE']
    else:
        return data['POLICY_BODY']


class BERTDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.datasize = len(dataset)

    def __len__(self):
        return self.datasize

    def __getitem__(self, index):
        return self.dataset[index]


def creat_category(category_title, data_drop_t):
    category_index = open(r'./results/category_index.txt', mode='w', encoding="utf-8")
    print('国家级', '省级', '市级', '区县级', file=category_index)
    for title in category_title:
        category = pd.Categorical(data_drop_t[title])
        print(list(category.categories), file=category_index)
        data_drop_t[title] = category.codes
    category_index.close()


def word2vec(data_drop_t, batch_size, device, data_size, data_type):
    title_dataset = BERTDataset(list(data_drop_t['POLICY_TITLE']))
    title_dataloader = DataLoader(title_dataset,
                                  batch_size=batch_size,
                                  collate_fn=coffate_fn,
                                  shuffle=False)
    body_dataset = BERTDataset(list(data_drop_t['POLICY_BODY']))
    body_dataloader = DataLoader(body_dataset,
                                 batch_size=batch_size,
                                 collate_fn=coffate_fn,
                                 shuffle=False)

    DBLPdata = open(r'./results/DBLPdata.txt', mode='w')

    data_attribute = list(data_type.keys())
    for name in data_attribute:
        print(name, data_type[name], data_size[name], end=' ', file=DBLPdata)
    print(file=DBLPdata)

    ind = 0
    bertmodel.to(device)
    with torch.no_grad():
        with tqdm(total=len(title_dataloader)) as t:
            for title, body in zip(title_dataloader, body_dataloader):
                title_output = bertmodel(**title.to(device))
                title_to_vec = title_output['pooler_output'].to('cpu')
                body_output = bertmodel(**body.to(device))
                body_to_vec = body_output['pooler_output'].to('cpu')
                for i in range(batch_size):
                    if ind + i == data_drop_t.shape[0]:
                        break
                    row = []
                    for name in data_attribute:
                        if name != 'POLICY_TITLE' and name != 'POLICY_BODY':
                            row.append(data_drop_t.iloc[ind + i][name])
                        elif name == 'POLICY_TITLE':
                            row.extend(title_to_vec[i].numpy().tolist())
                        else:
                            row.extend(body_to_vec[i].numpy().tolist())
                    print(row, file=DBLPdata)
                ind += batch_size
                # sleep(0.1)
                t.update(1)
    DBLPdata.close()


def create_vertex(data_drop_t, data_type, to_file, sampled):
    vertex = None
    save_root = None
    if not sampled:
        save_root = r'./results/vertex.txt'
    else:
        save_root = r'./results/random_sample/vertex.txt'
    if to_file:
        vertex = open(save_root, mode='w')
    policy_index = list(range(data_drop_t.shape[0]))
    node_type = {}  # 属性名: 属性类型编号
    attribute2num = {}  # 属性值：节点编号
    vertex_table = []  # 节点编号：属性类型编号

    ind = 1
    for attribute, attribute_type in data_type.items():
        if attribute_type == 'text':
            node_type[attribute] = ind
    for i in policy_index:
        vertex_table.append((i, 0))
        if to_file:
            print(i, 0, file=vertex)
    ind = policy_index[-1]
    for attribute, a_type in node_type.items():
        nodes = data_drop_t[attribute].unique()
        with tqdm(total=len(nodes)) as t:
            t.set_description(f"{attribute}:")
            for node in nodes:
                if node is not np.nan:
                    ind += 1
                    attribute2num[node] = ind
                    vertex_table.append((ind, a_type))
                    if to_file:
                        print(ind, a_type, file=vertex)
                t.update(1)
    if to_file:
        vertex.close()
    return node_type, attribute2num, vertex_table


def create_graph(data_drop_t, node_type, attribute2num, vertex_table, to_file, sampled):
    adjacency_list = {i: [] for i in range(len(vertex_table))}
    attributes = list(node_type.keys())
    with tqdm(total=len(data_drop_t)) as t:
        t.set_description("Creating_Graph:")
        for ind, policy in data_drop_t.iterrows():
            ind_i = ind
            for attribute in node_type.keys():
                val = policy[attribute]
                if val is not np.nan:
                    ind_j = attribute2num[val]
                    adjacency_list[ind_i].append(ind_j)
                    adjacency_list[ind_j].append(ind_i)
            for a1 in range(len(attributes)):
                for a2 in range(a1 + 1, len(attributes)):
                    if policy[attributes[a1]] is not np.nan and policy[attributes[a2]] is not np.nan:
                        i1 = attribute2num[policy[attributes[a1]]]
                        i2 = attribute2num[policy[attributes[a2]]]
                        adjacency_list[i1].append(i2)
                        adjacency_list[i2].append(i1)
            t.update(1)
    graph = None
    save_root = None
    if not sampled:
        save_root = r'./results/graph.txt'
    else:
        save_root = r'./results/random_sample/graph.txt'
    if to_file:
        graph = open(save_root, mode='w')
        for ind, lists in adjacency_list.items():
            lists = set(lists)
            print(ind, end=' ', file=graph)
            for i in lists:
                print(i, end=' ', file=graph)
            print(file=graph)
        graph.close()
    return adjacency_list


def cal_l2_distances_no_loop(X):
    """
    计算L2距离，通过矩阵运算
    :return:
    """

    first = cp.sum(cp.square(X), axis=1)
    second = cp.sum(cp.square(X), axis=1).T
    # 注意这里的np.dot(test_X, train_X.T)中的test_X, train_X位置是和公式中的顺序保持一致的
    three = -2 * cp.dot(X, X.T)

    dists = cp.sqrt(first + second + three)
    return dists


def cal_l2_distances_one_loop(X):
    """
    计算L2距离，一层循环
    :return:
    """
    length = X.shape[0]
    dists = cp.zeros((length, length), dtype=cp.float16)
    with tqdm(total=length) as t:
        for i in range(length):
            dists[i] = cp.sqrt(cp.sum(cp.square(X - X[i, :]), axis=1)).T
            t.update(1)
    return dists


def normalization(data):
    _range = np.max(data, axis=0) - np.min(data, axis=0)
    return (data - np.min(data, axis=0)) / _range


# def compute_distances_two_loop(X):
#     length = X.shape[0]
#     dists = np.zeros((length, length))    # shape(num_test, num-train)
#     for i in range(length):
#         for j in range(length):
#             # corresponding element in Numpy Array can compute directly,such as plus, multiply
#             dists[i][j] = np.sqrt(np.sum(np.square(X[i] - X[j])))
#     return dists

def row2vec(row, title, category_list):
    texts = row[title]
    if texts is np.nan:
        return -1
    return category_list.index(texts)


def change2vec(data):
    category_title = ['POLICY_GRADE', 'POLICY_SOURCE', 'POLICY_TYPE', 'PROVINCE', 'PUB_AGENCY_FULLNAME']
    category_list = []
    with open(r'./results/category_index.txt', encoding="utf-8") as f:
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
        print(title)
        data[title] = data.apply(row2vec, args=(title, category_list[ind]), axis=1)