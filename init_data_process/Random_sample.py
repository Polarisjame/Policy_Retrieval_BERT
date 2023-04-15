import copy
import random
from utils import *
from read_funcs import *
import cupy as cp


random.seed(42)
dim_pos = np.array([768, 769, 770, 771, 1541, 1540, 1542])
fileroot = r'./results/DBLPdata.txt'
length = 161150
rate = 0.01

# dblp_data = read_DBLP(fileroot, length)
sample_ind = random.sample(range(length), int(rate * length))
# dblp_data_sample = dblp_data[sample_ind]
# dblp_data_sample[dblp_data_sample == np.inf] = 0
# dblp_data_sample[dblp_data_sample == -np.inf] = 0
# dblp_data_sample[dblp_data_sample == -1] = 0
# dblp_data_sample = dblp_data_sample.astype(np.float16)
# dblp_data_sample[:, dim_pos] = normalization(dblp_data_sample[:, dim_pos])
# dblp_data_sample_body = dblp_data_sample[:,772:772+768]
# distance_matrix = cal_l2_distances_one_loop(dblp_data_sample_body)
# np.savetxt("./results/random_sample/distance_matrix_body.csv", distance_matrix, delimiter=",", fmt="%.6f")

data_ori = pd.read_csv(r"./data/policyinfo_new.tsv", encoding='gb18030', sep='\t')
del_title = ['PUB_AGENCY_ID', 'PUB_NUMBER', 'CITY', 'PUB_AGENCY']
data_drop_t = data_ori.drop(del_title, axis=1, inplace=False)
data_drop_t['POLICY_BODY'] = data_drop_t.apply(fill_body, axis=1)
data_sample = copy.deepcopy(data_drop_t.iloc[sample_ind])
data_sample.index = list(range(len(data_sample)))
data_sample.to_csv(r'./fine_tune/data/data_sample.csv')

data_sample['PUB_TIME'] = pd.to_datetime(data_sample['PUB_TIME']).apply(lambda x: x.value/1e14)
data_sample['UPDATE_DATE'] = pd.to_datetime(data_sample['UPDATE_DATE']).apply(lambda x: x.value/1e14)

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

change2vec(data_sample)
# #
device = 'cuda' if torch.cuda.is_available() else 'cpu'
word2vec(data_sample, 64, device, data_size, data_type)

dblp_data = read_DBLP(fileroot, length=len(data_sample))
# sample_ind = random.sample(range(length), int(rate * length))
dblp_data_sample = dblp_data
dblp_data_sample[dblp_data_sample == np.inf] = 0
dblp_data_sample[dblp_data_sample == -np.inf] = 0
dblp_data_sample[dblp_data_sample == -1] = 0
dblp_data_sample = dblp_data_sample.astype(np.float16)
dblp_data_sample[:, dim_pos] = normalization(dblp_data_sample[:, dim_pos])
dblp_data_sample_body = dblp_data_sample[:,772:772+768]
distance_matrix = cal_l2_distances_one_loop(dblp_data_sample_body)
np.savetxt("./results/random_sample/distance_matrix_body.csv", distance_matrix, delimiter=",", fmt="%.6f")

node_type, attribute2num, vertex_table = create_vertex(data_sample, data_type, to_file=True, sampled=True)
# node_type 属性名: 属性类型编号
# attribute2num 属性值：节点编号
# vertex_table 节点编号：属性类型编号

adjacency_list = create_graph(data_sample, node_type, attribute2num, vertex_table, to_file=True, sampled=True)
