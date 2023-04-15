import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import cupy as cp


def read_DBLP(fileroot, length=None):
    max_length = length
    result = cp.zeros((max_length, 1543), dtype=cp.float16)
    with tqdm(total=length) as t:
        t.set_description('Reading DBLP')
        with open(fileroot) as f:
            ind = 0
            for line in f:
                if ind == 0:
                    ind += 1
                    continue
                line = line[1:-2]
                feat = line.split(',')
                feat = [float(a) for a in feat]
                feat[770] = int(feat[770] / 1e14)
                feat[1542] = int(feat[1542] / 1e14)
                result[ind - 1, :] = cp.array(feat, dtype=cp.float16)
                t.update(1)
                ind += 1
                if ind == max_length:
                    break
    return result


def read_triplets(file_root, already_query, distance_matrix_norm, trip_inds, margin, neg_limit, pos_limit):
    is_in = False
    with open(file_root, mode='r') as f:
        for row, line in enumerate(f):
            if row % 3 == 0:
                query_index = int(line[:-1])
                if query_index in already_query:
                    is_in = True
                    continue
                else:
                    already_query.append(query_index)
                    is_in = False
            elif row % 3 == 1:
                if is_in:
                    continue
                positive = line[1:-2]
                positive = [int(a) for a in positive.split(',')]
                if len(positive) > pos_limit:
                    sample_ind = list(range(len(positive)))
                    sample = random.sample(sample_ind, pos_limit)
                    positive = [positive[a] for a in sample]
            else:
                if is_in:
                    continue
                negative = line[1:-2]
                negative = [int(a) for a in negative.split(',')]

                for pos_ind in positive:
                    d_ap = distance_matrix_norm[query_index][pos_ind]
                    ind = 0
                    for neg_ind in negative:
                        d_an = distance_matrix_norm[query_index][neg_ind]
                        L = max(d_ap - d_an + margin, 0)
                        if L > 0:
                            trip_inds.append((query_index, pos_ind, neg_ind))
                            ind += 1
                            if ind == neg_limit:
                                break


def read_tsv(file_root: str):
    data_dict = None
    with open(file_root, encoding='gb18030') as f:
        for ind, i in enumerate(f):
            i = i[:-1]
            if ind == 0:
                attributes = i.split('\t')
                data_dict = {a: [] for a in attributes}
                continue
            line = i.split('\t')
            for num, element in enumerate(line):
                data_dict[attributes[num]].append(element)
    data_df = pd.DataFrame.from_dict(data_dict)
    data_df.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)
    return data_df
