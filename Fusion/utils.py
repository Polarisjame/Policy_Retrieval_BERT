import pandas as pd
import os


def read_file(file_root: str):
    """
    :param file_root: str,Root of File
    :return: dfs:Dict{query_id:correspond_df}
    """
    with open(file_root, "r", encoding='utf-8') as f:
        dfs = {}
        for ind, row in enumerate(f):
            row = row[:-1].strip()
            if ind % 3 == 0:
                query = int(row)
            elif ind % 3 == 1:
                policy = [int(a) for a in row.split(' ')]
            else:
                distance = [float(a) for a in row.split(' ')]
                df = pd.DataFrame.from_dict({'policy': policy, 'distance': distance})
                dfs[query] = df
        return dfs


def merge_distance(x):
    return min(x.distance_x, x.distance_y, x.distance)


def evaluate(res_df: dict):
    out_file = r'./result'
    data_root = r'./data/policyinfo_new.tsv'
    if not os.path.exists(out_file):
        os.mkdir(out_file)
    with open(os.path.join(out_file, 'evaluate.txt'), 'w') as f:
        data_ori = pd.read_csv(data_root, sep='\t', encoding='gb18030')
        # print(data_ori.isnull().sum())
        del_title = ['PUB_AGENCY_ID', 'PUB_NUMBER', 'CITY', 'PUB_AGENCY']
        data_drop_t = data_ori.drop(del_title, axis=1, inplace=False)
        count = 0
        for ind, df in res_df.items():
            count += 1
            if 15 < count < 30:
                print('Indexing Title:', data_drop_t.iloc[ind]['POLICY_TITLE'], file=f)
                print(ind, 'Indexing Title:', data_drop_t.iloc[ind]['POLICY_TITLE'])
                print('Retrieval Res:', file=f)
                for policy_index in df['policy']:
                    print(data_drop_t.iloc[policy_index]['POLICY_TITLE'], file=f)
                print(' ----------------------------------------------------- ', file=f)
            elif count == 141540:
                break


def load_data_java(id, title_id, title_dis, body_id, body_dis, es_id, es_dis):
    es_dfs = {id: pd.DataFrame.from_dict({'policy': es_id, 'distance': es_dis})}
    title_dfs = {id: pd.DataFrame.from_dict({'policy': title_id, 'distance': title_dis})}
    body_dfs = {id: pd.DataFrame.from_dict({'policy': body_id, 'distance': body_dis})}
    result = [es_dfs, title_dfs, body_dfs]
    return result
