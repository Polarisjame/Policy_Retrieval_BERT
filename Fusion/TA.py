import pandas as pd
import numpy as np
from utils import *
import time


def Multi_Channel_Fusion(df_channels: list):
    '''
    :param list, channels: lists of channels:[channel1:Dict{id:Dataframe},channel2:Dataframe,...]
    :return: dict, Fused Result
    '''
    index = df_channels[0].keys()
    res_df = {}
    for ind in index:
        pd1 = df_channels[0][ind]
        pd2 = df_channels[1][ind]
        pd3 = df_channels[2][ind]
        df_combine = pd.merge(pd1, pd2, on='policy', how='outer')
        df_combine_2 = pd.merge(df_combine, pd3, on='policy', how='outer')
        df_combine_2.fillna(1, inplace=True)
        df_combine_2['combine_dis'] = df_combine_2.apply(merge_distance, axis=1)
        df_sorted = df_combine_2.sort_values(by='combine_dis', ascending=True)
        res_df[ind] = df_sorted.iloc[:100]
    return res_df


def Fusion(title_id, title_dis, body_id, body_dis, es_id, es_dis):
    # pd.set_option('display.max_rows', None)
    # file_root_body = r'./data/searchBody.txt'
    # body_dfs = read_file(file_root_body)
    # file_root_title = r'./data/searchTitle.txt'
    # title_dfs = read_file(file_root_title)
    # file_root_es = r'./data/es_result.txt'
    # es_dfs = read_file(file_root_es)
    # df_channels = [es_dfs, title_dfs, body_dfs]
    id = 0
    df_channels = load_data_java(id, title_id, title_dis, body_id, body_dis, es_id, es_dis)
    res_df = Multi_Channel_Fusion(df_channels)
    # evaluate(res_df)
    return list(res_df[id]['policy'])
