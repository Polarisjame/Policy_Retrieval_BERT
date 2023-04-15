import os
import sys
import copy
import dill
import pandas as pd
import torch
import json
sys.path.append("../")
from utils import *
from model import BERT_MLP
from config import *
from torch.utils.data import Dataset, DataLoader
from Conver2vec.convert_dataloader import *


def create_loader(data_root: str, opt) -> DataLoader:
    loader_save_file = opt.full_dataloader_save_file
    if not os.path.exists(os.path.join(loader_save_file, 'Dataloader_full_title.pkl')) or opt.reload:
        logging('Read From Origin Data')
        data_ori = pd.read_csv(data_root, sep='\t', encoding='gb18030')
        # print(data_ori.isnull().sum())
        del_title = ['PUB_AGENCY_ID', 'PUB_NUMBER', 'CITY', 'PUB_AGENCY']
        data_drop_t = data_ori.drop(del_title, axis=1, inplace=False)
        data_drop_t['POLICY_BODY'] = data_drop_t.apply(fill_body, axis=1)
        data_vec = copy.deepcopy(data_drop_t)
        change2vec(data_vec)
        data_vec['PUB_TIME'] = pd.to_datetime(data_vec['PUB_TIME']).apply(lambda x: x.value / 1e14)
        data_vec['UPDATE_DATE'] = pd.to_datetime(data_vec['UPDATE_DATE']).apply(lambda x: x.value / 1e14)

        BERT_set = Conver_Dataset(data_vec, opt)
        coffate_fn = collater(opt)
        if not os.path.exists(loader_save_file):
            os.mkdir(loader_save_file)
        title_dataloader = DataLoader(BERT_set,
                                      batch_size=opt.batch_size,
                                      collate_fn=coffate_fn,
                                      shuffle=False)
        with open(os.path.join(loader_save_file, 'Dataloader_full_title.pkl'), 'wb') as f:
            dill.dump(title_dataloader, f)
    else:
        logging('Loading From Saved Dataloader')
        with open(os.path.join(loader_save_file, 'Dataloader_full_title.pkl'), 'rb') as f:
            title_dataloader = dill.load(f)
    return title_dataloader


def convert2text(opt):
    model = BERT_MLP(opt)
    logging(f'Cur_dir:{os.getcwd ()}')
    data_root = './Conver2vec/origin_data/policyinfo_new.tsv'
    if opt.pretrain_model != '':
        chkpt = torch.load(opt.pretrain_model, map_location=torch.device('cpu'))
        model.load_state_dict(chkpt['checkpoint'])
        logging('load model from {}'.format(opt.pretrain_model))
        start_epoch = chkpt['epoch'] + 1
        # lr = chkpt['lr']
        # history_best_loss = chkpt['best_loss']
        logging('resume from epoch {} '.format(start_epoch - 1))
    else:
        assert 1 == 2, 'please provide checkpoint to evaluate.'

    dataloader = create_loader(data_root,opt)

    logging('Start Evaluating')

    # model.half()
    model = get_cuda(model)
    model.eval()

    with torch.no_grad():
        step = 0
        out_tensor = None
        length = len(dataloader)
        for query in dataloader:
            query = get_bert_cuda(query)
            query_emb: torch.Tensor = model(query)
            if out_tensor is None:
                out_tensor = query_emb
            else:
                out_tensor = torch.cat([out_tensor, query_emb], axis=0)
            step += 1
            if step % opt.log_step == 0:
                logging(f'Step:{step} / {length}')
            # if step == 40:
            #     break

    if not os.path.exists(opt.full_data_save_file):
        os.mkdir(opt.full_data_save_file)
    # file = open(os.path.join(opt.full_data_save_file, 'Wordvec.txt'), mode="w")
    out_tensor = out_tensor.cpu()
    out_npy = out_tensor.numpy()
    out_npy = np.nan_to_num(out_npy)
    np.save(os.path.join(opt.full_data_save_file, 'Wordvec_body.npy'), out_npy)
    # for tens in out_tensor:
    #     print(tens.numpy().tolist(),file=file)
    # file.close()
    logging('Finish')


if __name__ == "__main__":
    print('processId:', os.getpid())
    print('prarent processId:', os.getppid())
    os.chdir('../')
    opt = get_opt()
    # with open('commandline_args.txt', 'r') as f:
    #     opt.__dict__ = json.load(f)
    convert2text(opt)
