import os

import dill
import torch
from data_inference import BERT_Dataset, collater
from torch.utils.data import DataLoader
from config import *
from model import *
from utils import *


def evaluate(model, dataloader):
    with torch.no_grad():
        step = 0
        lossF = triplet_loss(margin=1)
        for query, positive, negative in dataloader:
            query = get_bert_cuda(query)
            positive = get_bert_cuda(positive)
            negative = get_bert_cuda(negative)
            query_emb = model(query)
            positive_emb = model(positive)
            negative_emb = model(negative)
            loss = lossF(query_emb,positive_emb,negative_emb)
            print('Anchor - Positive: ', get_tensor_distance(query_emb, positive_emb))
            print('Anchor - Negative: ', get_tensor_distance(query_emb, negative_emb))
            logging('| step {:4d} | Loss {:3.5f} '.format(step, loss.item()))
            step += 1
            # if step == 20:
            #     break


def test(opt):
    model = BERT_MLP(opt)
    loader_save_file = opt.loader_save_file

    if opt.pretrain_model != '':
        chkpt = torch.load(opt.pretrain_model, map_location=torch.device('cpu'))
        model.load_state_dict(chkpt['checkpoint'])
        logging('load model from {}'.format(opt.pretrain_model))
        start_epoch = chkpt['epoch'] + 1
        lr = chkpt['lr']
        # history_best_loss = chkpt['best_loss']
        logging('resume from epoch {} with lr {}'.format(start_epoch, lr))
    else:
        assert 1 == 2, 'please provide checkpoint to evaluate.'

    model.half()
    model = get_cuda(model)
    model.eval()

    coffate_fn = collater(opt)
    if (not os.path.exists(os.path.join(loader_save_file, 'Inference_Dataloader.pkl'))) or opt.reload:
        logging('Read From Origin Data')
        triplets = pd.read_csv(triplets_dir, header=None)
        data = pd.read_csv(policy_dir, index_col=0)
        change2vec(data)
        data['PUB_TIME'] = pd.to_datetime(data['PUB_TIME']).apply(lambda x: x.value / 1e14)
        data['UPDATE_DATE'] = pd.to_datetime(data['UPDATE_DATE']).apply(lambda x: x.value / 1e14)
        BERT_set = BERT_Dataset(data, triplets)
        if not os.path.exists(loader_save_file):
            os.mkdir(loader_save_file)
        title_dataloader = DataLoader(BERT_set,
                                      batch_size=opt.batch_size,
                                      collate_fn=coffate_fn,
                                      drop_last=True,
                                      shuffle=True)
        with open(os.path.join(loader_save_file, 'Inference_Dataloader.pkl'), 'wb') as f:
            dill.dump(title_dataloader, f)
    else:
        logging('Loading From Saved Dataloader')
        with open(os.path.join(loader_save_file, 'Inference_Dataloader.pkl'), 'rb') as f:
            title_dataloader = dill.load(f)
    # if os.path.exists(loader_save_file):
    #     logging('Loading From Saved Dataloader')
    #     with open(os.path.join(loader_save_file, 'Dataloader.pkl'), 'rb') as f:
    #         title_dataloader = dill.load(f)
    # else:
    #     assert 1 == 2, 'please provide DataloaderFile to evaluate.'

    logging('Start Evaluating')
    evaluate(model, title_dataloader)


if __name__ == "__main__":
    print('processId:', os.getpid())
    print('prarent processId:', os.getppid())
    opt = get_opt()
    test(opt)