import json

import dill
import pandas as pd
import numpy as np
from torch.optim import AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from config import *
from utils import *
from model import BERT_MLP
from data import BERT_Dataset, collater
from torch.utils.data import DataLoader
from torch.nn.functional import triplet_margin_loss
from tqdm import tqdm
from matplotlib import pyplot as plt


def train(opt):
    lr = opt.lr
    loader_save_file = os.path.join(opt.loader_save_file, f'Dataloader_{opt.index_type}.pkl')
    coffate_fn = collater(opt)
    if (not os.path.exists(loader_save_file)) or opt.reload:
        logging('Read From Origin Data')
        triplets = pd.read_csv(triplets_dir, header=None)
        data = pd.read_csv(policy_dir, index_col=0)
        change2vec(data)
        data['PUB_TIME'] = pd.to_datetime(data['PUB_TIME']).apply(lambda x: x.value / 1e14)
        data['UPDATE_DATE'] = pd.to_datetime(data['UPDATE_DATE']).apply(lambda x: x.value / 1e14)
        BERT_set = BERT_Dataset(data, triplets, opt)
        if not os.path.exists(loader_save_file):
            os.mkdir(opt.loader_save_file)
        title_dataloader = DataLoader(BERT_set,
                                      batch_size=opt.batch_size,
                                      collate_fn=coffate_fn,
                                      drop_last=True,
                                      shuffle=True)
        with open(loader_save_file, 'wb') as f:
            dill.dump(title_dataloader, f)
    else:
        logging('Loading From Saved Dataloader')
        with open(loader_save_file, 'rb') as f:
            title_dataloader = dill.load(f)
    model = BERT_MLP(opt)
    start_epoch = 1

    history_best_loss = 10000
    if opt.pretrain_model != '':
        chkpt = torch.load(opt.pretrain_model, map_location=torch.device('cpu'))
        model.load_state_dict(chkpt['checkpoint'])
        logging('load model from {}'.format(opt.pretrain_model))
        start_epoch = chkpt['epoch'] + 1
        if not opt.force_lr:
            lr = chkpt['lr']
        history_best_loss = chkpt['best_loss']
        logging('resume from epoch {} with lr {}'.format(start_epoch, lr))
    else:
        logging('training from scratch with lr {}'.format(lr))

    bert_param_ids = list(map(id, model.bert.parameters()))
    base_params = filter(lambda p: p.requires_grad and id(p) not in bert_param_ids, model.parameters())

    optimizer = SGD([
        {'params': model.bert.parameters(), 'lr': lr * 0.01, 'initial_lr': opt.lr},
        {'params': base_params, 'weight_decay': opt.weight_decay, 'initial_lr': opt.lr}
    ], lr=lr)

    if opt.lr_cosine_warm:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    checkpoint_dir = opt.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    train_loss_list = []

    # start training
    logging('Begin')
    model.half()
    model = get_cuda(model)
    lossF = triplet_loss(margin=opt.margin)
    t = 0
    for epoch in range(start_epoch, opt.epochs + 1):
        model.train()
        for query, positive, negative in title_dataloader:
            optimizer.zero_grad()
            cur_lr = optimizer.param_groups[1]['lr']
            query = get_bert_cuda(query)
            # print(query)
            positive = get_bert_cuda(positive)
            negative = get_bert_cuda(negative)
            query_emb = model(query)
            positive_emb = model(positive)
            negative_emb = model(negative)
            loss = triplet_margin_loss(query_emb, positive_emb, negative_emb, margin=opt.margin, reduction='mean', p=2, swap=True)
            train_loss_list.append(loss.item())
            # loss = lossF(query_emb, positive_emb, negative_emb)
            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), 3, norm_type=2)
            optimizer.step()
            if t % opt.log_step == 0:
                logging('| epoch {:2d} | step {:4d} | lr {:.4E} | Train Loss {:3.5f} '.format(epoch, t, cur_lr,
                                                                                              loss.item()))
            t += 1
        if opt.lr_cosine_warm:
            scheduler.step()
        # evaluate
        # with torch.no_grad():
        #     model.eval()
        #     total_loss = 0
        #     logging('-' * 89)
        #     logging('evaluating')
        #     for query, positive, negative in title_dataloader:
        #         query = get_bert_cuda(query)
        #         positive = get_bert_cuda(positive)
        #         negative = get_bert_cuda(negative)
        #         query_emb = model(query)
        #         positive_emb = model(positive)
        #         negative_emb = model(negative)
        #         # loss = triplet_margin_loss(query_emb, positive_emb, negative_emb, margin=opt.margin, reduction='mean')
        #         loss = lossF(query_emb, positive_emb, negative_emb)
        #         total_loss += loss.item()
        #     total_loss /= len(title_dataloader)
        #     logging(f"Epoch:{epoch}, Average Evaluate Loss:{total_loss}")
        #     logging('-' * 89)
        #     if total_loss < history_best_loss:
        #         history_best_loss = total_loss
        #         path = os.path.join(checkpoint_dir, opt.model_name + '_best.pt')
        #         torch.save({
        #             'epoch': epoch,
        #             'checkpoint': model.state_dict(),
        #             'lr': lr,
        #             'best_loss': total_loss,
        #             'best_epoch': epoch
        #         }, path)
        #     if epoch % opt.save_round == 0:
        #         path = os.path.join(checkpoint_dir, opt.model_name + f'_epoch{epoch}.pt')
        #         torch.save({
        #             'epoch': epoch,
        #             'checkpoint': model.state_dict(),
        #             'lr': lr,
        #             'loss': total_loss,
        #             'best_loss': history_best_loss,
        #         }, path)
        plt.plot([step for step in range(t)], train_loss_list)
        plt.savefig(r"./logs/loss_fig.png")
        dataframe = pd.DataFrame({'loss_item': train_loss_list})
        dataframe.to_csv("loss.csv", index=False, sep=',')
        


if __name__ == '__main__':
    print('processId:', os.getpid())
    print('prarent processId:', os.getppid())
    opt = get_opt()
    print(json.dumps(opt.__dict__, indent=4))
    train(opt)
