import argparse
import os

data_dir = './data/'
triplets_dir = os.path.join(data_dir, 'triplets_body.csv')
policy_dir = os.path.join(data_dir, 'data_sample.csv')


def get_opt():
    parser = argparse.ArgumentParser()

    # checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/')
    parser.add_argument('--loader_save_file', type=str, default='./data/prepro_data/')
    parser.add_argument('--pretrain_model', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--full_dataloader_save_file', type=str, default='')
    parser.add_argument('--full_data_save_file', type=str, default='./Conver2vec/results/')

    # Model Config
    parser.add_argument('--model_output_dim', type=int, default=200)
    parser.add_argument('--use_bert_finetune', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--index_type', type=str, default='Body')

    # training settings
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--margin', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_cosine_warm', action='store_true')
    parser.add_argument('--force_lr', action='store_true')
    parser.add_argument('--save_round', type=int, default=5)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--reload', action='store_true')

    # pretrain model name
    parser.add_argument('--tokenize_model_name', type=str, default='clue/roberta_chinese_base')
    parser.add_argument('--encode_model_name', type=str, default='clue/roberta_chinese_base')

    return parser.parse_args()
