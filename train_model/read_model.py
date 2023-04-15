import torch
file_root = r'./checkpoint/BERT_DoubleMLP_Body_best.pt'

chpt = torch.load(file_root)

print(chpt['best_epoch'],chpt['best_loss'])