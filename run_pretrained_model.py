import torch
from model import S2DTA
from util import *
from evaluate_metrics import *
from training import predicting

sequence_test_data = dataset('data/test_data_with_pocket.csv')
compound_test_data = CompoundDataset(root='data', dataset='compound_test_data')
protein_test_data = ProteinDataset(root='data', dataset='protein_test_data')
pocket_test_data = ProteinDataset(root='data', dataset='pocket_test_data')
sequence_test_loader = DataLoader(dataset=sequence_test_data, batch_size=128)
compound_test_loader = DataLoader(compound_test_data, batch_size=128, shuffle=False)
protein_test_loader = DataLoader(protein_test_data, batch_size=128, shuffle=False)
pocket_test_loader = DataLoader(pocket_test_data, batch_size=128, shuffle=False)

# load best pretrained model
model = S2DTA(1, 27, 128, 256, 32, 0.4, 3, pocket=True, attn=False, net='SAGE').to(device)
model.load_state_dict(torch.load('data/pretrained_models/best_model.pt'))
label, preds = predicting(model, device, sequence_test_loader, compound_test_loader, protein_test_loader, pocket_test_loader, pocket=True, isTest=True)
result = [mae(label, preds), rmse(label, preds), pearson(label, preds), spearman(label, preds), ci(label, preds), r_squared(label, preds)]
print(result)