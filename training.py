import torch
import torch.nn as nn
from model import S2DTA
from util import *
from util import _to_onehot
from evaluate_metrics import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

weight_decay = 5e-4
LOG_INTERVAL = 20
# 损失函数
criterion = nn.MSELoss().to(device)

def training(model, device, sequence_loader, compound_train_loader, protein_train_loader, pocket_train_loader, optimizer, epoch, pocket=False):
    print('Current Epoch:', epoch)
    model.train()
    sequence_list = []
    compound_list = []
    protein_list = []
    pocket_list = []
    for batch, sequence_data in enumerate(sequence_loader):
        sequence_list.append(sequence_data)
    for batch, data in enumerate(compound_train_loader):
        compound_list.append(data)
    for batch, data in enumerate(protein_train_loader):
        protein_list.append(data)
    if pocket == True:
        for batch, data in enumerate(pocket_train_loader):
            pocket_list.append(data)
    total_loss = 0
    num_data = len(compound_train_loader)
    for i in range(len(compound_list)):
        if pocket == True:
            seq_data = _to_onehot(sequence_list[i][2], 2100), _to_onehot(sequence_list[i][3], 125)
            data = seq_data, compound_list[i].to(device), protein_list[i].to(device), pocket_list[i].to(device)
        else:
            seq_data = [_to_onehot(sequence_list[i][2], 2100)]
            data = seq_data, compound_list[i].to(device), protein_list[i].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data[1].y.view(-1, 1).float().to(device))
        total_loss += loss
        loss.backward()
        optimizer.step()
    print('Epoch: {}, mean_loss: {}'.format(epoch, total_loss / num_data))
    

def predicting(model, device, sequence_loader, compound_loader, protein_loader, pocket_loader, pocket=False, isTest=False):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    sequence_list = []
    compound_list = []
    protein_list = []
    pocket_list = []
    with torch.no_grad():
        for batch, sequence_data in enumerate(sequence_loader):
            sequence_list.append(sequence_data)
        for batch, data in enumerate(compound_loader):
            compound_list.append(data)
        for batch, data in enumerate(protein_loader):
            protein_list.append(data)
        if pocket == True:
            for batch, data in enumerate(pocket_loader):
                pocket_list.append(data)
        for i in range(len(compound_list)):
            if pocket == True:
                seq_data = _to_onehot(sequence_list[i][2], 2100), _to_onehot(sequence_list[i][3], 125)
                data = seq_data, compound_list[i].to(device), protein_list[i].to(device),  pocket_list[i].to(device)
            else:
                seq_data = [_to_onehot(sequence_list[i][2], 2100)]
                data = seq_data, compound_list[i].to(device), protein_list[i].to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data[1].y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

if __name__ == '__main__':
    # load sequence data
    sequence_train_data = dataset('data/train_data_with_pocket.csv')
    sequence_val_data = dataset('data/val_data_with_pocket.csv')
    sequence_test_data = dataset('data/test_data_with_pocket.csv')
    # load graph data
    compound_train_data = CompoundDataset(root='data', dataset='compound_train_data')
    compound_val_data = CompoundDataset(root='data', dataset='compound_val_data')
    compound_test_data = CompoundDataset(root='data', dataset='compound_test_data')
    protein_train_data = ProteinDataset(root='data', dataset='protein_train_data')
    protein_val_data = ProteinDataset(root='data', dataset='protein_val_data')
    protein_test_data = ProteinDataset(root='data', dataset='protein_test_data')
    pocket_train_data = ProteinDataset(root='data', dataset='pocket_train_data')
    pocket_val_data = ProteinDataset(root='data', dataset='pocket_val_data')
    pocket_test_data = ProteinDataset(root='data', dataset='pocket_test_data')

    batch_size = 128
    sequence_train_loader = DataLoader(dataset=sequence_train_data, batch_size=batch_size)
    sequence_val_loader = DataLoader(dataset=sequence_val_data, batch_size=batch_size)
    sequence_test_loader = DataLoader(dataset=sequence_test_data, batch_size=batch_size)
    compound_train_loader = DataLoader(compound_train_data, batch_size=batch_size, shuffle=False)
    compound_val_loader = DataLoader(compound_val_data, batch_size=batch_size, shuffle=False)
    compound_test_loader = DataLoader(compound_test_data, batch_size=batch_size, shuffle=False)
    protein_train_loader = DataLoader(protein_train_data, batch_size=batch_size, shuffle=False)
    protein_val_loader = DataLoader(protein_val_data, batch_size=batch_size, shuffle=False)
    protein_test_loader = DataLoader(protein_test_data, batch_size=batch_size, shuffle=False)
    pocket_train_loader = DataLoader(pocket_train_data, batch_size=batch_size, shuffle=False)
    pocket_val_loader = DataLoader(pocket_val_data, batch_size=batch_size, shuffle=False)
    pocket_test_loader = DataLoader(pocket_test_data, batch_size=batch_size, shuffle=False)

    is_pocket = True
    is_attn = True

    model = S2DTA(1, 27, 128, 256, 32, 0.4, 3, pocket=is_pocket, attn=is_attn, net='SAGE').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_rmse = 9999
    print(model)
    for epoch in range(500):
        training(model, device, sequence_train_loader, compound_train_loader, protein_train_loader, pocket_train_loader,
                optimizer, epoch + 1, pocket=is_pocket)
        val_label, val_preds = predicting(model, device, sequence_val_loader, compound_val_loader, protein_val_loader,
                                        pocket_val_loader, pocket=is_pocket)
        val_result = [mae(val_label, val_preds), rmse(val_label, val_preds), pearson(val_label, val_preds),
                    spearman(val_label, val_preds), ci(val_label, val_preds), r_squared(val_label, val_preds)]
        label, preds = predicting(model, device, sequence_test_loader, compound_test_loader, protein_test_loader,
                                pocket_test_loader, pocket=is_pocket, isTest=True)
        result = [mae(label, preds), rmse(label, preds), pearson(label, preds), spearman(label, preds),
                ci(label, preds), r_squared(label, preds)]
        print(result)
        with open('result.txt', 'a') as write:
                write.writelines(str(result) + '\n')
        if result[1] < best_rmse:
            best_rmse = result[1]
            # save model
            torch.save(model, 'best_model.pt')
            # save parameters
            torch.save(model.state_dict(), 'best_model_param.pt')