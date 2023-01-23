import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, ChebConv, GATConv, GCNConv, global_max_pool as gmp
from SelfAttention import SelfAttention

device = torch.device('cuda:0')

def all_item_count(arr):
    result = {}
    for i in set(arr):
        result[i] = arr.count(i)
    return result

class Sequence_Model(nn.Module):
    def __init__(self, in_channel, med_channel, out_channel, kernel_size, stride, padding):
        super(Sequence_Model, self).__init__()
        self.in_channel = in_channel
        self.med_channel = med_channel
        self.out_channel = out_channel
        
        self.layers = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding),
        )
    
    def forward(self, x):
        x = self.layers(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)

class S2DTA(nn.Module):
    def __init__(self, n_output_dim, compound_feature, protein_feature, output_dim, hidden_dim, dropout, K, pocket=False, attn=False, net='SAGE'):
        super(S2DTA, self).__init__()

        self.n_output_dim = n_output_dim
        self.dropout = dropout
        self.hidden = hidden_dim
        self.K = int(K)
        self.pocket = pocket
        self.attn = attn

        self.sequence_layer = Sequence_Model(21, med_channel=[], out_channel=128, kernel_size=5, stride=1, padding=2).to(device)
        if self.pocket == True:
            self.pocket_layer = Sequence_Model(21, med_channel=[], out_channel=128, kernel_size=3, stride=1, padding=1).to(device)

        if net == 'SAGE':
            self.cconv1 = SAGEConv(compound_feature, hidden_dim, aggr='sum')
            self.cconv2 = SAGEConv(hidden_dim, hidden_dim * 2, aggr='sum')
            self.cconv3 = SAGEConv(hidden_dim * 2, hidden_dim * 4, aggr='sum')
            self.cconv_fc1 = nn.Linear(hidden_dim * 4, output_dim)
            # self.cconv_fc2 = nn.Linear(128, output_dim)
            # 定义蛋白质卷积层
            self.pconv1 = SAGEConv(protein_feature, hidden_dim, aggr='sum')
            self.pconv2 = SAGEConv(hidden_dim, hidden_dim * 2, aggr='sum')
            self.pconv3 = SAGEConv(hidden_dim * 2, hidden_dim * 4, aggr='sum')
            self.pconv_fc1 = nn.Linear(hidden_dim * 4, output_dim)
            # 定义结合口袋卷积层
            if self.pocket == True:
                self.pocket_conv1 = SAGEConv(protein_feature, hidden_dim, aggr='sum')
                self.pocket_conv2 = SAGEConv(hidden_dim, hidden_dim * 2, aggr='sum')
                self.pocket_conv3 = SAGEConv(hidden_dim * 2, hidden_dim * 4, aggr='sum')
                self.pocket_conv_fc1 = nn.Linear(hidden_dim * 4, output_dim)
        elif net == 'Cheb':
            self.cconv1 = ChebConv(compound_feature, hidden_dim, K=self.K, aggr='sum')
            self.cconv2 = ChebConv(hidden_dim, hidden_dim * 2, K=self.K, aggr='sum')
            self.cconv3 = ChebConv(hidden_dim * 2, hidden_dim * 4, K=self.K, aggr='sum')
            self.cconv_fc1 = nn.Linear(hidden_dim * 4, output_dim)
            # self.cconv_fc2 = nn.Linear(128, output_dim)
            # 定义蛋白质卷积层
            self.pconv1 = ChebConv(protein_feature, hidden_dim, K=self.K, aggr='sum')
            self.pconv2 = ChebConv(hidden_dim, hidden_dim * 2, K=self.K, aggr='sum')
            self.pconv3 = ChebConv(hidden_dim * 2, hidden_dim * 4, K=self.K, aggr='sum')
            self.pconv_fc1 = nn.Linear(hidden_dim * 4, output_dim)
            # 定义结合口袋卷积层
            if self.pocket == True:
                self.pocket_conv1 = ChebConv(protein_feature, hidden_dim, K=self.K, aggr='sum')
                self.pocket_conv2 = ChebConv(hidden_dim, hidden_dim * 2, K=self.K, aggr='sum')
                self.pocket_conv3 = ChebConv(hidden_dim * 2, hidden_dim * 4, K=self.K, aggr='sum')
                self.pocket_conv_fc1 = nn.Linear(hidden_dim * 4, output_dim)
        elif net == 'GAT':
            self.compound_gcn = nn.Sequential(
                GATConv(compound_feature, hidden_dim),
                nn.LeakyReLU(),
                GATConv(hidden_dim, hidden_dim * 2),
                nn.LeakyReLU(),
                gmp(),
                nn.Linear(hidden_dim * 4, output_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            )
            self.protein_gcn = nn.Sequential(
                GATConv(protein_feature, hidden_dim),
                nn.LeakyReLU(),
                GATConv(hidden_dim, hidden_dim * 2),
                nn.LeakyReLU(),
                gmp(),
                nn.Linear(hidden_dim * 4, output_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.compound_gcn = nn.Sequential(
                GCNConv(compound_feature, hidden_dim),
                nn.LeakyReLU(),
                GCNConv(hidden_dim, hidden_dim * 2),
                nn.LeakyReLU(),
                GCNConv(hidden_dim * 2, hidden_dim * 4),
                nn.LeakyReLU(),
                gmp(),
                nn.Linear(hidden_dim * 4, output_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            )
            self.protein_gcn = nn.Sequential(
                GCNConv(protein_feature, hidden_dim),
                nn.LeakyReLU(),
                GCNConv(hidden_dim, hidden_dim * 2),
                nn.LeakyReLU(),
                GCNConv(hidden_dim * 2, hidden_dim * 4),
                nn.LeakyReLU(),
                gmp(),
                nn.Linear(hidden_dim * 4, output_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            )

        if self.attn == True:
            self.attention = SelfAttention(self.hidden * 4, self.hidden * 4, self.hidden * 4, 8).to(device)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        if self.pocket == False:
            self.fc1 = nn.Linear(output_dim * 2, 256)
        else:
            self.fc1 = nn.Linear(output_dim * 3, 256)

        self.out = nn.Linear(256, self.n_output_dim)
    
    def forward(self, data):
        protein_sequence_feature = self.sequence_layer(data[0][0][0])
        if self.pocket == True:
            pocket_sequence_feature = self.pocket_layer(data[0][1][0])
            new_pocket_feature = torch.Tensor().to(device)
        p_feature = torch.Tensor().to(device)
        for i in range(len(protein_sequence_feature)):
            p_feature = torch.cat((p_feature, protein_sequence_feature[i][:data[0][0][1][i], :]), 0)
            if self.pocket == True:
                new_pocket_feature = torch.cat((new_pocket_feature, pocket_sequence_feature[i][:data[0][1][1][i], :]), 0)
        
        compound_feature, compound_index, compound_batch = data[1].x, data[1].edge_index, data[1].batch
        _, protein_index, protein_batch = data[2].x, data[2].edge_index, data[2].batch

        # Obtain the lengths of the drug data samples
        compound_lengths = []
        for i in range(compound_batch.cpu().tolist()[-1] + 1):
            compound_lengths.append(compound_batch.cpu().tolist().count(i))
        
        # Obtain the lengths of the target data samples
        protein_lengths = []
        for i in range(protein_batch.cpu().tolist()[-1] + 1):
            protein_lengths.append(protein_batch.cpu().tolist().count(i))
        
        # graph conv for drug
        compound_feature = self.cconv1(compound_feature, compound_index)
        compound_feature = self.relu(compound_feature)

        compound_feature = self.cconv2(compound_feature, compound_index)
        compound_feature = self.relu(compound_feature)

        compound_feature = self.cconv3(compound_feature, compound_index)
        compound_feature = self.relu(compound_feature)

        # graph conv for target
        protein_feature = self.pconv1(p_feature, protein_index)
        protein_feature = self.relu(protein_feature)

        protein_feature = self.pconv2(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)

        protein_feature = self.pconv3(protein_feature, protein_index)
        protein_feature = self.relu(protein_feature)
    
        if self.attn == True:
            compound_index = 0
            protein_index = 0
            compound_attn = torch.Tensor().to(device)
            protein_attn = torch.Tensor().to(device)
            for i in range(len(protein_lengths)):
                attn_data = torch.cat((compound_feature[compound_index: compound_index + compound_lengths[i]], protein_feature[protein_index: protein_index + protein_lengths[i]]), dim=0).to(device)
                temp_x = self.attention(attn_data.unsqueeze(0).to(device), attn_data.unsqueeze(0).to(device), attn_data.unsqueeze(0).to(device))
                compound_index += compound_lengths[i]
                protein_index += protein_lengths[i]
                attn_out = temp_x[0].squeeze(0)
                attn_score = temp_x[1].squeeze(0)
                attn_score = attn_score.mean(axis=0, keepdim=False)
                compound_attn = torch.cat((compound_attn, attn_out[:compound_lengths[i]]), 0)
                protein_attn = torch.cat((protein_attn, attn_out[compound_lengths[i]:]), 0)
            compound_feature = compound_attn
            protein_feature = protein_attn

        # pooling for drug
        compound_feature = gmp(compound_feature, compound_batch)
        # flatten
        compound_feature = self.relu(self.cconv_fc1(compound_feature))
        compound_feature = self.dropout(compound_feature)

        # pooling for target
        protein_feature = gmp(protein_feature, protein_batch)
        # flatten
        protein_feature = self.relu(self.pconv_fc1(protein_feature))
        protein_feature = self.dropout(protein_feature)

        if self.pocket == True:
            _, pocket_index, pocket_batch = data[3].x, data[3].edge_index, data[3].batch

            # graph conv for pocket
            pocket_feature = self.pocket_conv1(new_pocket_feature, pocket_index)
            pocket_feature = self.relu(pocket_feature)

            pocket_feature = self.pocket_conv2(pocket_feature, pocket_index)
            pocket_feature = self.relu(pocket_feature)

            pocket_feature = self.pocket_conv3(pocket_feature, pocket_index)
            pocket_feature = self.relu(pocket_feature)

            # 对卷积后的蛋白质进行图的最大值池化
            pocket_feature = gmp(pocket_feature, pocket_batch)

            # 对池化后的蛋白质进行flatten
            pocket_feature = self.relu(self.pocket_conv_fc1(pocket_feature))
            pocket_feature = self.dropout(pocket_feature)

            # max-pooling for pocket
            protein_feature = torch.cat((protein_feature, pocket_feature), 1)

        x = torch.cat((compound_feature, protein_feature), 1)
    
        # fully connected
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.out(x)
        return out