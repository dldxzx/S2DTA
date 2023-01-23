import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric import data as DATA

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

smile_seq_max = 212
protein_seq_max = 2100
pocket_seq_max = 125
amino_acids = ['Pad', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
atom_list = ['Pad', 'C', 'H', 'O', 'N', 'F', 'S', 'P', 'I', 'Cl', 'As', 'Se', 'Br', 'B', 'Pt', 'V', 'Fe', 'Hg', 'Rh', 'Mg', 'Be', 'Si', 'Ru', 'Sb', 'Cu', 'Re', 'Ir', 'Os']

device = torch.device('cuda')

# 化合物SMILES转化为字符编码
def smiles_to_int(smile):
    smiles_int = []
    for i in range(len(smile)):
        if smile[i].isalpha() and (smile[i].isupper() or smile[i] == 'c' or smile[i] == 'o' or smile[i] == 'n' or smile[i] == 's'):
            temp = smile[i].upper()
            if (i + 1) < len(smile) and (smile[i] == 'A' and smile[i + 1] == 's') or ((i + 1) < len(smile) and smile[i] == 'O' and smile[i + 1] == 's'):
                temp += smile[i + 1]
            elif ((smile[i - 1] == 'A' and smile[i] == 's') or ((i - 1) >= 0 and smile[i - 1] == 'O' and smile[i] == 's')):
                continue
            elif i + 1 < len(smile) and smile[i + 1].islower() and smile[i + 1] != 'c' and smile[i + 1] !='o' and smile[i + 1] != 'n' and smile[i + 1] != 's' and smile[i] != 'A':
                temp += smile[i + 1]
        else:
            continue
        index = [i for i, x in enumerate(atom_list) if x == temp]
        if len(index) != 0:
            smiles_int.append([index[0]])
    return smiles_int
# 氨基酸序列转为字符编码
def protein_to_int(protein):
    protein_int = []
    for i in range(len(protein)):
        temp = protein[i]
        index = [i for i, x in enumerate(amino_acids) if x == temp]
        if len(index) != 0:
            protein_int.append([index[0]])
    return protein_int
# 化合物SMILES转为独热编码
def smiles_to_onehot(seq):
    integer_encoded = smiles_to_int(seq)
    for i in range(len(integer_encoded)):
        integer_encoded[i] = integer_encoded[i][0]
    # print(integer_encoded)
    onehot_encoded = []
    for value in integer_encoded:
        letter = [0 for _ in range(len(atom_list))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded
# 氨基酸序列转为独热编码
def protein_to_onehot(seq):
    protein_to_int = dict((c, i) for i, c in enumerate(amino_acids))
    integer_encoded = [protein_to_int[char] for char in seq]
    # print(integer_encoded)
    onehot_encoded = []
    for value in integer_encoded:
        letter = [0 for _ in range(len(amino_acids))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded
# 对序列的字符编码进行PADDING
def _to_label(data, max_len):
    label_encoder = []
    for seq in data:
        if max_len == 2100:
            label = protein_to_int(seq)
            label_encoder.append(label)
        elif max_len == 212:
            label = smiles_to_int(seq)
            label_encoder.append(label)
        else:
            label = protein_to_int(seq)
            label_encoder.append(label)
    for i in range(len(label_encoder)):
        if len(label_encoder[i]) != max_len:
            if max_len == 2100:
                append_len = [[0]] * (max_len - len(label_encoder[i]))
                member = label_encoder[i] + append_len
                label_encoder[i] = member
            elif max_len == 212:
                append_len = [[0]] * (max_len - len(label_encoder[i]))
                member = label_encoder[i] + append_len
                label_encoder[i] = member
            else:
                append_len = [[0]] * (max_len - len(label_encoder[i]))
                member = label_encoder[i] + append_len
                label_encoder[i] = member
    return torch.from_numpy(np.array(label_encoder, dtype=np.float32)).to(device)
# 对序列的独热编码进行PADDING
def _to_onehot(data, max_len):
    onehot_encoder = []
    lengths = []
    for seq in data:
        if max_len == 2100:
            if len(seq) > 2100:
                seq = seq[:2100]
            onehot = protein_to_onehot(seq)
            onehot_encoder.append(onehot)
        elif max_len == 212:
            onehot = smiles_to_onehot(seq)
            onehot_encoder.append(onehot)
        else:
            onehot = protein_to_onehot(seq)
            onehot_encoder.append(onehot)
        lengths.append(len(onehot))
    for i in range(len(onehot_encoder)):
        if len(onehot_encoder[i]) != max_len:
            for j in range(max_len - len(onehot_encoder[i])):
                if max_len == 2100:
                    temp = [0] * len(amino_acids)
                elif max_len == 212:
                    temp = [0] * len(atom_list)
                else: 
                    temp = [0] * len(amino_acids)
                temp[0] = 1
                onehot_encoder[i].append(temp)
    # return torch.LongTensor(lengths)
    return torch.from_numpy(np.array(onehot_encoder, dtype=np.float32)).to(device), torch.LongTensor(lengths)


class dataset(Dataset):
    def __init__(self, filepath):
        data = pd.read_csv(filepath)
        self.data_len = len(data)
        self.id = data['PDBID']
        self.smile = data['Smiles']
        self.protein = data['Sequence']
        self.pocket = data['Pocket']
        self.affinity = torch.from_numpy(np.array(data['affinity']).astype(np.float32))

    def __getitem__(self, index):
        return self.id[index], self.smile[index], self.protein[index], self.pocket[index], self.affinity[index]
    
    def __len__(self):
        return self.data_len

class CompoundDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset=None , compound=None, protein=None, affinity=None, transform=None, pre_transform=None, compound_graph=None, protein_graph=None):
        super(CompoundDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processd data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processd data not found: {}, doing pre-processing ...'.format(self.processed_paths[0]))
            self.process(compound, affinity, compound_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        pass
    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']
    
    def download(self):
        # download_url(url='', self.raw_dir)
        pass
    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    # def process(self, compound, protein, affinity, compound_graph, protein_graph):
    def process(self, compound, affinity, compound_graph):
        # assert (len(compound) == len(protein) ==  len(affinity)), '这三个列表必须是相同的长度!'
        assert (len(compound) == len(affinity)), '这两个列表必须是相同的长度!'
        data_list = []
        data_len = len(compound)
        for i in range(data_len):
            print('将分子格式转换为图结构：{}/{}'.format(i + 1, data_len))
            smiles = compound[i]
            # target = protein[i]
            label = affinity[i]
            print(smiles)
            # print(target)
            print(label)

            size, features, edge_index = compound_graph[i][smiles]
            # p_size, p_features, p_edge_index = protein_graph[i][target]
            GCNCompound = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(-1, 0), y=torch.FloatTensor([label]))
            # GCNProtein = DATA.Data(x=torch.Tensor(p_features), edge_index=torch.LongTensor(p_edge_index).transpose(1, 0), y=torch.FloatTensor([label]))
            # GCNData.target = torch.Tensor([p_features])
            # GCNData.protein_edge = torch.LongTensor([p_edge_index])
            GCNCompound.__setitem__('size', torch.LongTensor([size]))
            # GCNProtein.__setitem__('size', torch.LongTensor([p_size]))
            # GCNData.p_feature = torch.Tensor(p_features)
            # GCNData.p_edge_index = torch.LongTensor(p_edge_index).transpose(1, 0)
            data_list.append(GCNCompound)
            # data_list.append(GCNProtein)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('将构建完的图信息保存到文件中')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class ProteinDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset=None, protein=None, affinity=None, transform=None, pre_transform=None, protein_graph=None):
        super(ProteinDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processd data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processd data not found: {}, doing pre-processing ...'.format(self.processed_paths[0]))
            self.process(protein, affinity, protein_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        pass
    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']
    
    def download(self):
        # download_url(url='', self.raw_dir)
        pass
    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def process(self, protein, affinity, protein_graph):
        assert (len(protein) ==  len(affinity)), '这三个列表必须是相同的长度!'
        data_list = []
        data_len = len(protein)
        for i in range(data_len):
            print('将分子格式转换为图结构：{}/{}'.format(i + 1, data_len))
            # smiles = compound[i]
            target = protein[i]
            label = affinity[i]
            # print(smiles)
            print(target)
            print(label)

            # c_size, c_features, c_edge_index = compound_graph[i][smiles]
            # p_size, p_features, p_edge_index = protein_graph[i][target]
            size, features, edge_index = protein_graph[i][target]
            GCNProtein = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(-1, 0), y=torch.FloatTensor([label]))
            # GCNProtein = DATA.Data(x=torch.Tensor(p_features), edge_index=torch.LongTensor(p_edge_index).transpose(1, 0), y=torch.FloatTensor([label]))
            # GCNData.target = torch.Tensor([p_features])
            # GCNData.protein_edge = torch.LongTensor([p_edge_index])
            GCNProtein.__setitem__('size', torch.LongTensor([size]))
            # GCNProtein.__setitem__('size', torch.LongTensor([p_size]))
            # GCNData.p_feature = torch.Tensor(p_features)
            # GCNData.p_edge_index = torch.LongTensor(p_edge_index).transpose(1, 0)
            data_list.append(GCNProtein)
            # data_list.append(GCNProtein)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('将构建完的图信息保存到文件中')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
