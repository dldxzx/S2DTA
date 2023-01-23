import numpy as np
import pandas as pd
from data_process import (getCompoundGraph, getProteinGraph,
                          getSkeletonAtoms, getPocketGraph)
from util import *

train_data = pd.read_csv('data/train_data_with_pocket.csv')
val_data = pd.read_csv('data/val_data_with_pocket.csv')
test_data = pd.read_csv('data/test_data_with_pocket.csv')

train_cid = train_data['PDBID']
train_ic50 = train_data['affinity']
val_cid = val_data['PDBID']
val_ic50 = val_data['affinity']
test_cid = test_data['PDBID']
test_ic50 = test_data['affinity']

train_compound_graph = []
train_protein_graph = []
train_pocket_graph = []
val_compound_graph = []
val_protein_graph = []
val_pocket_graph = []
test_compound_graph = []
test_protein_graph = []
test_pocket_graph = []

for compound in train_cid:
    compound_info = {}
    g = getCompoundGraph(compound)
    compound_info[compound] = g
    train_compound_graph.append(compound_info)
print('训练集化合物转换完成')
for compound in val_cid:
    compound_info = {}
    g = getCompoundGraph(compound)
    compound_info[compound] = g
    val_compound_graph.append(compound_info)
print('验证集化合物转换完成')
for compound in test_cid:
    compound_info = {}
    g = getCompoundGraph(compound)
    compound_info[compound] = g
    test_compound_graph.append(compound_info)
print('测试集化合物转换完成')

for protein in train_cid:
    protein_info = {}
    g = getPocketGraph(protein)
    protein_info[protein] = g
    train_pocket_graph.append(protein_info)
print('训练集结合口袋转换完成')
for protein in val_cid:
    protein_info = {}
    g = getPocketGraph(protein)
    protein_info[protein] = g
    val_pocket_graph.append(protein_info)
print('验证集结合口袋转换完成')
for protein in test_cid:
    protein_info = {}
    g = getPocketGraph(protein)
    protein_info[protein] = g
    test_pocket_graph.append(protein_info)
print('测试集结合口袋转换完成')

for protein in test_cid:
    protein_info = {}
    g = getProteinGraph(getSkeletonAtoms(protein))
    protein_info[protein] = g
    test_protein_graph.append(protein_info)
print('测试集蛋白质转换完成')
for protein in val_cid:
    protein_info = {}
    g = getProteinGraph(getSkeletonAtoms(protein))
    protein_info[protein] = g
    val_protein_graph.append(protein_info)
print('验证集蛋白质转换完成')
for protein in train_cid:
    protein_info = {}
    g = getProteinGraph(getSkeletonAtoms(protein))
    protein_info[protein] = g
    train_protein_graph.append(protein_info)
print('训练集蛋白质转换完成')

train_key = []
val_key = []
test_key = []
for temp in train_protein_graph:
    train = ''
    for i in temp.keys():
        train = i
    train_key.append(train)

for temp in val_protein_graph:
    train = ''
    for i in temp.keys():
        train = i
    val_key.append(train)

for temp in test_protein_graph:
    test = ''
    for i in temp.keys():
        test = i
    test_key.append(test)
print(test_key)

train_compound, train_protein, train_ic50 = list(
    train_cid.values), list(train_cid.values), list(train_ic50.values)
train_compound, train_ptotein, train_ic50 = np.asarray(
    train_compound), np.asarray(train_protein), np.asarray(train_ic50)

val_compound, val_protein, val_ic50 = list(
    val_cid.values), list(val_cid.values), list(val_ic50.values)
val_compound, val_ptotein, val_ic50 = np.asarray(
    val_compound), np.asarray(val_protein), np.asarray(val_ic50)

test_compound, test_protein, test_ic50 = list(
    test_cid.values), list(test_cid.values), list(test_ic50.values)
test_compound, test_ptotein, test_ic50 = np.asarray(
    test_compound), np.asarray(test_protein), np.asarray(test_ic50)

print('准备将化合物训练集数据转化为Pytorch格式')
compound_train_data = CompoundDataset(root='data', dataset='compound_train_data',
                                      compound=train_compound, affinity=train_ic50, compound_graph=train_compound_graph)

print('准备将化合物验证集数据转化为Pytorch格式')
compound_val_data = CompoundDataset(root='data', dataset='compound_val_data',
                                    compound=val_compound, affinity=val_ic50, compound_graph=val_compound_graph)
    
print('准备将化合物测试集数据转化为Pytorch格式')
compound_test_data = CompoundDataset(root='data', dataset='compound_test_data',
                                    compound=test_compound, affinity=test_ic50, compound_graph=test_compound_graph)
    
print('准备将蛋白质验证集数据转化为Pytorch格式')
protein_val_data = ProteinDataset(root='data', dataset='protein_val_data',
                                  protein=val_protein, affinity=val_ic50, protein_graph=val_protein_graph)
    
print('准备将蛋白质测试集数据转化为Pytorch格式')
protein_test_data = ProteinDataset(root='data', dataset='protein_test_data',
                                  protein=test_protein, affinity=test_ic50, protein_graph=test_protein_graph)
    
print('准备将结合口袋训练集数据转化为Pytorch格式')
protein_train_data = ProteinDataset(root='data', dataset='pocket_train_data',
                                    protein=train_protein, affinity=train_ic50, protein_graph=train_pocket_graph)
    
print('准备将结合口袋验证集数据转化为Pytorch格式')
protein_val_data = ProteinDataset(root='data', dataset='pocket_val_data',
                                  protein=val_protein, affinity=val_ic50, protein_graph=val_pocket_graph)
    
print('准备将结合口袋测试集数据转化为Pytorch格式')
protein_test_data = ProteinDataset(root='data', dataset='pocket_test_data',
                                  protein=test_protein, affinity=test_ic50, protein_graph=test_pocket_graph)

print('准备将蛋白质训练集数据转化为Pytorch格式')
protein_train_data = ProteinDataset(root='data', dataset='protein_train_data_loader',
                                    protein=train_protein, affinity=train_ic50, protein_graph=train_protein_graph)

print('集数据转化为Pytorch格式完成')

