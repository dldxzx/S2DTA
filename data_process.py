import math
import pdb
import numpy as np
import pandas as pd
# numpy打包工具
# 简化构建图结构的工具
import os
import Bio.PDB as PDB
from Bio.PDB.PDBParser import PDBParser
import warnings

warnings.filterwarnings("ignore")

parser = PDBParser(PERMISSIVE=1)
# 定义氨基酸
amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
# 定义SMILES格式
atom_list = ['C', 'H', 'O', 'N', 'F', 'S', 'P', 'I', 'Cl', 'As', 'Se', 'Br', 'B', 'Pt', 'V', 'Fe', 'Hg', 'Rh', 'Mg', 'Be', 'Si', 'Ru', 'Sb', 'Cu', 'Re', 'Ir', 'Os']
atom_outest_ele = [4, 1, 6, 5, 7, 6, 5, 7, 7, 5, 6, 7, 3, 2, 5, 2, 2, 2, 2, 2, 4, 1, 5, 1, 2, 1, 2]

Amino_acids = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
Amino_acids_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
Amino_acids_dict = {}
for i in range(len(Amino_acids)):
    Amino_acids_dict[Amino_acids[i]] = Amino_acids_num[i]

stru_dict = {'B': 0, 'C': 1, 'E': 2, 'G': 3, 'H': 4, 'I': 5, 'S': 6, 'T': 7}

def protein_to_int(protein):
    protein_int = []
    for i in range(len(protein)):
        temp = protein[i]
        index = [i for i, x in enumerate(amino_acids) if x == temp]
        if len(index) != 0:
            protein_int.append(index[0])
    return protein_int

def protein_to_onehot(seq):
    protein_to_int = dict((c, i) for i, c in enumerate(amino_acids))
    integer_encoded = [protein_to_int[char] for char in seq]
    onehot_encoded = []
    for value in integer_encoded:
        letter = [0 for _ in range(len(amino_acids))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded

def smiles_to_int(smile):
    smiles_int = []
    for i in range(len(smile)):
        if smile[i].isalpha() and (smile[i].isupper() or smile[i] == 'c' or smile[i] == 'o' or smile[i] == 'n' or smile[i] == 's'):
            temp = smile[i].upper()
            if (smile[i] == 'A' and smile[i + 1] == 's') or ((i + 1) < len(smile) and smile[i] == 'O' and smile[i + 1] == 's'):
                temp += smile[i + 1]
            elif ((smile[i - 1] == 'A' and smile[i] == 's') or ((i - 1) >= 0 and smile[i - 1] == 'O' and smile[i] == 's')):
                continue
            elif i + 1 < len(smile) and smile[i + 1].islower() and smile[i + 1] != 'c' and smile[i + 1] !='o' and smile[i + 1] != 'n' and smile[i + 1] != 's' and smile[i] != 'A':
                temp += smile[i + 1]
        else:
            continue
        # index = smiles.index(temp)
        index = [i for i, x in enumerate(atom_list) if x == temp]
        if len(index) != 0:
            smiles_int.append(index[0])
        # else:
        #     smiles_int.append(len(atom_list) - 1)
    return smiles_int

def get_protein_sequence(pdbid):
    sequence = ''
    p_str = parser.get_structure(pdbid, 'data/protein_pdb/' + pdbid + '.pdb')
    for chain in p_str[0]:
        if chain.get_id() != ' ':
            for residue in chain:
                if residue.has_id('CA'):
                    if str(residue.get_resname()) in Amino_acids:
                        sequence += amino_acids[Amino_acids_dict[residue.get_resname()]]
    return sequence

def load_pdb(pdbid):
    pocket_info_list = []
    with open('data/pocket_pdb/' + pdbid + '.pdb', 'r') as read:
        lines = read.readlines()
        for line in lines:
            if line[:4] == 'ATOM':
                line = list(filter(None, line.split(' ')))
                if 'CA' in line:
                    if len(line[4]) != 1 and line[4][-1].isalpha():
                        pocket_info = line[3], int(line[4][:-1]), float(line[5]), float(line[6]), float(line[7])
                    elif len(line[4]) != 1 and line[4][0].isalpha():
                        pocket_info = line[3], int(line[4][1:]), float(line[5]), float(line[6]), float(line[7])
                    elif len(line[5]) != 1 and line[5][-1].isalpha():
                        if len(line[6]) >= 10:
                            temp = line[6].split('-')
                            if len(temp) == 2:
                                pocket_info = line[3], int(line[5][:-1]), float(temp[0]), 0 - float(temp[1]), float(line[7])
                            else:
                                pocket_info = line[3], int(line[5][:-1]), 0 - float(temp[1]), 0 - float(temp[2]), float(line[7])
                        elif len(line[7]) >= 10:
                            temp = line[7].split('-')
                            if len(temp) == 2:
                                pocket_info = line[3], int(line[5][:-1]), float(line[6]), float(temp[0]), 0 - float(temp[1])
                            else:
                                pocket_info = line[3], int(line[5][:-1]), float(line[6]), 0 - float(temp[1]), 0 - float(temp[2])
                        else:
                            pocket_info = line[3], int(line[5][:-1]), float(line[6]), float(line[7]), float(line[8])
                    elif len(line[5]) != 1 and line[5][0].isalpha():
                        pocket_info = line[3], int(line[5][1:]), float(line[6]), float(line[7]), float(line[8])
                    elif len(line[4]) == 1 and line[4].isalpha():
                        if len(line[6]) >= 10:
                            temp = line[6].split('-')
                            if line[4].isalpha():
                                if len(temp) == 2:
                                    pocket_info = line[3], int(line[5]), float(temp[0]), 0 - float(temp[1]), float(line[7])
                                elif len(temp) == 3:
                                    pocket_info = line[3], int(line[5]), 0 - float(temp[1]), 0 - float(temp[2]), float(line[7])
                                else:
                                    temp = line[6].split('.')
                                    pocket_info = line[3], int(line[5]), float(temp[0] + '.' + temp[1][:3]), float(temp[1][3:] + '.' + temp[2]), float(line[7])
                            else:
                                if len(temp) == 2:
                                    pocket_info = line[3], int(line[4]), float(temp[0]), 0 - float(temp[1]), float(line[7])
                                elif len(temp) == 3:
                                    pocket_info = line[3], int(line[4]), 0 - float(temp[1]), 0 - float(temp[2]), float(line[7])
                                else:
                                    temp = line[6].split('.')
                                    pocket_info = line[3], int(line[4]), float(temp[0] + '.' + temp[1][:3]), float(temp[1][3:] + '.' + temp[2]), float(line[7])
                        elif len(line[7]) >= 10:
                            temp = line[7].split('-')
                            if line[4].isalpha():
                                if len(temp) == 2:
                                    pocket_info = line[3], int(line[5]), float(line[6]), float(temp[0]), 0 - float(temp[1])
                                elif len(temp) == 3:
                                    pocket_info = line[3], int(line[5]), float(line[6]), 0 - float(temp[1]), 0 - float(temp[2])
                                else:
                                    temp = line[7].split('.')
                                    pocket_info = line[3], int(line[5]), float(line[6]), float(temp[0] + '.' + temp[1][:3]), float(temp[1][3:] + '.' + temp[2])
                            else:
                                if len(temp) == 2:
                                    pocket_info = line[3], int(line[4]), float(line[6]), float(temp[0]), 0 - float(temp[1])
                                elif len(temp) == 3:
                                    pocket_info = line[3], int(line[4]), float(line[6]), 0 - float(temp[1]), 0 - float(temp[2])
                                else:
                                    temp = line[7].split('.')
                                    pocket_info = line[3], int(line[4]), float(line[6]), float(temp[0] + '.' + temp[1][:3]), float(temp[1][3:] + '.' + temp[2])
                        else:
                            pocket_info = line[3], int(line[5]), float(line[6]), float(line[7]), float(line[8])
                    else:
                        if len(line[5]) >= 10:
                            temp = line[5].split('-')
                            if len(temp) == 2:
                                pocket_info = line[3], int(line[4]), float(temp[0]), 0 - float(temp[1]), float(line[6])
                            elif len(temp) == 3:
                                pocket_info = line[3], int(line[4]), 0 - float(temp[1]), 0 - float(temp[2]), float(line[6])
                            else:
                                temp = line[5].split('.')
                                if len(temp) == 3:
                                    pocket_info = line[3], int(line[4]), float(temp[0] + '.' + temp[1][:3]), float(temp[1][3:] + '.' + temp[2]), float(line[6])
                                else:
                                    pocket_info = line[3], int(line[4]), float(temp[0] + '.' + temp[1][:3]), float(temp[1][3:] + '.' + temp[2][:3]), float(temp[2][3:] + '.' + temp[3])
                        elif len(line[6]) >= 10:
                            temp = line[6].split('-')
                            if len(temp) == 2:
                                pocket_info = line[3], int(line[4]), float(line[5]), float(temp[0]), 0 - float(temp[1])
                            elif len(temp) == 3:
                                pocket_info = line[3], int(line[4]), float(line[5]), 0 - float(temp[1]), 0 - float(temp[2])
                            else:
                                temp = line[6].split('.')
                                pocket_info = line[3], int(line[4]), float(line[5]), float(temp[0] + '.' + temp[1][:3]), float(temp[1][3:] + '.' + temp[2])
                        else:
                            pocket_info = line[3], int(line[4]), float(line[5]), float(line[6]), float(line[7])
                    pocket_info_list.append(list(pocket_info))
    return pocket_info_list

# 获取全局蛋白质的相关信息
def get_global_position(pdbid):
    protein_info_list = []
    p_str = parser.get_structure(pdbid, 'data/protein_pdb/' + pdbid + '.pdb')
    for chain in p_str[0]:
        if chain.get_id() != ' ':
            for residue in chain:
                if residue.has_id('CA'):
                    if str(residue.get_resname()) in Amino_acids:
                        residue_info = residue.get_resname(), int(residue.get_id()[1]), round(float(residue['CA'].get_coord()[0]), 3), round(float(residue['CA'].get_coord()[1]), 3), round(float(residue['CA'].get_coord()[2]), 3)
                        protein_info_list.append(list(residue_info))
    return protein_info_list
# 获取结合口袋的相关信息
def get_pocket_position(pdbid):
    pocket_info_list = []
    pocket_info = []
    with open('data/pocket_pdb/' + pdbid + '.pdb', 'r') as read:
        lines = read.readlines()
        for line in lines:
            if line[:4] == 'ATOM':
                line = list(filter(None, line.split(' ')))
                if 'CA' in line:
                    if len(line[4]) != 1 and line[4][-1].isupper():
                        pocket_info = line[3], int(line[4][:-1]), float(line[5]), float(line[6]), float(line[7])
                    elif len(line[4]) != 1 and line[4][0].isalpha():
                        pocket_info = line[3], int(line[4][1:]), float(line[5]), float(line[6]), float(line[7])
                    elif len(line[5]) != 1 and line[5][-1].isupper():
                        if len(line[6]) >= 10 and len(line[6]) < 18:
                            temp = line[6].split('-')
                            if len(temp) == 2:
                                pocket_info = line[3], int(line[5][:-1]), float(temp[0]), 0 - float(temp[1]), float(line[7])
                            else:
                                pocket_info = line[3], int(line[5][:-1]), 0 - float(temp[1]), 0 - float(temp[2]), float(line[7])
                        elif len(line[7]) >= 10:
                            temp = line[7].split('-')
                            if len(temp) == 2:
                                pocket_info = line[3], int(line[5][:-1]), float(line[6]), float(temp[0]), 0 - float(temp[1])
                            else:
                                pocket_info = line[3], int(line[5][:-1]), float(line[6]), 0 - float(temp[1]), 0 - float(temp[2])
                        else:
                            pocket_info = line[3], int(line[5][:-1]), float(line[6]), float(line[7]), float(line[8])
                    elif len(line[5]) != 1 and line[5][0].isalpha():
                        pocket_info = line[3], int(line[5][1:]), float(line[6]), float(line[7]), float(line[8])
                    elif len(line[4]) == 1 and line[4].isupper():
                        if len(line[6]) >= 10 and len(line[6]) < 18:
                            temp = line[6].split('-')
                            if line[4].isupper():
                                if len(temp) == 2:
                                    pocket_info = line[3], int(line[5]), float(temp[0]), 0 - float(temp[1]), float(line[7])
                                elif len(temp) == 3:
                                    pocket_info = line[3], int(line[5]), 0 - float(temp[1]), 0 - float(temp[2]), float(line[7])
                                else:
                                    temp = line[6].split('.')
                                    pocket_info = line[3], int(line[5]), float(temp[0] + '.' + temp[1][:3]), float(temp[1][3:] + '.' + temp[2]), float(line[7])
                            else:
                                if len(temp) == 2:
                                    pocket_info = line[3], int(line[4]), float(temp[0]), 0 - float(temp[1]), float(line[7])
                                elif len(temp) == 3:
                                    pocket_info = line[3], int(line[4]), 0 - float(temp[1]), 0 - float(temp[2]), float(line[7])
                                else:
                                    temp = line[6].split('.')
                                    pocket_info = line[3], int(line[4]), float(temp[0] + '.' + temp[1][:3]), float(temp[1][3:] + '.' + temp[2]), float(line[7])
                        elif len(line[7]) >= 10 and len(line[7]) < 18:
                            temp = line[7].split('-')
                            if line[4].isalpha():
                                if len(temp) == 2:
                                    pocket_info = line[3], int(line[5]), float(line[6]), float(temp[0]), 0 - float(temp[1])
                                elif len(temp) == 3:
                                    pocket_info = line[3], int(line[5]), float(line[6]), 0 - float(temp[1]), 0 - float(temp[2])
                                else:
                                    temp = line[7].split('.')
                                    pocket_info = line[3], int(line[5]), float(line[6]), float(temp[0] + '.' + temp[1][:3]), float(temp[1][3:] + '.' + temp[2])
                            else:
                                if len(temp) == 2:
                                    pocket_info = line[3], int(line[4]), float(line[6]), float(temp[0]), 0 - float(temp[1])
                                elif len(temp) == 3:
                                    pocket_info = line[3], int(line[4]), float(line[6]), 0 - float(temp[1]), 0 - float(temp[2])
                                else:
                                    temp = line[7].split('.')
                                    pocket_info = line[3], int(line[4]), float(line[6]), float(temp[0] + '.' + temp[1][:3]), float(temp[1][3:] + '.' + temp[2])
                        elif len(line[6]) > 18:
                            temp = line[6].split('-')
                            if len(temp) == 3:
                                pocket_info = line[3], int(line[5]), float(temp[0]), 0 - float(temp[1]), 0 - float(temp[2])
                            else:
                                pocket_info = line[3], int(line[5]), 0 - float(temp[0]), 0 - float(temp[1]), 0 - float(temp[2])
                        else:
                            pocket_info = line[3], int(line[5]), float(line[6]), float(line[7]), float(line[8])
                    else:
                        if len(line[5]) >= 10 and len(line[5]) < 18:
                            temp = line[5].split('-')
                            if len(temp) == 2:
                                pocket_info = line[3], int(line[4]), float(temp[0]), 0 - float(temp[1]), float(line[6])
                            elif len(temp) == 3:
                                pocket_info = line[3], int(line[4]), 0 - float(temp[1]), 0 - float(temp[2]), float(line[6])
                            else:
                                temp = line[5].split('.')
                                if len(temp) == 3:
                                    pocket_info = line[3], int(line[4]), float(temp[0] + '.' + temp[1][:3]), float(temp[1][3:] + '.' + temp[2]), float(line[6])
                                else:
                                    pocket_info = line[3], int(line[4]), float(temp[0] + '.' + temp[1][:3]), float(temp[1][3:] + '.' + temp[2][:3]), float(temp[2][3:] + '.' + temp[3])
                        elif len(line[5]) >= 18:
                            temp = line[5].split('-')
                            if len(temp) == 3:
                                pocket_info = line[3], int(line[4]), float(temp[0]), 0 - float(temp[1]), 0 - float(temp[2])
                            else:
                                pocket_info = line[3], int(line[4]), 0 - float(temp[1]), 0 - float(temp[2]), 0 - float(temp[3])
                        elif len(line[6]) >= 10 and len(line[6]) < 18:
                            temp = line[6].split('-')
                            if len(temp) == 2 and line[4].isdigit():
                                pocket_info = line[3], int(line[4]), float(line[5]), float(temp[0]), 0 - float(temp[1])
                            elif len(temp) == 3 and line[4].isdigit():
                                pocket_info = line[3], int(line[4]), float(line[5]), 0 - float(temp[1]), 0 - float(temp[2])
                            elif len(temp) == 2 and line[4].isalpha():
                                pocket_info = line[3], int(line[5]), float(temp[0]), 0 - float(temp[1]), float(line[7])
                            elif len(temp) == 3 and line[4].isalpha():
                                pocket_info = line[3], int(line[5]), 0 - float(temp[1]), 0 - float(temp[2]), float(line[7])
                            else:
                                temp = line[6].split('.')
                                pocket_info = line[3], int(line[4]), float(line[5]), float(temp[0] + '.' + temp[1][:3]), float(temp[1][3:] + '.' + temp[2])
                        elif len(line[6]) >= 18 and len(line[4]) == 1 and line[4].isupper():
                            temp = line[6].split('-')
                            if len(temp) == 3:
                                pocket_info = line[3], int(line[5]), float(temp[0]), 0 - float(temp[1]), 0 - float(temp[2])
                            else:
                                pocket_info = line[3], int(line[5]), 0 - float(temp[0]), 0 - float(temp[1]), 0 - float(temp[2])
                        else:
                            pocket_info = line[3], int(line[4]), float(line[5]), float(line[6]), float(line[7])
                    pocket_info_list.append(list(pocket_info))
    return pocket_info_list

# 获取口袋在蛋白质序列中的下标索引及口袋蛋白residue的坐标
def get_pocket_protein_index_coordinate(pdbid):
    index_list = []
    coordinate_list = []
    pocket_info = get_pocket_position(pdbid)
    global_info = get_global_position(pdbid)
    for i in range(len(pocket_info)):
        for j in range(len(global_info)):
            # print(pocket_info[i])
            if pocket_info[i][0] == global_info[j][0] and pocket_info[i][1] == global_info[j][1] and pocket_info[i][2] == global_info[j][2] and pocket_info[i][3] == global_info[j][3] and pocket_info[i][4] == global_info[j][4]:
                index_list.append(j)
                coordinate = global_info[j][2], global_info[j][3], global_info[j][4]
                coordinate_list.append(list(coordinate))
    if len(index_list) != len(pocket_info):
        print(len(index_list), len(pocket_info))
        print('发现信息丢失的口袋蛋白：{}'.format(pdbid))
        with open('error_pdb', 'a') as write:
            write.writelines(pdbid + '\n')
    return index_list, coordinate_list

# 标签编码提取特征
def compound_label_code(pdbid):
    compound_feature = []
    smile = ''
    with open('data/compound_sdf/' + pdbid + '.sdf', 'r') as read:
        while 1:
            line = read.readline()[:-1]
            if line == '$$$$':
                break
            line = line.split(' ')
            line = list(filter(None, line))
            # print(line)
            if len(line) > 4 and line[3] in atom_list:
                smile += line[3]
    smile = smiles_to_int(smile)
    # print(smile)
    for i in smile:
        feature = [i]
        compound_feature.append(feature)
    return compound_feature

def protein_label_code(pdbid, data='global'):
    if data == 'global':
        Amino_acid_list = []
        p_str = parser.get_structure(pdbid, 'data/protein_pdb/' + pdbid + '.pdb')
        for chain in p_str[0]:
            if chain.get_id() != ' ':
                for residue in chain:
                    if residue.has_id('CA'):
                        if str(residue.get_resname()) in Amino_acids:
                            Amino_acid_list.append(residue.get_resname())
    else:
        with open('data/pocket_pdb/' + pdbid + '.pdb', 'r') as read:
            Amino_acid_list = []
            while 1:
                line = read.readline()[:-1]
                if line == '':
                    break
                if line[:4] == 'ATOM':
                    line = line.split(' ')
                    line = list(filter(None, line))
                    if 'CA' in line:
                        Amino_acid = line[3]
                        if len(Amino_acid) != 3:
                            Amino_acid = Amino_acid[1:]
                        Amino_acid_list.append(Amino_acid)
    protein_feature = []
    for i in Amino_acid_list:
        num = Amino_acids_dict[i]
        feature = [num]
        protein_feature.append(feature)
    return protein_feature

# 提取特征
# 方式一：one-hot编码
def compoundOneHotProperties(pdbid):
    compound_feature = []
    smile = ''
    with open('data/compound_sdf/' + pdbid + '.sdf', 'r') as read:
        while 1:
            line = read.readline()[:-1]
            if line == '$$$$':
                break
            line = line.split(' ')
            line = list(filter(None, line))
            if len(line) > 4 and line[3] in atom_list:
                smile += line[3]
    smile_int = smiles_to_int(smile)
    for i in range(len(smile_int)):
        feature = [0 for j in range(len(atom_list))]
        feature[smile_int[i]] = 1
        compound_feature.append(feature)
    return compound_feature

def proteinOneHotProperties(pdbid, data='global'):
    if data == 'global':
        Amino_acid_list = []
        p_str = parser.get_structure(pdbid, 'data/protein_pdb/' + pdbid + '.pdb')
        for chain in p_str[0]:
            if chain.get_id() != ' ':
                for residue in chain:
                    if residue.has_id('CA'):
                        if str(residue.get_resname()) in Amino_acids:
                            Amino_acid_list.append(residue.get_resname())
    else:
        with open('data/pocket_pdb/' + pdbid + '.pdb', 'r') as read:
            Amino_acid_list = []
            while 1:
                line = read.readline()[:-1]
                if line == '':
                    break
                if line[:4] == 'ATOM':
                    line = line.split(' ')
                    line = list(filter(None, line))
                    if 'CA' in line:
                        Amino_acid = line[3]
                        if len(Amino_acid) != 3:
                            Amino_acid = Amino_acid[1:]
                        Amino_acid_list.append(Amino_acid)
    protein_feature = []
    for i in Amino_acid_list:
        num = Amino_acids_dict[i]
        feature = [0 for i in range(len(Amino_acids))]
        feature[num] = 1
        protein_feature.append(feature)
    return protein_feature

# 方式二：蛋白质和小分子的理化性质
def compoundPhyAndCheProperties(pdbid):
    with open('data/atom_properties.csv', 'r') as read:
        read.readline()
        feature_list = []
        # print(read.readline())
        while 1:
            line = read.readline()[:-1]
            if line == '':
                break
            line = line.split(';')
            for i in range(len(line)):
                if line[i][0] == "'":
                    line[i] = line[i][1:-1]
            if line[1] in atom_list:
                feature_list.append(line)
    compound_feature = []
    smile = []
    with open('data/compound_sdf/' + pdbid + '.sdf', 'r') as read:
        while 1:
            line = read.readline()[:-1]
            if line == '$$$$':
                break
            line = line.split(' ')
            line = list(filter(None, line))
            # print(line)
            if len(line) > 4 and line[3] in atom_list:
                smile.append(line[3])
    error_list = []
    feature5 = 0
    feature8 = 0
    for feature in feature_list:
        if feature[5] != '-':
            feature5 = feature5 + float(feature[5])
        if feature[8] != '-':
            feature8 = feature8 + float(feature[8])
    error_list.append(feature5 / (len(feature_list) - 4))
    error_list.append(feature8 / (len(feature_list) - 6))
    # print(error_list)
    for atom in smile:
        for feature in feature_list:
            if atom in feature:
                if feature[5] == '-':
                    feature[5] = error_list[0]
                if feature[8] == '-':
                    feature[8] = error_list[1]
                feature = float(feature[2]), float(feature[3]), float(feature[4]), float(feature[5]), float(feature[6]), float(feature[7]), float(feature[8]), float(feature[11]), float(atom_outest_ele[atom_list.index(atom)])
                feature = list(feature)
                compound_feature.append(feature)
    return compound_feature

def proteinPhyAndCheProperties(pdbid, data='global'):
    feature_list = []
    with open('data/proteinPC.csv', 'r') as read:
        line = read.readline()
        while 1:
            line = read.readline()[:-1]
            if line == '':
                break
            line = line.split(' ')
            feature_list.append(line)
        for i in range(len(feature_list)):
            for j in range(1, len(feature_list[i])):
                if feature_list[i][j] == '-':
                    feature_list[i][j] = 0.0
                else:
                    feature_list[i][j] = float(feature_list[i][j])
    if data == 'global':
        Amino_acid_list = []
        p_str = parser.get_structure(pdbid, 'data/protein_pdb/' + pdbid + '.pdb')
        for chain in p_str[0]:
            if chain.get_id() != ' ':
                for residue in chain:
                    if residue.has_id('CA'):
                        if str(residue.get_resname()) in Amino_acids:
                            Amino_acid_list.append(residue.get_resname())
    else:
        with open('data/pocket_pdb/' + pdbid + '.pdb', 'r') as read:
            Amino_acid_list = []
            while 1:
                line = read.readline()[:-1]
                if line == '':
                    break
                if line[:4] == 'ATOM':
                    line = line.split(' ')
                    line = list(filter(None, line))
                    if 'CA' in line:
                        Amino_acid = line[3]
                        if len(Amino_acid) != 3:
                            Amino_acid = Amino_acid[1:]
                        Amino_acid_list.append(Amino_acid)
    protein_feature = []
    for i in Amino_acid_list:
        num = Amino_acids_dict[i]
        protein_feature.append(feature_list[num][1:])
    return protein_feature

# 蛋白质位置特异性打分矩阵
def proteinPSSM(pdbid):
    feature_list = []
    with open('data/protein_pssm/' + pdbid + '.pssm', 'r') as read:
        lines = read.readlines()
        for pssm in lines[3:-6]:
            pssm = list(filter(None, pssm.split(' ')))[1:22]
            feature_list.append([int(s) for s in pssm[1:]])
    return feature_list

# 通过隐马尔科夫模型获取蛋白质的特征表征
def proteinHMM(pdbid):
    feature_list = []
    with open('data/protein_hhm/' + pdbid + '.hhm', 'r') as read:
        lines = read.readlines()
        for i in range(len(lines)):
            line = list(filter(None, lines[i][:-1].split('\t')))
            if len(line) > 20 and line[0].split(' ')[0].isupper():
                # line = line[:-1] + list(filter(None, lines[i + 1][:-1].split('\t')))
                line = line[:-1]
                line[0] = list(filter(None, line[0].split(' ')))[2]
                for i in range(len(line)):
                    if line[i] == '*':
                        line[i] = 9999
                    else:
                        line[i] = int(line[i])
                feature_list.append(line)
    return feature_list

def get_pocket_feature(pdbid, way='onehot'):
    pssm_list = []
    hmm_list = []
    feature_list = []
    index_list, _ = get_pocket_protein_index_coordinate(pdbid)
    if way == 'label':
        feature_list = protein_label_code(pdbid, data='pocket')
    elif way == 'onehot':
        feature_list = proteinOneHotProperties(pdbid, data='pocket')
    elif way == 'pccs':
        feature_list = proteinPhyAndCheProperties(pdbid, data='pocket')
    elif way == 'pssm':
        pssm_list = proteinPSSM(pdbid)
        for index in index_list:
            feature_list.append(pssm_list[index])
    elif way == 'hmm':
        hmm_list = proteinHMM(pdbid)
        for index in index_list:
            feature_list.append(hmm_list[index])
    return feature_list

def getCompoundGraph(pdbid):
    with open('data/compound_sdf/' + pdbid + '.sdf', 'r') as read:
        read.readline()
        read.readline()
        read.readline()
        atom_num = 0
        edge_list = []
        while 1:
            line = read.readline()[:-1]
            if line == '$$$$':
                break
            line = line.split(' ')
            line = list(filter(None, line))
            if len(line) > 4 and line[3] in atom_list:
                atom_num += 1
            if len(line) == 6:
                edge_list.append([int(line[0]) - 1, int(line[1]) - 1])
    features = compoundOneHotProperties(pdbid)
    return atom_num, features, edge_list

# 蛋白质转为图结构的处理函数
def getSkeletonAtoms(pdbid, data='train'):
    print(pdbid)
    atom_num = 0
    coordinate_list = []
    p_str = parser.get_structure(pdbid, 'data/protein_pdb/' + pdbid + '.pdb')
    for chain in p_str[0]:
        if chain.get_id() != ' ':
            for residue in chain:
                if residue.has_id('CA'):
                    if str(residue.get_resname()) in Amino_acids:
                        atom_num += 1
                        coordinate = round(float(residue['CA'].get_coord()[0]), 3), round(float(residue['CA'].get_coord()[1]), 3), round(float(residue['CA'].get_coord()[2]), 3)
                        coordinate_list.append(list(coordinate))
    features = proteinOneHotProperties(pdbid)
    # features = proteinHMM(pdbid)
    # features = proteinPSSM(pdbid)
    # features = proteinPhyAndCheProperties(pdbid)
    # features = protein_label_code(pdbid)
    return atom_num, features, coordinate_list

# 生成图的边集
def getProteinGraph(data):
    edge_list = []
    for i in range(len(data[2])):
        for j in range(len(data[2])):
                if i == j:
                    pass
                else:
                    dis = math.sqrt((data[2][i][0] - data[2][j][0]) ** 2 + (data[2][i][1] - data[2][j][1]) ** 2 + (data[2][i][2] - data[2][j][2]) ** 2)
                    if dis <= 8:
                        edge = [i, j]
                        edge_list.append(edge)
    return data[0], data[1], edge_list

def getPocketGraph(pdbid):
    edge_list = []
    _, coordinate_list = get_pocket_protein_index_coordinate(pdbid)
    for i in range(len(coordinate_list)):
        for j in range(len(coordinate_list)):
                if i == j:
                    pass
                else:
                    dis = math.sqrt((coordinate_list[i][0] - coordinate_list[j][0]) ** 2 + (coordinate_list[i][1] - coordinate_list[j][1]) ** 2 + (coordinate_list[i][2] - coordinate_list[j][2]) ** 2)
                    if dis <= 8:
                        edge = [i, j]
                        edge_list.append(edge)
    feature_list = get_pocket_feature(pdbid, way='fusion')
    return len(coordinate_list), feature_list, edge_list
