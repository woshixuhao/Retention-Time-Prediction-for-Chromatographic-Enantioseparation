import pandas as pd
from tqdm import tqdm
import warnings
import torch
import numpy as np
from compound_tools import *
warnings.filterwarnings('ignore')
from rdkit.Chem.Descriptors import rdMolDescriptors
import os
from mordred import Calculator, descriptors,is_missing
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from sklearn.cluster import KMeans

def morgan(mol, r=3, nBits=8128, errors_as_zeros=True):
    try:
        arr = np.zeros((1,))
        ConvertToNumpyArray(GetMorganFingerprintAsBitVect(mol, r, nBits), arr)
        return arr.astype(np.float32)
    except:
        return np.NaN if not errors_as_zeros else np.zeros((nBits,), dtype=np.float32)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device=torch.device('cuda', 1)
calc = Calculator(descriptors, ignore_3D=False)


def mord(mol, nBits=1826, errors_as_zeros=True):
    try:
        result = calc(mol)
        desc_list = [r if not is_missing(r) else 0 for r in result]
        np_arr = np.array(desc_list)
        return np_arr
    except:
        return np.NaN if not errors_as_zeros else np.zeros((nBits,), dtype=np.float32)

def save_3D_mol(all_smile,mol_save_dir):
    index=0
    bad_conformer=[]
    pbar=tqdm(all_smile)
    try:
        os.makedirs(f'{mol_save_dir}')
    except OSError:
        pass
    for smiles in pbar:
        try:
            obtain_3D_mol(smiles,f'{mol_save_dir}/3D_mol_{index}')
        except ValueError:
            bad_conformer.append(index)
            index += 1
            continue
        index += 1
    return bad_conformer

def save_dataset(charity_smile,mol_save_dir,charity_name,moder_name,bad_conformer):
    dataset=[]
    dataset_mord=[]
    pbar = tqdm(charity_smile)
    index=0
    for smile in pbar:
        if index in bad_conformer:
            index+=1
            continue
        mol=Chem.MolFromMolFile(f"{mol_save_dir}/3D_mol_{index}.mol")
        #mol = AllChem.MolFromSmiles(smile)
        descriptor=mord(mol)
        data = mol_to_geognn_graph_data_MMFF3d(mol)
        dataset.append(data)
        dataset_mord.append(descriptor)
        index+=1

    dataset_mord=np.array(dataset_mord)
    np.save(f"{charity_name}.npy",dataset,allow_pickle=True)
    np.save(f'{moder_name}.npy',dataset_mord)

dataset_all=pd.read_csv('All_column_charity.csv')
dataset_IA=dataset_all[dataset_all['Column']=='IA']
dataset_AD=dataset_all[dataset_all['Column']=='AD']
dataset_ADH=dataset_all[dataset_all['Column']=='ADH']
dataset_IC=dataset_all[dataset_all['Column']=='IC']
dataset_IA.to_csv("IA_charity_0823.csv")
dataset_AD.to_csv("AD_charity_0823.csv")
dataset_ADH.to_csv("ADH_charity_0823.csv")
dataset_IC.to_csv("IC_charity_0823.csv")
all_smile_IA=dataset_IA['SMILES'].values
all_smile_AD=dataset_AD['SMILES'].values
all_smile_ADH=dataset_ADH['SMILES'].values
all_smile_IC=dataset_IC['SMILES'].values
bad_IA=save_3D_mol(all_smile_IA,'IA_3D_mol')
bad_AD=save_3D_mol(all_smile_AD,'AD_3D_mol')
bad_ADH=save_3D_mol(all_smile_ADH,'ADH_3D_mol')
bad_IC=save_3D_mol(all_smile_IC,'IC_3D_mol')
# print('bad_IA:',bad_IA)
# print('bad_AD:',bad_AD)
# print('bad_ADH:',bad_ADH)
# print('bad_IC:',bad_IC)
np.save('bad_IA.npy',np.array(bad_IA))
np.save('bad_AD.npy',np.array(bad_AD))
np.save('bad_ADH.npy',np.array(bad_ADH))
np.save('bad_IC.npy',np.array(bad_IC))
save_dataset(all_smile_IA,'IA_3D_mol','dataset_IA_charity_0823','dataset_IA_charity_morder_0823',bad_IA)
save_dataset(all_smile_AD,'AD_3D_mol','dataset_AD_charity_0823','dataset_AD_charity_morder_0823',bad_AD)
save_dataset(all_smile_ADH,'ADH_3D_mol','dataset_ADH_charity_0823','dataset_ADH_charity_morder_0823',bad_ADH)
save_dataset(all_smile_IC,'IC_3D_mol','dataset_IC_charity_0823','dataset_IC_charity_morder_0823',bad_IC)