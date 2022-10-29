import torch
from torch_geometric.nn import MessagePassing
from compound_tools import *
from rdkit.Chem import Descriptors
from torch_geometric.data import Data
import argparse
import warnings
import matplotlib.pyplot as plt
from rdkit.Chem.Descriptors import rdMolDescriptors
import os
from mordred import Calculator, descriptors, is_missing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from tkinter import Tk
from tkinter.simpledialog import Dialog
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem

DAY_LIGHT_FG_SMARTS_LIST = [
        # C
        "[CX4]",
        "[$([CX2](=C)=C)]",
        "[$([CX3]=[CX3])]",
        "[$([CX2]#C)]",
        # C & O
        "[CX3]=[OX1]",
        "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]",
        "[CX3](=[OX1])C",
        "[OX1]=CN",
        "[CX3](=[OX1])O",
        "[CX3](=[OX1])[F,Cl,Br,I]",
        "[CX3H1](=O)[#6]",
        "[CX3](=[OX1])[OX2][CX3](=[OX1])",
        "[NX3][CX3](=[OX1])[#6]",
        "[NX3][CX3]=[NX3+]",
        "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",
        "[NX3][CX3](=[OX1])[OX2H0]",
        "[NX3,NX4+][CX3](=[OX1])[OX2H,OX1-]",
        "[CX3](=O)[O-]",
        "[CX3](=[OX1])(O)O",
        "[CX3](=[OX1])([OX2])[OX2H,OX1H0-1]",
        "C[OX2][CX3](=[OX1])[OX2]C",
        "[CX3](=O)[OX2H1]",
        "[CX3](=O)[OX1H0-,OX2H1]",
        "[NX3][CX2]#[NX1]",
        "[#6][CX3](=O)[OX2H0][#6]",
        "[#6][CX3](=O)[#6]",
        "[OD2]([#6])[#6]",
        # H
        "[H]",
        "[!#1]",
        "[H+]",
        "[+H]",
        "[!H]",
        # N
        "[NX3;H2,H1;!$(NC=O)]",
        "[NX3][CX3]=[CX3]",
        "[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]",
        "[NX3;H2,H1;!$(NC=O)].[NX3;H2,H1;!$(NC=O)]",
        "[NX3][$(C=C),$(cc)]",
        "[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[O,N]",
        "[NX3H2,NH3X4+][CX4H]([*])[CX3](=[OX1])[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-]",
        "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-,N]",
        "[CH3X4]",
        "[CH2X4][CH2X4][CH2X4][NHX3][CH0X3](=[NH2X3+,NHX2+0])[NH2X3]",
        "[CH2X4][CX3](=[OX1])[NX3H2]",
        "[CH2X4][CX3](=[OX1])[OH0-,OH]",
        "[CH2X4][SX2H,SX1H0-]",
        "[CH2X4][CH2X4][CX3](=[OX1])[OH0-,OH]",
        "[$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H2][CX3](=[OX1])[OX2H,OX1-,N])]",
        "[CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:\
[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1",
        "[CHX4]([CH3X4])[CH2X4][CH3X4]",
        "[CH2X4][CHX4]([CH3X4])[CH3X4]",
        "[CH2X4][CH2X4][CH2X4][CH2X4][NX4+,NX3+0]",
        "[CH2X4][CH2X4][SX2][CH3X4]",
        "[CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1",
        "[$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[OX2H,OX1-,N]",
        "[CH2X4][OX2H]",
        "[NX3][CX3]=[SX1]",
        "[CHX4]([CH3X4])[OX2H]",
        "[CH2X4][cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12",
        "[CH2X4][cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1",
        "[CHX4]([CH3X4])[CH3X4]",
        "N[CX4H2][CX3](=[OX1])[O,N]",
        "N1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[O,N]",
        "[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]",
        "[$([NX1-]=[NX2+]=[NX1-]),$([NX1]#[NX2+]-[NX1-2])]",
        "[#7]",
        "[NX2]=N",
        "[NX2]=[NX2]",
        "[$([NX2]=[NX3+]([O-])[#6]),$([NX2]=[NX3+0](=[O])[#6])]",
        "[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]",
        "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]",
        "[NX3][NX3]",
        "[NX3][NX2]=[*]",
        "[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]",
        "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
        "[NX3+]=[CX3]",
        "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
        "[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])",
        "[CX3](=[OX1])[NX3H0]([NX3H0]([CX3](=[OX1]))[CX3](=[OX1]))[CX3](=[OX1])",
        "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",
        "[$([OX1]=[NX3](=[OX1])[OX1-]),$([OX1]=[NX3+]([OX1-])[OX1-])]",
        "[NX1]#[CX2]",
        "[CX1-]#[NX2+]",
        "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8].[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "[NX2]=[OX1]",
        "[$([#7+][OX1-]),$([#7v5]=[OX1]);!$([#7](~[O])~[O]);!$([#7]=[#7])]",
        # O
        "[OX2H]",
        "[#6][OX2H]",
        "[OX2H][CX3]=[OX1]",
        "[OX2H]P",
        "[OX2H][#6X3]=[#6]",
        "[OX2H][cX3]:[c]",
        "[OX2H][$(C=C),$(cc)]",
        "[$([OH]-*=[!#6])]",
        "[OX2,OX1-][OX2,OX1-]",
        # P
        "[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),\
$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-])\
,$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]",
        "[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),\
$([OX1-]),$([OX2][#6]),$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),\
$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]",
        # S
        "[S-][CX3](=S)[#6]",
        "[#6X3](=[SX1])([!N])[!N]",
        "[SX2]",
        "[#16X2H]",
        "[#16!H0]",
        "[#16X2H0]",
        "[#16X2H0][!#16]",
        "[#16X2H0][#16X2H0]",
        "[#16X2H0][!#16].[#16X2H0][!#16]",
        "[$([#16X3](=[OX1])[OX2H0]),$([#16X3+]([OX1-])[OX2H0])]",
        "[$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]",
        "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]",
        "[SX4](C)(C)(=O)=N",
        "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]",
        "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
        "[$([#16X3](=[OX1])([#6])[#6]),$([#16X3+]([OX1-])([#6])[#6])]",
        "[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]),$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6])]",
        "[$([SX4](=O)(=O)(O)O),$([SX4+2]([O-])([O-])(O)O)]",
        "[$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6]),$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2][#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2][#6])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2H,OX1H0-]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2H,OX1H0-])]",
        "[#16X2][OX2H,OX1H0-]",
        "[#16X2][OX2H0]",
        # X
        "[#6][F,Cl,Br,I]",
        "[F,Cl,Br,I]",
        "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]",
    ]


def get_gasteiger_partial_charges(mol, n_iter=12):
    """
    Calculates list of gasteiger partial charges for each atom in mol object.
    Args:
        mol: rdkit mol object.
        n_iter(int): number of iterations. Default 12.
    Returns:
        list of computed partial charges for each atom.
    """
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, nIter=n_iter,
                                                  throwOnParamFailure=True)
    partial_charges = [float(a.GetProp('_GasteigerCharge')) for a in
                       mol.GetAtoms()]
    return partial_charges


def create_standardized_mol_id(smiles):
    """
    Args:
        smiles: smiles sequence.
    Returns:
        inchi.
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if not mol is None:  # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles:  # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return


def check_smiles_validity(smiles):
    """
    Check whether the smile can't be converted to rdkit mol object.
    """
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except Exception as e:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively.
    Args:
        mol: rdkit mol object.
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one.
    Args:
        mol_list(list): a list of rdkit mol object.
    Returns:
        the largest mol.
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]


def rdchem_enum_to_list(values):
    """values = {0: rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            1: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            2: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            3: rdkit.Chem.rdchem.ChiralType.CHI_OTHER}
    """
    return [values[i] for i in range(len(values))]


def safe_index(alist, elem):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return alist.index(elem)
    except ValueError:
        return len(alist) - 1


def get_atom_feature_dims(list_acquired_feature_names):
    """ tbd
    """
    return list(map(len, [CompoundKit.atom_vocab_dict[name] for name in list_acquired_feature_names]))


def get_bond_feature_dims(list_acquired_feature_names):
    """ tbd
    """
    list_bond_feat_dim = list(map(len, [CompoundKit.bond_vocab_dict[name] for name in list_acquired_feature_names]))
    # +1 for self loop edges
    return [_l + 1 for _l in list_bond_feat_dim]


class CompoundKit(object):
    """
    CompoundKit
    """
    atom_vocab_dict = {
        "atomic_num": list(range(1, 119)) + ['misc'],
        "chiral_tag": rdchem_enum_to_list(rdchem.ChiralType.values),
        "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "explicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "hybridization": rdchem_enum_to_list(rdchem.HybridizationType.values),
        "implicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "is_aromatic": [0, 1],
        "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'num_radical_e': [0, 1, 2, 3, 4, 'misc'],
        'atom_is_in_ring': [0, 1],
        'valence_out_shell': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size5': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size6': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size7': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size8': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    }
    bond_vocab_dict = {
        "bond_dir": rdchem_enum_to_list(rdchem.BondDir.values),
        "bond_type": rdchem_enum_to_list(rdchem.BondType.values),
        "is_in_ring": [0, 1],

        'bond_stereo': rdchem_enum_to_list(rdchem.BondStereo.values),
        'is_conjugated': [0, 1],
    }
    # float features
    atom_float_names = ["van_der_waals_radis", "partial_charge", 'mass']
    # bond_float_feats= ["bond_length", "bond_angle"]     # optional

    ### functional groups
    day_light_fg_smarts_list = DAY_LIGHT_FG_SMARTS_LIST
    day_light_fg_mo_list = [Chem.MolFromSmarts(smarts) for smarts in day_light_fg_smarts_list]

    morgan_fp_N = 200
    morgan2048_fp_N = 2048
    maccs_fp_N = 167

    period_table = Chem.GetPeriodicTable()

    ### atom

    @staticmethod
    def get_atom_value(atom, name):
        """get atom values"""
        if name == 'atomic_num':
            return atom.GetAtomicNum()
        elif name == 'chiral_tag':
            return atom.GetChiralTag()
        elif name == 'degree':
            return atom.GetDegree()
        elif name == 'explicit_valence':
            return atom.GetExplicitValence()
        elif name == 'formal_charge':
            return atom.GetFormalCharge()
        elif name == 'hybridization':
            return atom.GetHybridization()
        elif name == 'implicit_valence':
            return atom.GetImplicitValence()
        elif name == 'is_aromatic':
            return int(atom.GetIsAromatic())
        elif name == 'mass':
            return int(atom.GetMass())
        elif name == 'total_numHs':
            return atom.GetTotalNumHs()
        elif name == 'num_radical_e':
            return atom.GetNumRadicalElectrons()
        elif name == 'atom_is_in_ring':
            return int(atom.IsInRing())
        elif name == 'valence_out_shell':
            return CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())
        else:
            raise ValueError(name)

    @staticmethod
    def get_atom_feature_id(atom, name):
        """get atom features id"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return safe_index(CompoundKit.atom_vocab_dict[name], CompoundKit.get_atom_value(atom, name))

    @staticmethod
    def get_atom_feature_size(name):
        """get atom features size"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return len(CompoundKit.atom_vocab_dict[name])

    ### bond

    @staticmethod
    def get_bond_value(bond, name):
        """get bond values"""
        if name == 'bond_dir':
            return bond.GetBondDir()
        elif name == 'bond_type':
            return bond.GetBondType()
        elif name == 'is_in_ring':
            return int(bond.IsInRing())
        elif name == 'is_conjugated':
            return int(bond.GetIsConjugated())
        elif name == 'bond_stereo':
            return bond.GetStereo()
        else:
            raise ValueError(name)

    @staticmethod
    def get_bond_feature_id(bond, name):
        """get bond features id"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return safe_index(CompoundKit.bond_vocab_dict[name], CompoundKit.get_bond_value(bond, name))

    @staticmethod
    def get_bond_feature_size(name):
        """get bond features size"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return len(CompoundKit.bond_vocab_dict[name])

    ### fingerprint

    @staticmethod
    def get_morgan_fingerprint(mol, radius=2):
        """get morgan fingerprint"""
        nBits = CompoundKit.morgan_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_morgan2048_fingerprint(mol, radius=2):
        """get morgan2048 fingerprint"""
        nBits = CompoundKit.morgan2048_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_maccs_fingerprint(mol):
        """get maccs fingerprint"""
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]

    ### functional groups

    @staticmethod
    def get_daylight_functional_group_counts(mol):
        """get daylight functional group counts"""
        fg_counts = []
        for fg_mol in CompoundKit.day_light_fg_mo_list:
            sub_structs = Chem.Mol.GetSubstructMatches(mol, fg_mol, uniquify=True)
            fg_counts.append(len(sub_structs))
        return fg_counts

    @staticmethod
    def get_ring_size(mol):
        """return (N,6) list"""
        rings = mol.GetRingInfo()
        rings_info = []
        for r in rings.AtomRings():
            rings_info.append(r)
        ring_list = []
        for atom in mol.GetAtoms():
            atom_result = []
            for ringsize in range(3, 9):
                num_of_ring_at_ringsize = 0
                for r in rings_info:
                    if len(r) == ringsize and atom.GetIdx() in r:
                        num_of_ring_at_ringsize += 1
                if num_of_ring_at_ringsize > 8:
                    num_of_ring_at_ringsize = 9
                atom_result.append(num_of_ring_at_ringsize)

            ring_list.append(atom_result)
        return ring_list

    @staticmethod
    def atom_to_feat_vector(atom):
        """ tbd """
        atom_names = {
            "atomic_num": safe_index(CompoundKit.atom_vocab_dict["atomic_num"], atom.GetAtomicNum()),
            "chiral_tag": safe_index(CompoundKit.atom_vocab_dict["chiral_tag"], atom.GetChiralTag()),
            "degree": safe_index(CompoundKit.atom_vocab_dict["degree"], atom.GetTotalDegree()),
            "explicit_valence": safe_index(CompoundKit.atom_vocab_dict["explicit_valence"], atom.GetExplicitValence()),
            "formal_charge": safe_index(CompoundKit.atom_vocab_dict["formal_charge"], atom.GetFormalCharge()),
            "hybridization": safe_index(CompoundKit.atom_vocab_dict["hybridization"], atom.GetHybridization()),
            "implicit_valence": safe_index(CompoundKit.atom_vocab_dict["implicit_valence"], atom.GetImplicitValence()),
            "is_aromatic": safe_index(CompoundKit.atom_vocab_dict["is_aromatic"], int(atom.GetIsAromatic())),
            "total_numHs": safe_index(CompoundKit.atom_vocab_dict["total_numHs"], atom.GetTotalNumHs()),
            'num_radical_e': safe_index(CompoundKit.atom_vocab_dict['num_radical_e'], atom.GetNumRadicalElectrons()),
            'atom_is_in_ring': safe_index(CompoundKit.atom_vocab_dict['atom_is_in_ring'], int(atom.IsInRing())),
            'valence_out_shell': safe_index(CompoundKit.atom_vocab_dict['valence_out_shell'],
                                            CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())),
            'van_der_waals_radis': CompoundKit.period_table.GetRvdw(atom.GetAtomicNum()),
            'partial_charge': CompoundKit.check_partial_charge(atom),
            'mass': atom.GetMass(),
        }
        return atom_names

    @staticmethod
    def get_atom_names(mol):
        """get atom name list
        TODO: to be remove in the future
        """
        atom_features_dicts = []
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts.append(CompoundKit.atom_to_feat_vector(atom))

        ring_list = CompoundKit.get_ring_size(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts[i]['in_num_ring_with_size3'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size3'], ring_list[i][0])
            atom_features_dicts[i]['in_num_ring_with_size4'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size4'], ring_list[i][1])
            atom_features_dicts[i]['in_num_ring_with_size5'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size5'], ring_list[i][2])
            atom_features_dicts[i]['in_num_ring_with_size6'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size6'], ring_list[i][3])
            atom_features_dicts[i]['in_num_ring_with_size7'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size7'], ring_list[i][4])
            atom_features_dicts[i]['in_num_ring_with_size8'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size8'], ring_list[i][5])

        return atom_features_dicts

    @staticmethod
    def check_partial_charge(atom):
        """tbd"""
        pc = atom.GetDoubleProp('_GasteigerCharge')
        if pc != pc:
            # unsupported atom, replace nan with 0
            pc = 0
        if pc == float('inf'):
            # max 4 for other atoms, set to 10 here if inf is get
            pc = 10
        return pc


class Compound3DKit(object):
    """the 3Dkit of Compound"""

    @staticmethod
    def get_atom_poses(mol, conf):
        """tbd"""
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses

    @staticmethod
    def get_MMFF_atom_poses(mol, numConfs=None, return_energy=False):
        """the atoms of mol will be changed in some cases."""
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return mol,atom_poses
        # try:
        #     new_mol = Chem.AddHs(mol)
        #     res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
        #     ### MMFF generates multiple conformations
        #     res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        #     new_mol = Chem.RemoveHs(new_mol)
        #     index = np.argmin([x[1] for x in res])
        #     energy = res[index][1]
        #     conf = new_mol.GetConformer(id=int(index))
        # except:
        #     new_mol = mol
        #     AllChem.Compute2DCoords(new_mol)
        #     energy = 0
        #     conf = new_mol.GetConformer()
        #
        # atom_poses = Compound3DKit.get_atom_poses(new_mol, conf)
        # if return_energy:
        #     return new_mol, atom_poses, energy
        # else:
        #     return new_mol, atom_poses

    @staticmethod
    def get_2d_atom_poses(mol):
        """get 2d atom poses"""
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return atom_poses

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        """get bond lengths"""
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))
        bond_lengths = np.array(bond_lengths, 'float32')
        return bond_lengths

    @staticmethod
    def get_superedge_angles(edges, atom_poses, dir_type='HT'):
        """get superedge angles"""

        def _get_vec(atom_poses, edge):
            return atom_poses[edge[1]] - atom_poses[edge[0]]

        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)  # 1e-5: prevent numerical errors
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle

        E = len(edges)
        edge_indices = np.arange(E)
        super_edges = []
        bond_angles = []
        bond_angle_dirs = []
        for tar_edge_i in range(E):
            tar_edge = edges[tar_edge_i]
            if dir_type == 'HT':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]
            elif dir_type == 'HH':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[1]]
            else:
                raise ValueError(dir_type)
            for src_edge_i in src_edge_indices:
                if src_edge_i == tar_edge_i:
                    continue
                src_edge = edges[src_edge_i]
                src_vec = _get_vec(atom_poses, src_edge)
                tar_vec = _get_vec(atom_poses, tar_edge)
                super_edges.append([src_edge_i, tar_edge_i])
                angle = _get_angle(src_vec, tar_vec)
                bond_angles.append(angle)
                bond_angle_dirs.append(src_edge[1] == tar_edge[0])  # H -> H or H -> T

        if len(super_edges) == 0:
            super_edges = np.zeros([0, 2], 'int64')
            bond_angles = np.zeros([0, ], 'float32')
        else:
            super_edges = np.array(super_edges, 'int64')
            bond_angles = np.array(bond_angles, 'float32')
        return super_edges, bond_angles, bond_angle_dirs


def new_smiles_to_graph_data(smiles, **kwargs):
    """
    Convert smiles to graph data.
    """
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    data = new_mol_to_graph_data(mol)
    return data


def new_mol_to_graph_data(mol):
    """
    mol_to_graph_data
    Args:
        atom_features: Atom features.
        edge_features: Edge features.
        morgan_fingerprint: Morgan fingerprint.
        functional_groups: Functional groups.
    """
    if len(mol.GetAtoms()) == 0:
        return None

    atom_id_names = list(CompoundKit.atom_vocab_dict.keys()) + CompoundKit.atom_float_names
    bond_id_names = list(CompoundKit.bond_vocab_dict.keys())

    data = {}

    ### atom features
    data = {name: [] for name in atom_id_names}

    raw_atom_feat_dicts = CompoundKit.get_atom_names(mol)
    for atom_feat in raw_atom_feat_dicts:
        for name in atom_id_names:
            data[name].append(atom_feat[name])

    ### bond and bond features
    for name in bond_id_names:
        data[name] = []
    data['edges'] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j and j->i
        data['edges'] += [(i, j), (j, i)]
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name)
            data[name] += [bond_feature_id] * 2

    #### self loop
    N = len(data[atom_id_names[0]])
    for i in range(N):
        data['edges'] += [(i, i)]
    for name in bond_id_names:
        bond_feature_id = get_bond_feature_dims([name])[0] - 1  # self loop: value = len - 1
        data[name] += [bond_feature_id] * N

    ### make ndarray and check length
    for name in list(CompoundKit.atom_vocab_dict.keys()):
        data[name] = np.array(data[name], 'int64')
    for name in CompoundKit.atom_float_names:
        data[name] = np.array(data[name], 'float32')
    for name in bond_id_names:
        data[name] = np.array(data[name], 'int64')
    data['edges'] = np.array(data['edges'], 'int64')

    ### morgan fingerprint
    data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), 'int64')
    # data['morgan2048_fp'] = np.array(CompoundKit.get_morgan2048_fingerprint(mol), 'int64')
    data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), 'int64')
    data['daylight_fg_counts'] = np.array(CompoundKit.get_daylight_functional_group_counts(mol), 'int64')
    return data


def mol_to_graph_data(mol):
    """
    mol_to_graph_data
    Args:
        atom_features: Atom features.
        edge_features: Edge features.
        morgan_fingerprint: Morgan fingerprint.
        functional_groups: Functional groups.
    """
    if len(mol.GetAtoms()) == 0:
        return None

    atom_id_names = [
        "atomic_num", "chiral_tag", "degree", "explicit_valence",
        "formal_charge", "hybridization", "implicit_valence",
        "is_aromatic", "total_numHs",
    ]
    bond_id_names = [
        "bond_dir", "bond_type", "is_in_ring",
    ]

    data = {}
    for name in atom_id_names:
        data[name] = []
    data['mass'] = []
    for name in bond_id_names:
        data[name] = []
    data['edges'] = []

    ### atom features
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return None
        for name in atom_id_names:
            data[name].append(CompoundKit.get_atom_feature_id(atom, name) + 1)  # 0: OOV
        data['mass'].append(CompoundKit.get_atom_value(atom, 'mass') * 0.01)

    ### bond features
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j and j->i
        data['edges'] += [(i, j), (j, i)]
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name) + 1  # 0: OOV
            data[name] += [bond_feature_id] * 2

    ### self loop (+2)
    N = len(data[atom_id_names[0]])
    for i in range(N):
        data['edges'] += [(i, i)]
    for name in bond_id_names:
        bond_feature_id = CompoundKit.get_bond_feature_size(name) + 2  # N + 2: self loop
        data[name] += [bond_feature_id] * N

    ### check whether edge exists
    if len(data['edges']) == 0:  # mol has no bonds
        for name in bond_id_names:
            data[name] = np.zeros((0,), dtype="int64")
        data['edges'] = np.zeros((0, 2), dtype="int64")

    ### make ndarray and check length
    for name in atom_id_names:
        data[name] = np.array(data[name], 'int64')
    data['mass'] = np.array(data['mass'], 'float32')
    for name in bond_id_names:
        data[name] = np.array(data[name], 'int64')
    data['edges'] = np.array(data['edges'], 'int64')

    ### morgan fingerprint
    data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), 'int64')
    # data['morgan2048_fp'] = np.array(CompoundKit.get_morgan2048_fingerprint(mol), 'int64')
    data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), 'int64')
    data['daylight_fg_counts'] = np.array(CompoundKit.get_daylight_functional_group_counts(mol), 'int64')
    return data


def mol_to_geognn_graph_data(mol, atom_poses, dir_type):
    """
    mol: rdkit molecule
    dir_type: direction type for bond_angle grpah
    """
    if len(mol.GetAtoms()) == 0:
        return None

    data = mol_to_graph_data(mol)

    data['atom_pos'] = np.array(atom_poses, 'float32')
    data['bond_length'] = Compound3DKit.get_bond_lengths(data['edges'], data['atom_pos'])
    BondAngleGraph_edges, bond_angles, bond_angle_dirs = \
        Compound3DKit.get_superedge_angles(data['edges'], data['atom_pos'])
    data['BondAngleGraph_edges'] = BondAngleGraph_edges
    data['bond_angle'] = np.array(bond_angles, 'float32')
    return data


def mol_to_geognn_graph_data_MMFF3d(mol):
    """tbd"""
    if len(mol.GetAtoms()) <= 400:
        mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
    else:
        atom_poses = Compound3DKit.get_2d_atom_poses(mol)
    return mol_to_geognn_graph_data(mol, atom_poses, dir_type='HT')


def mol_to_geognn_graph_data_raw3d(mol):
    """tbd"""
    atom_poses = Compound3DKit.get_atom_poses(mol, mol.GetConformer())
    return mol_to_geognn_graph_data(mol, atom_poses, dir_type='HT')

def obtain_3D_mol(smiles,name):
    mol = AllChem.MolFromSmiles(smiles)
    new_mol = Chem.AddHs(mol)
    res = AllChem.EmbedMultipleConfs(new_mol)
    ### MMFF generates multiple conformations
    res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
    new_mol = Chem.RemoveHs(new_mol)
    Chem.MolToMolFile(new_mol, name+'.mol')
    return new_mol

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings('ignore')

#============Parameter setting===============
MODEL = 'Test'  #['Train','Test','Test_other_method','Test_enantiomer','Test_excel']
test_mode='fixed' #fixed or random or enantiomer(extract enantimoers)
transfer_target='All_column' #trail name
Use_geometry_enhanced=True   #default:True
Use_column_info=True #default: True

atom_id_names = [
    "atomic_num", "chiral_tag", "degree", "explicit_valence",
    "formal_charge", "hybridization", "implicit_valence",
    "is_aromatic", "total_numHs",
]
bond_id_names = [
    "bond_dir", "bond_type", "is_in_ring"]

if Use_geometry_enhanced==True:
    bond_float_names = ["bond_length",'prop']

if Use_geometry_enhanced==False:
    bond_float_names=['prop']

bond_angle_float_names = ['bond_angle', 'TPSA', 'RASA', 'RPSA', 'MDEC', 'MATS']

column_specify={'ADH':[1,5,0,0],'ODH':[1,5,0,1],'IC':[0,5,1,2],'IA':[0,5,1,3],'OJH':[1,5,0,4],
                'ASH':[1,5,0,5],'IC3':[0,3,1,6],'IE':[0,5,1,7],'ID':[0,5,1,8],'OD3':[1,3,0,9],
                'IB':[0,5,1,10],'AD':[1,10,0,11],'AD3':[1,3,0,12],'IF':[0,5,1,13],'OD':[1,10,0,14],
                'AS':[1,10,0,15],'OJ3':[1,3,0,16],'IG':[0,5,1,17],'AZ':[1,10,0,18],'IAH':[0,5,1,19],
                'OJ':[1,10,0,20],'ICH':[0,5,1,21],'OZ3':[1,3,0,22],'IF3':[0,3,1,23],'IAU':[0,1.6,1,24]}
column_smile=['O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@H]1OC)NC4=CC(C)=CC(C)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@@H]1OC)NC4=CC(C)=CC(C)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(Cl)=CC(Cl)=C2)=O)[C@@H](OC(NC3=CC(Cl)=CC(Cl)=C3)=O)[C@@H]1OC)NC4=CC(Cl)=CC(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@H]1OC)NC4=CC(C)=CC(C)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(C2=CC=C(C)C=C2)=O)[C@@H](OC(C3=CC=C(C)C=C3)=O)[C@@H]1OC)C4=CC=C(C)C=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(N[C@@H](C)C2=CC=CC=C2)=O)[C@@H](OC(N[C@@H](C)C3=CC=CC=C3)=O)[C@H]1OC)N[C@@H](C)C4=CC=CC=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(Cl)=CC(Cl)=C2)=O)[C@@H](OC(NC3=CC(Cl)=CC(Cl)=C3)=O)[C@@H]1OC)NC4=CC(Cl)=CC(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(Cl)=CC(Cl)=C2)=O)[C@@H](OC(NC3=CC(Cl)=CC(Cl)=C3)=O)[C@H]1OC)NC4=CC(Cl)=CC(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC=CC(Cl)=C2)=O)[C@@H](OC(NC3=CC=CC(Cl)=C3)=O)[C@H]1OC)NC4=CC=CC(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@@H]1OC)NC4=CC(C)=CC(C)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@@H]1OC)NC4=CC(C)=CC(C)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@H]1OC)NC4=CC(C)=CC(C)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@H]1OC)NC4=CC(C)=CC(C)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC=C(C)C(Cl)=C2)=O)[C@@H](OC(NC3=CC=C(C)C(Cl)=C3)=O)[C@H]1OC)NC4=CC=C(C)C(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@@H]1OC)NC4=CC(C)=CC(C)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(N[C@@H](C)C2=CC=CC=C2)=O)[C@@H](OC(N[C@@H](C)C3=CC=CC=C3)=O)[C@H]1OC)N[C@@H](C)C4=CC=CC=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(C2=CC=C(C)C=C2)=O)[C@@H](OC(C3=CC=C(C)C=C3)=O)[C@@H]1OC)C4=CC=C(C)C=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(Cl)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(Cl)=C3)=O)[C@H]1OC)NC4=CC(C)=CC(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC=C(C)C(Cl)=C2)=O)[C@@H](OC(NC3=CC=C(C)C(Cl)=C3)=O)[C@H]1OC)NC4=CC=C(C)C(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@H]1OC)NC4=CC(C)=CC(C)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(C2=CC=C(C)C=C2)=O)[C@@H](OC(C3=CC=C(C)C=C3)=O)[C@@H]1OC)C4=CC=C(C)C=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(Cl)=CC(Cl)=C2)=O)[C@@H](OC(NC3=CC(Cl)=CC(Cl)=C3)=O)[C@@H]1OC)NC4=CC(Cl)=CC(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC=C(C)C(Cl)=C2)=O)[C@@H](OC(NC3=CC=C(C)C(Cl)=C3)=O)[C@@H]1OC)NC4=CC=C(C)C(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC=C(C)C(Cl)=C2)=O)[C@@H](OC(NC3=CC=C(C)C(Cl)=C3)=O)[C@H]1OC)NC4=CC=C(C)C(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@H]1OC)NC4=CC(C)=CC(C)=C4']
column_name=['ADH','ODH','IC','IA','OJH','ASH','IC3','IE','ID','OD3', 'IB','AD','AD3',
            'IF','OD','AS','OJ3','IG','AZ','IAH','OJ','ICH','OZ3','IF3','IAU']
full_atom_feature_dims = get_atom_feature_dims(atom_id_names)
full_bond_feature_dims = get_bond_feature_dims(bond_id_names)


if Use_column_info==True:
    bond_id_names.extend(['coated', 'immobilized'])
    bond_float_names.extend(['diameter'])
    if Use_geometry_enhanced==True:
        bond_angle_float_names.extend(['column_TPSA', 'column_TPSA', 'column_TPSA', 'column_MDEC', 'column_MATS'])
    else:
        bond_float_names.extend(['column_TPSA', 'column_TPSA', 'column_TPSA', 'column_MDEC', 'column_MATS'])
    full_bond_feature_dims.extend([2,2])

calc = Calculator(descriptors, ignore_3D=False)


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim + 5, emb_dim)  # 不同维度的属性用不同的Embedding方法
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding

class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim + 5, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding

class RBF(torch.nn.Module):
    """
    Radial Basis Function
    """

    def __init__(self, centers, gamma, dtype='float32'):
        super(RBF, self).__init__()
        self.centers = centers.reshape([1, -1])
        self.gamma = gamma

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = x.reshape([-1, 1])
        return torch.exp(-self.gamma * torch.square(x - self.centers))

class BondFloatRBF(torch.nn.Module):
    """
    Bond Float Encoder using Radial Basis Functions
    """

    def __init__(self, bond_float_names, embed_dim, rbf_params=None):
        super(BondFloatRBF, self).__init__()
        self.bond_float_names = bond_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_length': (nn.Parameter(torch.arange(0, 2, 0.1)), nn.Parameter(torch.Tensor([10.0]))),
                # (centers, gamma)
                'prop': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
                'diameter': (nn.Parameter(torch.arange(3, 12, 0.3)), nn.Parameter(torch.Tensor([1.0]))),
                ##=========Only for pure GNN===============
                'column_TPSA': (nn.Parameter(torch.arange(0, 1, 0.05).to(torch.float32)), nn.Parameter(torch.Tensor([1.0]))),
                'column_RASA': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
                'column_RPSA': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
                'column_MDEC': (nn.Parameter(torch.arange(0, 10, 0.5)), nn.Parameter(torch.Tensor([2.0]))),
                'column_MATS': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = torch.nn.ModuleList()
        self.rbf_list = torch.nn.ModuleList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers.to(device), gamma.to(device))
            self.rbf_list.append(rbf)
            linear = torch.nn.Linear(len(centers), embed_dim).cuda()
            self.linear_list.append(linear)

    def forward(self, bond_float_features):
        """
        Args:
            bond_float_features(dict of tensor): bond float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            x = bond_float_features[:, i].reshape(-1, 1)
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed

class BondAngleFloatRBF(torch.nn.Module):
    """
    Bond Angle Float Encoder using Radial Basis Functions
    """

    def __init__(self, bond_angle_float_names, embed_dim, rbf_params=None):
        super(BondAngleFloatRBF, self).__init__()
        self.bond_angle_float_names = bond_angle_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_angle': (nn.Parameter(torch.arange(0, torch.pi, 0.1)), nn.Parameter(torch.Tensor([10.0]))),
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = torch.nn.ModuleList()
        self.rbf_list = torch.nn.ModuleList()
        for name in self.bond_angle_float_names:
            if name == 'bond_angle':
                centers, gamma = self.rbf_params[name]
                rbf = RBF(centers.to(device), gamma.to(device))
                self.rbf_list.append(rbf)
                linear = nn.Linear(len(centers), embed_dim)
                self.linear_list.append(linear)
            else:
                linear = nn.Linear(len(self.bond_angle_float_names) - 1, embed_dim)
                self.linear_list.append(linear)
                break

    def forward(self, bond_angle_float_features):
        """
        Args:
            bond_angle_float_features(dict of tensor): bond angle float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_angle_float_names):
            if name == 'bond_angle':
                x = bond_angle_float_features[:, i].reshape(-1, 1)
                rbf_x = self.rbf_list[i](x)
                out_embed += self.linear_list[i](rbf_x)
            else:
                x = bond_angle_float_features[:, 1:]
                out_embed += self.linear_list[i](x)
                break
        return out_embed

class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
                                 nn.Linear(emb_dim, emb_dim))
        self.eps = nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = edge_attr
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

# GNN to generate node embedding
class GINNodeEmbedding(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layers, emb_dim, drop_ratio=0.5, JK="last", residual=False):
        """GIN Node Embedding Module
        采用多层GINConv实现图上结点的嵌入。
        """

        super(GINNodeEmbedding, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        # add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder=BondEncoder(emb_dim)
        self.bond_float_encoder=BondFloatRBF(bond_float_names,emb_dim)
        self.bond_angle_encoder=BondAngleFloatRBF(bond_angle_float_names,emb_dim)

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        self.convs_bond_angle=torch.nn.ModuleList()
        self.convs_bond_float=torch.nn.ModuleList()
        self.convs_bond_embeding=torch.nn.ModuleList()
        self.convs_angle_float=torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms_ba = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GINConv(emb_dim))
            self.convs_bond_angle.append(GINConv(emb_dim))
            self.convs_bond_embeding.append(BondEncoder(emb_dim))
            self.convs_bond_float.append(BondFloatRBF(bond_float_names,emb_dim))
            self.convs_angle_float.append(BondAngleFloatRBF(bond_angle_float_names,emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            self.batch_norms_ba.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_atom_bond,batched_bond_angle):
        x, edge_index, edge_attr = batched_atom_bond.x, batched_atom_bond.edge_index, batched_atom_bond.edge_attr
        edge_index_ba,edge_attr_ba= batched_bond_angle.edge_index, batched_bond_angle.edge_attr
        # computing input node embedding
        h_list = [self.atom_encoder(x)]  # 先将类别型原子属性转化为原子嵌入

        if Use_geometry_enhanced==True:
            h_list_ba=[self.bond_float_encoder(edge_attr[:,len(bond_id_names):edge_attr.shape[1]+1].to(torch.float32))+self.bond_encoder(edge_attr[:,0:len(bond_id_names)].to(torch.int64))]
            for layer in range(self.num_layers):
                h = self.convs[layer](h_list[layer], edge_index, h_list_ba[layer])
                cur_h_ba=self.convs_bond_embeding[layer](edge_attr[:,0:len(bond_id_names)].to(torch.int64))+self.convs_bond_float[layer](edge_attr[:,len(bond_id_names):edge_attr.shape[1]+1].to(torch.float32))
                cur_angle_hidden=self.convs_angle_float[layer](edge_attr_ba)
                h_ba=self.convs_bond_angle[layer](cur_h_ba, edge_index_ba, cur_angle_hidden)

                if layer == self.num_layers - 1:
                    # remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training=self.training)
                    h_ba = F.dropout(h_ba, self.drop_ratio, training=self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
                    h_ba = F.dropout(F.relu(h_ba), self.drop_ratio, training=self.training)
                if self.residual:
                    h += h_list[layer]
                    h_ba+=h_list_ba[layer]
                h_list.append(h)
                h_list_ba.append(h_ba)


            # Different implementations of Jk-concat
            if self.JK == "last":
                node_representation = h_list[-1]
                edge_representation = h_list_ba[-1]
            elif self.JK == "sum":
                node_representation = 0
                edge_representation = 0
                for layer in range(self.num_layers + 1):
                    node_representation += h_list[layer]
                    edge_representation += h_list_ba[layer]

            return node_representation,edge_representation
        if Use_geometry_enhanced==False:
            for layer in range(self.num_layers):
                h = self.convs[layer](h_list[layer], edge_index,
                                      self.convs_bond_embeding[layer](edge_attr[:, 0:len(bond_id_names)].to(torch.int64)) +
                                      self.convs_bond_float[layer](
                                          edge_attr[:, len(bond_id_names):edge_attr.shape[1] + 1].to(torch.float32)))
                h = self.batch_norms[layer](h)
                if layer == self.num_layers - 1:
                    # remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training=self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

                if self.residual:
                    h += h_list[layer]

                h_list.append(h)

            # Different implementations of Jk-concat
            if self.JK == "last":
                node_representation = h_list[-1]
            elif self.JK == "sum":
                node_representation = 0
                for layer in range(self.num_layers + 1):
                    node_representation += h_list[layer]

            return node_representation

class GINGraphPooling(nn.Module):

    def __init__(self, num_tasks=1, num_layers=5, emb_dim=300, residual=False, drop_ratio=0, JK="last", graph_pooling="attention",
                 descriptor_dim=1781):
        """GIN Graph Pooling Module

        此模块首先采用GINNodeEmbedding模块对图上每一个节点做嵌入，然后对节点嵌入做池化得到图的嵌入，最后用一层线性变换得到图的最终的表示（graph representation）。

        Args:
            num_tasks (int, optional): number of labels to be predicted. Defaults to 1 (控制了图表示的维度，dimension of graph representation).
            num_layers (int, optional): number of GINConv layers. Defaults to 5.
            emb_dim (int, optional): dimension of node embedding. Defaults to 300.
            residual (bool, optional): adding residual connection or not. Defaults to False.
            drop_ratio (float, optional): dropout rate. Defaults to 0.
            JK (str, optional): 可选的值为"last"和"sum"。选"last"，只取最后一层的结点的嵌入，选"sum"对各层的结点的嵌入求和。Defaults to "last".
            graph_pooling (str, optional): pooling method of node embedding. 可选的值为"sum"，"mean"，"max"，"attention"和"set2set"。 Defaults to "sum".

        Out:
            graph representation
        """
        super(GINGraphPooling, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.descriptor_dim=descriptor_dim
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn_node = GINNodeEmbedding(num_layers, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual)

        # Pooling function to generate whole-graph embeddings
        if graph_pooling == "sum":
            self.pool = global_add_pool

        elif graph_pooling == "mean":
            self.pool = global_mean_pool

        elif graph_pooling == "max":
            self.pool = global_max_pool

        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1)))


        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)

        self.NN_descriptor = nn.Sequential(nn.Linear(self.descriptor_dim, self.emb_dim),
                                           nn.Sigmoid(),
                                           nn.Linear(self.emb_dim, self.emb_dim))

        self.sigmoid = nn.Sigmoid()

    def forward(self, batched_atom_bond,batched_bond_angle):
        if Use_geometry_enhanced==True:
            h_node,h_node_ba= self.gnn_node(batched_atom_bond,batched_bond_angle)
        else:
            h_node= self.gnn_node(batched_atom_bond, batched_bond_angle)
        h_graph = self.pool(h_node, batched_atom_bond.batch)
        output = self.graph_pred_linear(h_graph)
        if self.training:
            return output,h_graph
        else:
            # At inference time, relu is applied to output to ensure positivity
            return torch.clamp(output, min=0, max=1e8),h_graph

def mord(mol, nBits=1826, errors_as_zeros=True):
    try:
        result = calc(mol)
        desc_list = [r if not is_missing(r) else 0 for r in result]
        np_arr = np.array(desc_list)
        return np_arr
    except:
        return np.NaN if not errors_as_zeros else np.zeros((nBits,), dtype=np.float32)

def load_3D_mol():
    dir = 'mol_save/'
    for root, dirs, files in os.walk(dir):
        file_names = files
    file_names.sort(key=lambda x: int(x[x.find('_') + 5:x.find(".")]))  # 按照前面的数字字符排序
    mol_save = []
    for file_name in file_names:
        mol_save.append(Chem.MolFromMolFile(dir + file_name))
    return mol_save

def parse_args():
    parser = argparse.ArgumentParser(description='Graph data miming with GNN')
    parser.add_argument('--task_name', type=str, default='GINGraphPooling',
                        help='task name')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='dimensionality of hidden units in GNNs (default: 256)')
    parser.add_argument('--drop_ratio', type=float, default=0.,
                        help='dropout ratio (default: 0.)')
    parser.add_argument('--save_test', action='store_true')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='early stop (default: 10)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset_root', type=str, default="dataset",
                        help='dataset root')
    args = parser.parse_args()

    return args

def calc_dragon_type_desc(mol):
    compound_mol = mol
    compound_MolWt = Descriptors.ExactMolWt(compound_mol)
    compound_TPSA = Chem.rdMolDescriptors.CalcTPSA(compound_mol)
    compound_nRotB = Descriptors.NumRotatableBonds(compound_mol)  # Number of rotable bonds
    compound_HBD = Descriptors.NumHDonors(compound_mol)  # Number of H bond donors
    compound_HBA = Descriptors.NumHAcceptors(compound_mol)  # Number of H bond acceptors
    compound_LogP = Descriptors.MolLogP(compound_mol)  # LogP
    return rdMolDescriptors.CalcAUTOCORR3D(mol) + rdMolDescriptors.CalcMORSE(mol) + \
           rdMolDescriptors.CalcRDF(mol) + rdMolDescriptors.CalcWHIM(mol) + \
           [compound_MolWt, compound_TPSA, compound_nRotB, compound_HBD, compound_HBA, compound_LogP]


def eval(model, device, loader_atom_bond,loader_bond_angle):
    model.eval()
    y_true = []
    y_pred = []
    y_pred_10=[]
    y_pred_90=[]

    with torch.no_grad():
        for _, batch in enumerate(zip(loader_atom_bond,loader_bond_angle)):
            batch_atom_bond = batch[0]
            batch_bond_angle = batch[1]
            batch_atom_bond = batch_atom_bond.to(device)
            batch_bond_angle = batch_bond_angle.to(device)
            pred = model(batch_atom_bond,batch_bond_angle)[0]

            y_true.append(batch_atom_bond.y.detach().cpu().reshape(-1))
            y_pred.append(pred[:,1].detach().cpu())
            y_pred_10.append(pred[:,0].detach().cpu())
            y_pred_90.append(pred[:,2].detach().cpu())
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    y_pred_10 = torch.cat(y_pred_10, dim=0)
    y_pred_90 = torch.cat(y_pred_90, dim=0)
    # plt.plot(y_pred.cpu().data.numpy(),c='blue')
    # plt.plot(y_pred_10.cpu().data.numpy(),c='yellow')
    # plt.plot(y_pred_90.cpu().data.numpy(),c='black')
    # plt.plot(y_true.cpu().data.numpy(),c='red')
    #plt.show()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return torch.mean((y_true - y_pred) ** 2).data.numpy()


def cal_prob(prediction):
    '''
    calculate the separation probability Sp
    '''
    #input  prediction=[pred_1,pred_2]
    #output: Sp
    a=prediction[0][0]
    b=prediction[1][0]
    if a[2]<b[0]:
        return 1
    elif a[0]>b[2]:
        return 1
    else:
        length=min(a[2],b[2])-max(a[0],b[0])
        all=max(a[2],b[2])-min(a[0],b[0])
        return 1-length/(all)



args = parse_args()
nn_params = {
    'num_tasks': 3,
    'num_layers': args.num_layers,
    'emb_dim': args.emb_dim,
    'drop_ratio': args.drop_ratio,
    'graph_pooling': args.graph_pooling,
    'descriptor_dim': 1827
}
device = args.device
model = GINGraphPooling(**nn_params).to(device)


'''
Given two compounds and predict the RT in different condition
'''

root = Tk()
root.withdraw()
input_smile=tk.simpledialog.askstring('QGeoGNN', 'Input smiles of two compounds \n'+'Split by comma(,)\n'
                                                 'Example: N[C@@H](C)C(O)=O,N[C@H](C)C(O)=O', initialvalue='')
input_eluent=tk.simpledialog.askfloat('QGeoGNN', 'Input Proportion\n'+'Example: 0.1', initialvalue='')
input_speed=tk.simpledialog.askfloat('QGeoGNN', 'Input flow rate\n'+'Example: 1', initialvalue='')
input_column=tk.simpledialog.askstring('QGeoGNN', 'Input candidate columns\n'+'Split by comma(,)\n'
                                       'Example: ADH,ODH,IA,IC,OJH,IG', initialvalue='')
smiles=input_smile.split(',')
column=input_column.split(',')
speed=[]
eluent=[]
for i in range(len(smiles)):
    speed.append(input_speed)
    eluent.append(input_eluent)


column_descriptor = np.load('column_descriptor.npy', allow_pickle=True)
Prediction=[]
Sp=[]
for predict_column in column:
    col_specify=column_specify[predict_column]
    col_des = np.array(column_descriptor[col_specify[3]])
    mols=[]
    y_pred=[]
    all_descriptor=[]
    dataset=[]
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        mols.append(mol)
    for smile in smiles:
        mol = obtain_3D_mol(smile, 'conform')
        mol = Chem.MolFromMolFile(f"conform.mol")
        all_descriptor.append(mord(mol))
        dataset.append(mol_to_geognn_graph_data_MMFF3d(mol))

    for i in range(0, len(dataset)):
        data = dataset[i]
        atom_feature = []
        bond_feature = []
        for name in atom_id_names:
            atom_feature.append(data[name])
        for name in bond_id_names[0:3]:
            bond_feature.append(data[name])
        atom_feature = torch.from_numpy(np.array(atom_feature).T).to(torch.int64)
        bond_feature = torch.from_numpy(np.array(bond_feature).T).to(torch.int64)
        bond_float_feature = torch.from_numpy(data['bond_length'].astype(np.float32))
        bond_angle_feature = torch.from_numpy(data['bond_angle'].astype(np.float32))
        y = torch.Tensor([float(speed[i])])
        edge_index = torch.from_numpy(data['edges'].T).to(torch.int64)
        bond_index = torch.from_numpy(data['BondAngleGraph_edges'].T).to(torch.int64)

        prop = torch.ones([bond_feature.shape[0]]) * eluent[i]
        coated = torch.ones([bond_feature.shape[0]]) * col_specify[0]
        diameter = torch.ones([bond_feature.shape[0]]) * col_specify[1]
        immobilized = torch.ones([bond_feature.shape[0]]) * col_specify[2]

        TPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][820] / 100
        RASA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][821]
        RPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][822]
        MDEC = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][1568]
        MATS = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][457]

        col_TPSA = torch.ones([bond_angle_feature.shape[0]]) * col_des[820] / 100
        col_RASA = torch.ones([bond_angle_feature.shape[0]]) * col_des[821]
        col_RPSA = torch.ones([bond_angle_feature.shape[0]]) * col_des[822]
        col_MDEC = torch.ones([bond_angle_feature.shape[0]]) * col_des[1568]
        col_MATS = torch.ones([bond_angle_feature.shape[0]]) * col_des[457]

        bond_feature = torch.cat([bond_feature, coated.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, immobilized.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, bond_float_feature.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, prop.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, diameter.reshape(-1, 1)], dim=1)

        bond_angle_feature = torch.cat([bond_angle_feature.reshape(-1, 1), TPSA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, RASA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, RPSA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, MDEC.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, MATS.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, col_TPSA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, col_RASA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, col_RPSA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, col_MDEC.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, col_MATS.reshape(-1, 1)], dim=1)

        data_atom_bond = Data(atom_feature, edge_index, bond_feature, y)
        data_bond_angle = Data(edge_index=bond_index, edge_attr=bond_angle_feature)
        model.load_state_dict(
            torch.load(f'GeoGNN_model.pth'))
        model.eval()

        pred, h_graph = model(data_atom_bond.to(device),data_bond_angle.to(device))

        y_pred.append(pred.detach().cpu().data.numpy()/speed[i])

    Prediction.append(y_pred)
    Sp.append(cal_prob(y_pred))

plt.figure(1,figsize=(2.5,3),dpi=300)
plt.style.use('ggplot')
font1egend = {'family': 'Arial',
         'weight': 'normal',
         #"style": 'italic',
         'size': 5,
         }

print(f'The smile is: {input_smile}\n'+f'The proportion is {input_eluent}\n'+f'The flow rate is {input_speed}\n')
for i in range(len(column)):
    column_type=column[i]
    pred_1=Prediction[i][0][0][1]
    pred_2 = Prediction[i][1][0][1]
    sp=Sp[i]
    plt.scatter([sp,sp],[pred_1,pred_2],alpha=0.7,label=column_type)
    print(f'Sp of {column_type} is {sp} with RTs are {pred_1},{pred_2}\n')

Sp_array=np.array(Sp)
max_index=np.argmax(Sp_array)
max_column=column[max_index]
print(f'I recommend {max_column} with Sp={np.max(Sp_array)}')
plt.xticks(fontproperties='Arial', size=7)
plt.yticks(fontproperties='Arial', size=7)
plt.xlabel('Sp',fontproperties='Arial', size=7)
plt.ylabel('Predicted RT',fontproperties='Arial', size=7)
plt.title(f'The smile is: {input_smile}\n'+f'The proportion is {input_eluent}\n'+f'The flow rate is {input_speed}\n'+
          f'I recommend {max_column} with Sp={np.max(Sp_array)}',fontproperties='Arial', size=5,color='red')
plt.legend(prop=font1egend,loc='lower center', ncol = 2)
plt.tight_layout()
plt.show()












