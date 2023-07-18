'''
made by Hao Xu
2022.10.28
Single_column_prediction by QGeoGNN
'''

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from compound_tools import *
import torch.nn.functional as F
import pymysql
import lightgbm as lgb
from torch_geometric.data import DataLoader
import pandas as pd
from rdkit.Chem import Descriptors
from torch_geometric.data import Data
import argparse
from tqdm import tqdm
# from torch_geometric.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import warnings
import matplotlib.pyplot as plt
from rdkit.Chem.Descriptors import rdMolDescriptors
import os
from mordred import Calculator, descriptors, is_missing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from pyDOE import *
from scipy.stats import spearmanr
import mordred
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


#==================Default   Settings====================
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings('ignore')
calc = Calculator(descriptors, ignore_3D=False)
atom_id_names = [
    "atomic_num", "chiral_tag", "degree", "explicit_valence",
    "formal_charge", "hybridization", "implicit_valence",
    "is_aromatic", "total_numHs",
]
bond_id_names = [
    "bond_dir", "bond_type", "is_in_ring"]
bond_float_names=["bond_length",'prop']
bond_angle_float_names=['bond_angle', 'TPSA', 'RASA', 'RPSA', 'MDEC', 'MATS']
condition_name=['silica_surface','replace_basis']   #Not used in Single column
condition_float_name=['eluent','grain_radian']      #Not used in Single column
full_atom_feature_dims = get_atom_feature_dims(atom_id_names)
full_bond_feature_dims = get_bond_feature_dims(bond_id_names)


#===================Adjust settings===============
MODEL = 'Test'   #['Train','Test','Test_enantiomer']
test_mode='fixed' #fixed or random
transfer_target='IC'  #target column:['ADH','ODH','IA','IC']


#==================Functions===========================
class AtomEncoder(torch.nn.Module):
    """该类用于对原子属性做嵌入。
    记`N`为原子属性的维度，则原子属性表示为`[x1, x2, ..., xi, xN]`，其中任意的一维度`xi`都是类别型数据。full_atom_feature_dims[i]存储了原子属性`xi`的类别数量。
    该类将任意的原子属性`[x1, x2, ..., xi, xN]`转换为原子的嵌入`x_embedding`（维度为emb_dim）。
    """

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
                # 'TPSA':(nn.Parameter(torch.arange(0, 100, 5).to(torch.float32)), nn.Parameter(torch.Tensor([5.0]))),
                # 'RASA': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
                # 'RPSA': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0])))
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
                # (centers, gamma)
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

class ConditionEmbeding(torch.nn.Module):
    """
    Not used in single_column prediction
    """

    def __init__(self, condition_names,condition_float_names, embed_dim, rbf_params=None):
        super(ConditionEmbeding, self).__init__()
        self.condition_names = condition_names
        self.condition_float_names=condition_float_names

        if rbf_params is None:
            self.rbf_params = {
                'eluent': (nn.Parameter(torch.arange(0,1,0.1)), nn.Parameter(torch.Tensor([10.0]))),
                'grain_radian': (nn.Parameter(torch.arange(0,10,0.1)), nn.Parameter(torch.Tensor([10.0])))# (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = torch.nn.ModuleList()
        self.rbf_list = torch.nn.ModuleList()
        self.embedding_list=torch.nn.ModuleList()
        for name in self.condition_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers.to(device), gamma.to(device))
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim).to(device)
            self.linear_list.append(linear)
        for name in self.condition_names:
            if name=='silica_surface':
                emb = torch.nn.Embedding(2 + 5, embed_dim).to(device)
                torch.nn.init.xavier_uniform_(emb.weight.data)
                self.embedding_list.append(emb)
            elif name=='replace_basis':
                emb = torch.nn.Embedding(6 + 5, embed_dim).to(device)
                torch.nn.init.xavier_uniform_(emb.weight.data)
                self.embedding_list.append(emb)

    def forward(self, condition):
        """
        Args:
            bond_angle_float_features(dict of tensor): bond angle float features.
        """
        out_embed = 0
        for i, name in enumerate(self.condition_float_names):
            x = condition[:,2*i+1]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        for i, name in enumerate(self.condition_names):
            x = self.embedding_list[i](condition[:,2*i].to(torch.int64))
            out_embed += x
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

class GINNodeEmbedding(torch.nn.Module):
    """
    Node embedding
    """

    def __init__(self, num_layers, emb_dim, drop_ratio=0.5, JK="last", residual=False):
        """GIN Node Embedding Module
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
        self.condition_encoder=ConditionEmbeding(condition_name,condition_float_name,emb_dim)  #Not used in single_column prediction
        # List of GNNs
        self.convs = torch.nn.ModuleList()
        self.convs_bond_angle=torch.nn.ModuleList()
        self.convs_bond_float=torch.nn.ModuleList()
        self.convs_bond_embeding=torch.nn.ModuleList()
        self.convs_angle_float=torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms_ba = torch.nn.ModuleList()
        self.convs_condition=torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GINConv(emb_dim))
            self.convs_bond_angle.append(GINConv(emb_dim))
            self.convs_bond_embeding.append(BondEncoder(emb_dim))
            self.convs_bond_float.append(BondFloatRBF(bond_float_names,emb_dim))
            self.convs_angle_float.append(BondAngleFloatRBF(bond_angle_float_names,emb_dim))
            self.convs_condition.append(ConditionEmbeding(condition_name,condition_float_name,emb_dim)) #Not used in single_column prediction
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            self.batch_norms_ba.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_atom_bond,batched_bond_angle):
        x, edge_index, edge_attr = batched_atom_bond.x, batched_atom_bond.edge_index, batched_atom_bond.edge_attr
        edge_index_ba,edge_attr_ba= batched_bond_angle.edge_index, batched_bond_angle.edge_attr
        # computing input node embedding
        h_list = [self.atom_encoder(x)]  # 先将类别型原子属性转化为原子嵌入
        h_list_ba=[self.bond_float_encoder(edge_attr[:,3:edge_attr.shape[1]+1].to(torch.float32))+self.bond_encoder(edge_attr[:,0:3].to(torch.int64))]
        for layer in range(self.num_layers):
            h = self.convs[layer](h_list[layer], edge_index, h_list_ba[layer])
            cur_h_ba=self.convs_bond_embeding[layer](edge_attr[:,0:3].to(torch.int64))+self.convs_bond_float[layer](edge_attr[:,3:edge_attr.shape[1]+1].to(torch.float32))
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

class GINGraphPooling(nn.Module):

    def __init__(self, num_tasks=1, num_layers=5, emb_dim=300, residual=False, drop_ratio=0, JK="last", graph_pooling="attention",
                 descriptor_dim=1781):
        """GIN Graph Pooling Module
        Args:
            num_tasks (int, optional): number of labels to be predicted. Defaults to 1 (dimension of graph representation).
            num_layers (int, optional): number of GINConv layers. Defaults to 5.
            emb_dim (int, optional): dimension of node embedding. Defaults to 300.
            residual (bool, optional): adding residual connection or not. Defaults to False.
            drop_ratio (float, optional): dropout rate. Defaults to 0.
            JK (str, optional): Defaults to "last".
            graph_pooling (str, optional): pooling method of node embedding. Defaults to "sum".

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
        h_node,h_node_ba= self.gnn_node(batched_atom_bond,batched_bond_angle)
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
    parser.add_argument('--device', type=int, default=1,
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

def prepartion(args):
    save_dir = os.path.join('saves', args.task_name)
    args.save_dir = save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    args.output_file = open(os.path.join(args.save_dir, 'output'), 'a')
    print(args, file=args.output_file, flush=True)

def q_loss(q,y_true,y_pred):
    e = (y_true-y_pred)
    return torch.mean(torch.maximum(q*e, (q-1)*e))

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


def train(model, device, loader_atom_bond, loader_bond_angle, optimizer, criterion_fn):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(zip(loader_atom_bond,loader_bond_angle)):
        batch_atom_bond=batch[0]
        batch_bond_angle=batch[1]
        batch_atom_bond = batch_atom_bond.to(device)
        batch_bond_angle=batch_bond_angle.to(device)
        pred = model(batch_atom_bond,batch_bond_angle)[0]#.view(-1, )
        true=batch_atom_bond.y
        optimizer.zero_grad()
        loss=q_loss(0.1,true,pred[:,0])+torch.mean((true-pred[:,1])**2)+q_loss(0.9,true,pred[:,2])\
             +torch.mean(torch.relu(pred[:,0]-pred[:,1]))+torch.mean(torch.relu(pred[:,1]-pred[:,2]))+torch.mean(torch.relu(2-pred))
        #loss = criterion_fn(pred, batch_atom_bond.y)
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)


def test(model, device, loader_atom_bond,loader_bond_angle):
    model.eval()
    y_pred = []
    y_true = []
    y_pred_10 = []
    y_pred_90 = []
    with torch.no_grad():
        for _, batch in enumerate(zip(loader_atom_bond,loader_bond_angle)):
            batch_atom_bond = batch[0]
            batch_bond_angle = batch[1]
            batch_atom_bond = batch_atom_bond.to(device)
            batch_bond_angle = batch_bond_angle.to(device)
            pred = model(batch_atom_bond,batch_bond_angle)[0]
            y_true.append(batch_atom_bond.y.detach().cpu().reshape(-1,))
            y_pred.append(pred[:, 1].detach().cpu())
            y_pred_10.append(pred[:, 0].detach().cpu())
            y_pred_90.append(pred[:, 2].detach().cpu())
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        y_pred_10 = torch.cat(y_pred_10, dim=0)
        y_pred_90 = torch.cat(y_pred_90, dim=0)
        # plt.figure(10)
        # box=np.array([y_pred_10.cpu().data.numpy(),y_pred.cpu().data.numpy(),y_pred_90.cpu().data.numpy()])
        # plt.boxplot(box)
        # plt.scatter(np.arange(0,y_true.shape[0],1),y_true.cpu().data.numpy(), c='red')

        # plt.plot(y_pred.cpu().data.numpy(), c='blue')
        # plt.plot(y_pred_10.cpu().data.numpy(), c='yellow')
        # plt.plot(y_pred_90.cpu().data.numpy(), c='black')
        # plt.plot(y_true.cpu().data.numpy(), c='red')
        #plt.show()


    R_square = 1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())
    test_mae=torch.mean((y_true - y_pred) ** 2)
    print(R_square)
    return y_pred, y_true,R_square,test_mae,y_pred_10,y_pred_90


def get_feature(model, device, loader):
    model.eval()
    y_pred = []
    y_true = []
    GNN_feature = []
    with torch.no_grad():
        for _, batch in enumerate(loader):
            batch = batch.to(device)
            pred, h_graph = model(batch)
            y_pred.append(pred.detach().cpu())
            y_true.append(batch.y.detach().cpu().reshape(pred.shape[0], pred.shape[1]))
            GNN_feature.append(h_graph.detach().cpu())
    y_pred = torch.cat(y_pred, dim=0).cpu().data.numpy()
    y_true = torch.cat(y_true, dim=0).cpu().data.numpy()
    GNN_feature = torch.cat(GNN_feature, dim=0).cpu().data.numpy()
    T_1_pred = y_pred[:, 0]
    T_1_true = y_true[:, 0]
    return y_pred, y_true, GNN_feature

class ANN(nn.Module):
    '''
    Construct artificial neural network
    '''
    def __init__(self, in_neuron, hidden_neuron, out_neuron):
        super(ANN, self).__init__()
        self.input_layer = nn.Linear(in_neuron, hidden_neuron)
        self.hidden_layer = nn.Linear(hidden_neuron, hidden_neuron)
        self.output_layer = nn.Linear(hidden_neuron, out_neuron)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.sigmoid(x)
        x = self.hidden_layer(x)
        x = F.sigmoid(x)
        x = self.hidden_layer(x)
        x = F.sigmoid(x)
        x = self.hidden_layer(x)
        x = F.sigmoid(x)
        x = self.hidden_layer(x)
        x = F.sigmoid(x)
        x = self.output_layer(x)
        return x

def Construct_dataset(dataset,data_index, T1, speed, eluent,column):
    graph_atom_bond = []
    graph_bond_angle = []
    big_index = []
    if column=='ODH':
        all_descriptor=np.load('dataset/dataset_ODH_morder.npy')
    if column=='ADH':
        all_descriptor = np.load('dataset/dataset_ADH_charity_morder_0606.npy')
    if column=='IC':
        all_descriptor=np.load('dataset/dataset_IC_charity_morder_0823.npy')
    if column == 'IA':
        all_descriptor = np.load('dataset/dataset_IA_charity_morder_0823.npy')

    for i in range(len(dataset)):
        data = dataset[i]
        atom_feature = []
        bond_feature = []
        for name in atom_id_names:
            atom_feature.append(data[name])
        for name in bond_id_names:
            bond_feature.append(data[name])
        atom_feature = torch.from_numpy(np.array(atom_feature).T).to(torch.int64)
        bond_feature = torch.from_numpy(np.array(bond_feature).T).to(torch.int64)
        bond_float_feature = torch.from_numpy(data['bond_length'].astype(np.float32))
        bond_angle_feature = torch.from_numpy(data['bond_angle'].astype(np.float32))
        y = torch.Tensor([float(T1[i]) * float(speed[i])])
        edge_index = torch.from_numpy(data['edges'].T).to(torch.int64)
        bond_index = torch.from_numpy(data['BondAngleGraph_edges'].T).to(torch.int64)
        data_index_int=torch.from_numpy(np.array(data_index[i])).to(torch.int64)

        prop=torch.ones([bond_feature.shape[0]])*eluent[i]

        TPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 820]/100
        RASA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 821]
        RPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 822]
        MDEC=torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 1568]
        MATS=torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 457]

        bond_feature=torch.cat([bond_feature,bond_float_feature.reshape(-1,1)],dim=1)
        bond_feature = torch.cat([bond_feature, prop.reshape(-1, 1)], dim=1)
        bond_angle_feature=bond_angle_feature.reshape(-1,1)
        bond_angle_feature = torch.cat([bond_angle_feature.reshape(-1, 1), TPSA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, RASA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, RPSA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, MDEC.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, MATS.reshape(-1, 1)], dim=1)

        if y[0]>60:
            big_index.append(i)
            continue

        data_atom_bond = Data(atom_feature, edge_index, bond_feature, y,data_index=data_index_int)
        data_bond_angle= Data(edge_index=bond_index, edge_attr=bond_angle_feature)
        graph_atom_bond.append(data_atom_bond)
        graph_bond_angle.append(data_bond_angle)
    return graph_atom_bond,graph_bond_angle,big_index



#-------------load data----------------
'''
The Graph construction is prepared and saved beforehand to accelerate the process by the code:
for smile in smiles:
    mol = obtain_3D_mol(smile, 'trail')
    mol = Chem.MolFromMolFile(f"trail.mol")
    all_descriptor.append(mord(mol))
    dataset.append(mol_to_geognn_graph_data_MMFF3d(mol))
'''

if transfer_target=='ODH':
    HPLC_ODH=pd.read_csv('dataset/ODH_charity_0616.csv')
    HPLC_ODH=HPLC_ODH.drop(4231)  #conformer error
    all_smile_ODH = HPLC_ODH['SMILES'].values
    T1_ODH = HPLC_ODH['RT'].values
    Speed_ODH = HPLC_ODH['Speed'].values
    Prop_ODH = HPLC_ODH['i-PrOH_proportion'].values
    dataset_ODH=np.load('dataset/dataset_ODH.npy',allow_pickle=True).tolist()
    index_ODH=HPLC_ODH['Unnamed: 0'].values

if transfer_target=='ADH':
    HPLC_ADH=pd.read_csv('dataset/ADH_charity_0606.csv')
    all_smile_ADH = HPLC_ADH['SMILES'].values
    T1_ADH = HPLC_ADH['RT'].values
    Speed_ADH = HPLC_ADH['Speed'].values
    Prop_ADH = HPLC_ADH['i-PrOH_proportion'].values
    dataset_ADH=np.load('dataset/dataset_ADH_charity_0606.npy',allow_pickle=True).tolist()
    index_ADH=HPLC_ADH['Unnamed: 0'].values

if transfer_target=='IC':
    HPLC_IC=pd.read_csv('dataset/IC_charity_0823.csv')
    bad_IC_index=np.load('dataset/bad_IC.npy')  #Some compounds that cannot get 3D conformer by RDKit
    HPLC_IC=HPLC_IC.drop(bad_IC_index)  #conformer error
    all_smile_IC = HPLC_IC['SMILES'].values
    T1_IC = HPLC_IC['RT'].values
    Speed_IC = HPLC_IC['Speed'].values
    Prop_IC = HPLC_IC['i-PrOH_proportion'].values
    dataset_IC=np.load('dataset/dataset_IC_charity_0823.npy',allow_pickle=True).tolist()
    index_IC=HPLC_IC['Unnamed: 0'].values

if transfer_target=='IA':
    HPLC_IA=pd.read_csv('dataset/IA_charity_0823.csv')
    bad_IA_index=np.load('dataset/bad_IA.npy')
    HPLC_IA=HPLC_IA.drop(bad_IA_index)  #conformer error
    all_smile_IA = HPLC_IA['SMILES'].values
    T1_IA = HPLC_IA['RT'].values
    Speed_IA = HPLC_IA['Speed'].values
    Prop_IA = HPLC_IA['i-PrOH_proportion'].values
    dataset_IA=np.load('dataset/dataset_IA_charity_0823.npy',allow_pickle=True).tolist()
    index_IA=HPLC_IA['Unnamed: 0'].values

#===========Construct dataset==============

if transfer_target=='ADH':
    dataset_graph_atom_bond,dataset_graph_bond_angle,big_index = Construct_dataset(dataset_ADH,index_ADH, T1_ADH, Speed_ADH, Prop_ADH,column=transfer_target)
if transfer_target=='IC':
    dataset_graph_atom_bond,dataset_graph_bond_angle,big_index = Construct_dataset(dataset_IC,index_IC, T1_IC, Speed_IC, Prop_IC,column=transfer_target)
if transfer_target=='IA':
    dataset_graph_atom_bond,dataset_graph_bond_angle,big_index = Construct_dataset(dataset_IA,index_IA, T1_IA, Speed_IA, Prop_IA,column=transfer_target)
if transfer_target=='ODH':
    dataset_graph_atom_bond,dataset_graph_bond_angle,big_index = Construct_dataset(dataset_ODH,index_ODH, T1_ODH, Speed_ODH, Prop_ODH,column=transfer_target)


total_num = len(dataset_graph_atom_bond)
print('data num:',total_num)

train_ratio = 0.90
validate_ratio = 0.05
test_ratio = 0.05
args = parse_args()
prepartion(args)
nn_params = {
    'num_tasks': 3,
    'num_layers': args.num_layers,
    'emb_dim': args.emb_dim,
    'drop_ratio': args.drop_ratio,
    'graph_pooling': args.graph_pooling,
    'descriptor_dim': 1827
}



#given random seed
if transfer_target=='ODH':
    np.random.seed(388)
if transfer_target=='ADH':
    np.random.seed(505)
if transfer_target=='IC':
    np.random.seed(526)
if transfer_target=='IA':
    np.random.seed(388)


# automatic dataloading and splitting
data_array = np.arange(0, total_num, 1)
np.random.shuffle(data_array)
torch.random.manual_seed(525)

train_data_atom_bond = []
valid_data_atom_bond = []
test_data_atom_bond = []
train_data_bond_angle = []
valid_data_bond_angle = []
test_data_bond_angle = []

train_num = int(len(data_array) * train_ratio)
test_num = int(len(data_array) * test_ratio)
val_num = int(len(data_array) * validate_ratio)


train_index = data_array[0:train_num]
valid_index = data_array[train_num:train_num + val_num]
if test_mode == 'fixed':
    test_index = data_array[total_num-test_num:]
if test_mode=='random':
    test_index = data_array[train_num + val_num:train_num + val_num + test_num]


for i in test_index:
    test_data_atom_bond.append(dataset_graph_atom_bond[i])
    test_data_bond_angle.append(dataset_graph_bond_angle[i])
for i in valid_index:
    valid_data_atom_bond.append(dataset_graph_atom_bond[i])
    valid_data_bond_angle.append(dataset_graph_bond_angle[i])
for i in train_index:
    train_data_atom_bond.append(dataset_graph_atom_bond[i])
    train_data_bond_angle.append(dataset_graph_bond_angle[i])



train_loader_atom_bond = DataLoader(train_data_atom_bond, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
valid_loader_atom_bond = DataLoader(valid_data_atom_bond, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
test_loader_atom_bond = DataLoader(test_data_atom_bond, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
train_loader_bond_angle = DataLoader(train_data_bond_angle, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
valid_loader_bond_angle = DataLoader(valid_data_bond_angle, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
test_loader_bond_angle = DataLoader(test_data_bond_angle, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

device = args.device
criterion_fn = torch.nn.MSELoss()
model = GINGraphPooling(**nn_params).to(device)
num_params = sum(p.numel() for p in model.parameters())

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
writer = SummaryWriter(log_dir=args.save_dir)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
print('===========Data Prepared================')

if MODEL == 'Train':
    try:
        os.makedirs(f'saves/model_{transfer_target}')
    except OSError:
        pass

    for epoch in tqdm(range(1500)):
        train_mae = train(model, device, train_loader_atom_bond,train_loader_bond_angle, optimizer, criterion_fn)
        if (epoch + 1) % 100 == 0:
            valid_mae = eval(model, device, valid_loader_atom_bond,valid_loader_bond_angle)
            print(train_mae, valid_mae)
            torch.save(model.state_dict(), f'saves/model_{transfer_target}/model_save_{epoch + 1}.pth')

if MODEL == 'Test':
    if transfer_target=='ODH':
        model.load_state_dict(
            torch.load(f'saves/model_ODH_388/model_save_1500.pth'))
    if transfer_target=='ADH':
        model.load_state_dict(
            torch.load(f'saves/model_ADH_505/model_save_1500.pth'))
    if transfer_target=='IC':
        model.load_state_dict(
            torch.load(f'saves/model_IC_526/model_save_1500.pth'))
    if transfer_target=='IA':
        model.load_state_dict(
            torch.load(f'saves/model_IA_388/model_save_1500.pth'))
    y_pred, y_true, R_square, test_mae,y_pred_10,y_pred_90 = test(model, device, test_loader_atom_bond, test_loader_bond_angle)
    y_pred=y_pred.cpu().data.numpy()
    y_true = y_true.cpu().data.numpy()
    y_pred_10=y_pred_10.cpu().data.numpy()
    y_pred_90=y_pred_90.cpu().data.numpy()
    print('relative_error',np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2)))
    print('MAE',np.mean(np.abs(y_true - y_pred) / y_true))
    print('RMSE',np.sqrt(np.mean((y_true - y_pred) ** 2)))
    R_square = 1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())
    print(R_square)
    plt.figure(1,figsize=(2.5,2.5),dpi=300)
    plt.style.use('ggplot')
    plt.scatter(y_true, y_pred, c='#8983BF',s=15,alpha=0.4)
    plt.plot(np.arange(0, 60), np.arange(0, 60),linewidth=1.5,linestyle='--',color='black')
    plt.yticks(np.arange(0,66,10),np.arange(0,66,10),fontproperties='Arial', size=8)
    plt.xticks(np.arange(0,66,10),np.arange(0,66,10),fontproperties='Arial', size=8)
    plt.xlabel('Observed data', fontproperties='Arial', size=8)
    plt.ylabel('Predicted data', fontproperties='Arial', size=8)
    plt.show()

if MODEL == 'Test_enantiomer':
    transfer_target='IC'
    draw_picture=True
    smiles = ['CCCCC[C@H](F)C(=O)c1nccn1c2ccccc2','CCCCC[C@@H](F)C(=O)c1nccn1c2ccccc2']
    y_pred=[]

    speed = [1.0, 1.0]
    eluent = [0.02, 0.02]

    mols=[]
    all_descriptor=[]
    dataset=[]
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        mols.append(mol)
    from rdkit.Chem import Draw
    if draw_picture==True:
        index=0
        for mol in mols:
            smiles_pic = Draw.MolToImage(mol, size=(200, 100), kekulize=True)
            plt.imshow(smiles_pic)
            plt.axis('off')
            plt.savefig(f'fig_save/molecular_{index}.tiff')
            index+=1
            plt.clf()

    for smile in smiles:
        mol = obtain_3D_mol(smile, 'trail')
        mol = Chem.MolFromMolFile(f"trail.mol")
        all_descriptor.append(mord(mol))
        dataset.append(mol_to_geognn_graph_data_MMFF3d(mol))


    for i in range(0, len(dataset)):
        data = dataset[i]
        atom_feature = []
        bond_feature = []
        for name in atom_id_names:
            atom_feature.append(data[name])
        for name in bond_id_names:
            bond_feature.append(data[name])
        atom_feature = torch.from_numpy(np.array(atom_feature).T).to(torch.int64)
        bond_feature = torch.from_numpy(np.array(bond_feature).T).to(torch.int64)
        bond_float_feature = torch.from_numpy(data['bond_length'].astype(np.float32))
        bond_angle_feature = torch.from_numpy(data['bond_angle'].astype(np.float32))
        y = torch.Tensor([float(speed[i])])
        edge_index = torch.from_numpy(data['edges'].T).to(torch.int64)
        bond_index = torch.from_numpy(data['BondAngleGraph_edges'].T).to(torch.int64)

        prop = torch.ones([bond_feature.shape[0]]) * eluent[i]

        TPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][820] / 100
        RASA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][821]
        RPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][822]
        MDEC = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][1568]
        MATS = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][457]

        bond_feature = torch.cat([bond_feature, bond_float_feature.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, prop.reshape(-1, 1)], dim=1)

        bond_angle_feature = torch.cat([bond_angle_feature.reshape(-1, 1), TPSA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, RASA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, RPSA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, MDEC.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, MATS.reshape(-1, 1)], dim=1)


        data_atom_bond = Data(atom_feature, edge_index, bond_feature, y)
        data_bond_angle = Data(edge_index=bond_index, edge_attr=bond_angle_feature)

        if transfer_target == 'ODH':
            model.load_state_dict(
                torch.load(f'saves/model_ODH_388/model_save_1500.pth'))
        if transfer_target == 'ADH':
            model.load_state_dict(
                torch.load(f'saves/model_ADH_505/model_save_1500.pth'))
        if transfer_target == 'IC':
            model.load_state_dict(
                torch.load(f'saves/model_IC_526/model_save_1500.pth'))
        if transfer_target == 'IA':
            model.load_state_dict(
                torch.load(f'saves/model_IA_388/model_save_1500.pth'))

        model.eval()

        pred, h_graph = model(data_atom_bond.to(device),data_bond_angle.to(device))
        y_pred.append(pred.detach().cpu().data.numpy()/speed[i])
    print(y_pred)



