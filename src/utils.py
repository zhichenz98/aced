import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
from torch_geometric.nn import NNConv
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import os
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Load data
def get_path(setting, idx, sig, bias):
    if setting=='small':
        nodes=30
        b_path = 'data/data_case30/case30_topo{}_b.csv'.format(idx)
        g_path = 'data/data_case30/case30_topo{}_g.csv'.format(idx)
        branch_path = 'data/data_case30/case30_topo{}_branches.csv'.format(idx)
        feat_path = 'data/data_case30/results_case30_topo{}_sigma{:.2f}.csv'.format(idx,sig)
    elif setting=='large':
        nodes=118
        b_path = 'data/data_case118/case118_topo20_20_{}_b.csv'.format(idx)
        g_path = 'data/data_case118/case118_topo20_20_{}_g.csv'.format(idx)
        branch_path = 'data/data_case118/case118_topo20_20_{}_branches.csv'.format(idx)
        feat_path = 'data/data_case118/results_case118_topo20_20_{}_sigma{:.2f}_bias{:.2f}.csv'.format(idx,sig,bias)
    elif setting=='small_toy':
        nodes=idx
        b_path = 'data/data_toy/line{}_b.csv'.format(idx)
        g_path = 'data/data_toy/line{}_g.csv'.format(idx)
        branch_path = 'data/data_toy/line{}_branches.csv'.format(idx)
        feat_path = 'data/data_toy/results_line{}_sigma{:.2f}_bias{:.2f}.csv'.format(idx,sig,bias)
    elif setting=='small_index':
        nodes=30
        b_path = 'data/data_case30_changeindex/case30_topo{}_b.csv'.format(idx)
        g_path = 'data/data_case30_changeindex/case30_topo{}_g.csv'.format(idx)
        branch_path = 'data/data_case30_changeindex/case30_topo{}_branches.csv'.format(idx)
        feat_path = 'data/data_case30_changeindex/results_case30_topo{}_sigma{:.2f}_bias{:.2f}.csv'.format(idx,sig,bias)
    elif setting=='small_toy_0422':
        nodes=10
        b_path = 'data/data_toy_0422/line10_topo{}_b.csv'.format(idx)
        g_path = 'data/data_toy_0422/line10_topo{}_g.csv'.format(idx)
        branch_path = 'data/data_toy_0422/line10_topo{}_branches.csv'.format(idx)
        feat_path = 'data/data_toy_0422/results_line10_topo{}_sigma{:.2f}_bias{:.2f}.csv'.format(idx, sig, bias)
    elif setting=='small_toy_0422_ab':
        nodes=10
        b_path = 'data/data_toy_0422/line10_topo0_{}_b.csv'.format(idx)
        g_path = 'data/data_toy_0422/line10_topo0_{}_g.csv'.format(idx)
        branch_path = 'data/data_toy_0422/line10_topo0_{}_branches.csv'.format(idx)
        feat_path = 'data/data_toy_0422/results_line10_topo0_{}_sigma{:.2f}_bias{:.2f}.csv'.format(idx, sig, bias)
    else:
        raise ValueError('No such dataset')
    return nodes, b_path, g_path, branch_path, feat_path


def load_data(set_name='small', idx=0, sig=0.01, bias=0.00, append_type=True):
    
    train_nodes, train_b_path, train_g_path, train_branch_path, train_feat_path = get_path(set_name, idx, sig, bias)

    train_b = np.loadtxt(train_b_path, delimiter=',', dtype=np.float32)
    train_g = np.loadtxt(train_g_path, delimiter=',', dtype=np.float32)
    train_branch = np.loadtxt(train_branch_path, delimiter=',', dtype=np.int64) - 1
    train_feat = np.loadtxt(train_feat_path, delimiter=',', skiprows=1, dtype=np.float32)
    
    ## Format train and test graphs
    train_x = torch.from_numpy(train_feat[:, :train_nodes*2]).to(device)   # (100, 60)
    train_y = torch.from_numpy(train_feat[:, train_nodes*2:-1]).to(device)

    train_edge_idx = torch.from_numpy(train_branch.T).to(device).to(torch.long)
    train_edge_feat = [[train_b[i,j], train_g[i,j]] for (i,j) in train_branch]
    train_edge_feat = torch.tensor(train_edge_feat, dtype=torch.float32, device=device)
    
    ## Normalize data
    num_train_sample = train_x.shape[0]
    train_x = train_x.reshape(num_train_sample, train_nodes, -1)
    train_y = train_y.reshape(num_train_sample, train_nodes, -1)
    train_mask_idx = train_y[0,:,:].sum(dim=-1, keepdim=False).nonzero()
    
    if set_name == 'small_toy_0422_ab':
        if idx == 0:
            ab_value = [0.02, 2]
        elif idx == 1:
            ab_value = [0.01, 2]
        elif idx == 2:
            ab_value = [0.03, 2]
        elif idx == 3:
            ab_value = [0.03, 3]
        ab_emd = torch.zeros((1,train_nodes, 2)).to(device)
        ab_emd[0,5,:] = torch.tensor(ab_value)
        ab_emd = ab_emd.repeat((num_train_sample, 1, 1))
        train_x = torch.cat((train_x, ab_emd), dim=-1)
    
    if append_type:
        train_mask = torch.zeros(train_nodes)
        train_mask[train_mask_idx] = 1
        train_mask = train_mask.reshape(1, train_nodes, 1).repeat(num_train_sample, 1, 1).to(device)
        train_x = torch.cat((train_x, train_mask), dim=-1)

    train_graph = []
    for i in range(train_x.size(0)):
        train_graph.append(Data(x=train_x[i,:], edge_index=train_edge_idx, edge_attr=train_edge_feat, y=train_y[i,:]))
   
    return train_graph, train_nodes




## GNN model
class GNNRegression(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_in_channels, num_conv=1):
        super(GNNRegression, self).__init__()
        # A small network to transform edge features into weight matrices.
        self.nn_conv_in = nn.Sequential(
            nn.Linear(edge_in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, in_channels * hidden_channels),
        )
        self.num_conv = num_conv
        self.conv_layers = [NNConv(in_channels, hidden_channels, self.nn_conv_in, aggr='mean')]
        # self.bn_layers = [nn.BatchNorm1d(hidden_channels)]
        for _ in range(num_conv-1):
            self.nn_conv_hidden = nn.Sequential(
                nn.Linear(edge_in_channels, 64),
                nn.ReLU(),
                nn.Linear(64, hidden_channels * hidden_channels),
            )
            self.conv_layers.append(NNConv(hidden_channels, hidden_channels, self.nn_conv_hidden, aggr='mean'))
            # self.bn_layers.append(nn.BatchNorm1d(hidden_channels))
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        # self.bn_layers = torch.nn.ModuleList(self.bn_layers)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i in range(self.num_conv):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            # x = self.bn_layers[i](x)
            x = torch.relu(x)
        out = self.fc(x)
        return out
    


def compute_covariance(input_data):
        """
        Compute Covariance matrix of the input data
        """
        n = input_data.size(0)  # batch_size
        id_row = torch.ones(n).resize(1, n).to(device=input_data.device)
        sum_column = torch.mm(id_row, input_data)
        mean_column = torch.div(sum_column, n)
        term_mul_2 = torch.mm(mean_column.t(), mean_column)
        d_t_d = torch.mm(input_data.t(), input_data)
        c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)
        return c
    
def coral_loss(source, target):
    d = source.shape[1]  # dim vector
    source_c = compute_covariance(source)
    target_c = compute_covariance(target)
    domain_loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))
    domain_loss = domain_loss / (4 * d * d)
    return domain_loss


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

def mmd_loss(source, target):
    mind = min(source.shape[0], target.shape[0])
    domain_loss = mmd_rbf_noaccelerate(source[:mind], target[:mind])
    return domain_loss