import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
from torch_geometric.nn import NNConv
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import os
from utils import *
warnings.filterwarnings("ignore")


def main(train_set='small', test_set='large', train_idx=0, test_idx=5, train_sig=0.01, test_sig=0.01, train_bias=0.00, test_bias=0.00, apply_da=True, lambda_da=1, num_conv=1):
    # train_graph, test_graph, train_nodes, test_nodes = load_data(train_set, test_set, train_idx, test_idx, train_sig, test_sig, train_bias, test_bias)
    train_graph, train_nodes = load_data(train_set, train_idx, train_sig, train_bias)
    test_graph, test_nodes = load_data(test_set, test_idx, test_sig, test_bias)
    in_channels = train_graph[0].x.size(-1) 
    hidden_channels = 32
    out_channels = train_graph[0].y.size(-1)
    edge_in_channels = train_graph[0].edge_attr.size(-1)

    train_loader = DataLoader(train_graph, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_graph, batch_size=64, shuffle=True)

    # Setup device, model, optimizer, and loss function.
    model = GNNRegression(in_channels, hidden_channels, out_channels, edge_in_channels, num_conv).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Train
    model.train()
    for epoch in range(1, num_epoch+1):
        train_out = []
        train_y = []
        test_iter = iter(test_loader)
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            out = out * batch.x[:,-1].reshape(-1,1)
            mse_loss = criterion(out, batch.y)
            loss = mse_loss
            if apply_da:
                try:
                    test_batch = next(test_iter).to(device)
                except StopIteration:
                    test_iter = iter(test_loader)
                    test_batch = next(test_iter).to(device)
                test_out = model(test_batch)
                test_out = test_out * test_batch.x[:,-1].reshape(-1,1)
                da_loss = mmd_loss(out, test_out, kernel='rbf', sigma=1.0)
                loss += lambda_da * da_loss
            loss.backward()
            optimizer.step()
            train_out.append(out.detach())
            train_y.append(batch.y)
        train_out, train_y = torch.cat(train_out, dim=0).cpu(), torch.cat(train_y, dim=0).cpu()
        train_R2 = r2_score(train_out, train_y)
        train_mse = mean_squared_error(train_out, train_y)
        # if epoch % 50 == 0:
        #     print('Epoch {}: mse {:.2f}, R2 {:.2f}, MAPE {:.2f}%'.format(epoch+1, train_loss, train_R2, train_MAPE))
    # np.savetxt('train_gnd.csv', train_y, fmt='%.2f', delimiter=',')
    # np.savetxt('train_pred.csv', train_out, fmt='%.2f', delimiter=',')
    

    # Test
    test_out = []
    test_y = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            out = out * batch.x[:,-1].reshape(-1,1)
            test_out.append(out.detach())
            test_y.append(batch.y)
    test_out, test_y = torch.cat(test_out, dim=0).cpu(), torch.cat(test_y, dim=0).cpu()
    test_R2 = r2_score(test_out, test_y)
    test_mse = mean_squared_error(test_out, test_y)

    # ## for debug
    # np.savetxt('test_gnd.csv', test_y, fmt='%.2f', delimiter=',')
    # np.savetxt('test_pred.csv', test_out, fmt='%.2f', delimiter=',')
    
    print('({}, {}, {}, {}) to ({}, {}, {}, {}):\nTrain MSE {:.4f}, R2: {:.4f}\nTest MSE {:.4f}, R2: {:.4f}\n'.format(train_nodes, train_idx, train_sig, train_bias, test_nodes, test_idx, test_sig, test_bias, train_mse, train_R2, test_mse, test_R2))
    
    return [train_mse, train_R2], [test_mse, test_R2]

##### Test Cases (Small -> Large, Large -> Small)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epoch = 500

train_set = 'small'
test_set = 'large'
train_nodes = 30
test_nodes = 118
train_idx_list = [0,1,2]
test_idx_list = [5]
train_sig_list = [0.01, 0.02]
test_sig_list = [0.01]
train_bias_list = [0.00]
test_bias_list = [0.00]
apply_da = False
lambda_da = 300
num_conv=1

# train_set = 'large'
# test_set = 'small'
# train_nodes = 118
# test_nodes = 30
# train_idx_list = [5]
# test_idx_list = [0]
# train_sig_list = [0.01, 0.03, 0.05]
# test_sig_list = [0.01]
# train_bias_list = [0.00, 0.01]
# test_bias_list = [0.00]
# apply_da = True
# lambda_da = 300
# num_conv=2

# train_set = 'small'
# test_set = 'small'
# train_nodes = 30
# test_nodes = 30
# train_idx_list = [0,1,2]
# test_idx_list = [0]
# train_sig_list = [0.01, 0.02]
# test_sig_list = [0.01]
# train_bias_list = [0.00]
# test_bias_list = [0.00]
# apply_da = False
# lambda_da=0
# num_conv=1

# train_set = 'large'
# test_set = 'large'
# train_nodes = 118
# test_nodes = 118
# train_idx_list = [5,10,26,33]
# test_idx_list = [5]
# train_sig_list = [0.01, 0.03, 0.05]
# test_sig_list = [0.01]
# train_bias_list = [0.00, 0.01]
# test_bias_list = [0.00]
# apply_da = False
# lambda_da=0
# num_conv=2

if not os.path.exists('result/{}2{}/'.format(train_set, test_set)):
    os.mkdir('result/{}2{}/'.format(train_set, test_set))
for test_idx in test_idx_list:
    for test_sig in test_sig_list:
        for test_bias in test_bias_list:
            metric = [['source', 'train_mse', 'train_R2', 'test_mse', 'test_R2']]
            for train_idx in train_idx_list:
                for train_sig in train_sig_list:
                    for train_bias in train_bias_list:
                        tmp_train_metric, tmp_test_metric = main(train_set, test_set, train_idx, test_idx, train_sig, test_sig, train_bias, test_bias, apply_da=apply_da, lambda_da=lambda_da, num_conv=num_conv)
                        metric.append(['{:d}-{:.2f}'.format(train_idx, train_sig)] + ['{:.2f}'.format(ele) for ele in tmp_train_metric] + ['{:.2f}'.format(ele) for ele in tmp_test_metric])
            if apply_da:
                np.savetxt('result/{}2{}/{:d}_{:.2f}_{:.2f}_da.csv'.format(train_set, test_set, test_nodes, test_idx, test_sig, test_bias), metric, fmt='%s', delimiter=',')
            else:
                np.savetxt('result/{}2{}/{:d}_{:.2f}_{:.2f}.csv'.format(train_set, test_set, test_nodes, test_idx, test_sig, test_bias), metric, fmt='%s', delimiter=',')








####### Test Case (Index Transfer)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# num_epoch = 500
# train_set = 'small_toy'
# test_set = 'small_toy'
# train_idx_list = [10,15,20]
# test_idx_list = [25]
# test_nodes = 25
# train_sig_list = [0.01, 0.03, 0.05]
# test_sig_list = [0.01]
# train_bias_list = [0.00]
# test_bias_list = [0.00]
# apply_da = True
# lambda_da = 1
# num_conv=1


# if not os.path.exists('result/{}2{}/'.format(train_set, test_set)):
#     os.mkdir('result/{}2{}/'.format(train_set, test_set))
# for test_idx in test_idx_list:
#     for test_sig in test_sig_list:
#         for test_bias in test_bias_list:
#             metric = [['source', 'train_mse', 'train_R2', 'test_mse', 'test_R2']]
#             for train_idx in train_idx_list:
#                 for train_sig in train_sig_list:
#                     for train_bias in train_bias_list:
#                         tmp_train_metric, tmp_test_metric = main(train_set, test_set, train_idx, test_idx, train_sig, test_sig, train_bias, test_bias, apply_da=apply_da, lambda_da=lambda_da, num_conv=num_conv)
#                         metric.append(['{:d}-{:.2f}'.format(train_idx, train_sig)] + ['{:.2f}'.format(ele) for ele in tmp_train_metric] + ['{:.2f}'.format(ele) for ele in tmp_test_metric])
#             if apply_da:
#                 np.savetxt('result/{}2{}/{:d}_{:.2f}_{:.2f}_da.csv'.format(train_set, test_set, test_nodes, test_idx, test_sig, test_bias), metric, fmt='%s', delimiter=',')
#             else:
#                 np.savetxt('result/{}2{}/{:d}_{:.2f}_{:.2f}.csv'.format(train_set, test_set, test_nodes, test_idx, test_sig, test_bias), metric, fmt='%s', delimiter=',')









