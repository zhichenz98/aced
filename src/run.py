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

###### Test Cases (Samll -> Large, Large -> Small)

# train_set = 'small'
# test_set = 'large'
# train_nodes = 30
# test_nodes = 118
# train_idx_list = [0,1,2]
# test_idx_list = [5]
# train_sig_list = [0.01, 0.02]
# test_sig_list = [0.01]
# train_bias_list = [0.00]
# test_bias_list = [0.00]
# apply_da = False
# lambda_da = 300
# num_conv=1

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








####### Test Case (Index Transfer)
# ### INDEX TRANSFER

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






# ####### Test Case (Multi -> One, Index Transfer)
def index_main(train_idx=[1,2,3,4], test_idx=[5], train_sig=[0.01,0.03, 0.05], test_sig=[0.01,0.03,0.05], train_bias=[0.00, 0.01], test_bias=[0.00, 0.01], lambda_da=1, num_conv=1, da_type=None, apply_mask=False):
    train_cases = [[idx, sig, bias] for idx in train_idx for sig in train_sig for bias in train_bias]
    test_cases = [[idx, sig, bias] for idx in test_idx for sig in test_sig for bias in test_bias]
    train_graphs, test_graphs = [], []
    for ele in train_cases:
        tmp_train_graph, tmp_train_nodes = load_data('small_toy_0422', ele[0], ele[1], ele[2])
        train_graphs += tmp_train_graph
    for ele in test_cases:
        tmp_test_graph, tmp_test_nodes = load_data('small_toy_0422', ele[0], ele[1], ele[2])
        test_graphs += tmp_test_graph
    in_channels = train_graphs[0].x.size(-1) 
    hidden_channels = 32
    out_channels = train_graphs[0].y.size(-1)
    edge_in_channels = train_graphs[0].edge_attr.size(-1)

    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=True)

    # Setup device, model, optimizer, and loss function.
    model = GNNRegression(in_channels, hidden_channels, out_channels, edge_in_channels, num_conv).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Train
    model.train()
    for epoch in range(1, num_epoch+1):
        train_out = []
        train_y = []
        train_mask = []
        train_loss = 0
        test_iter = iter(test_loader)
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            if apply_mask:
                out = out * batch.x[:,-1].reshape(-1,1)
            train_mask.append(batch.x[:,-1].reshape(-1,1))
            mse_loss = criterion(out, batch.y)
            loss = mse_loss
            if da_type is not None:
                try:
                    test_batch = next(test_iter).to(device)
                except StopIteration:
                    test_iter = iter(test_loader)
                    test_batch = next(test_iter).to(device)
                test_out = model(test_batch)
                if apply_mask:
                    test_out = test_out * test_batch.x[:,-1].reshape(-1,1)
                if da_type == 'mmd':
                    da_loss = mmd_loss(out, test_out)
                elif da_type == 'coral':
                    da_loss = coral_loss(out, test_out)
                else:
                    raise ValueError("Invalid domain adaptation taype")
                loss += lambda_da * da_loss
            loss.backward()
            optimizer.step()
            train_out.append(out.detach())
            train_y.append(batch.y)
        train_out, train_y, train_mask = torch.cat(train_out, dim=0).cpu(), torch.cat(train_y, dim=0).cpu(), torch.cat(train_mask, dim=0).cpu()
        train_R2 = r2_score(train_out * train_mask, train_y)
        train_mse = mean_squared_error(train_out * train_mask, train_y)
        # if epoch % 50 == 0:
        #     print('Epoch {}: mse {:.2f}, R2 {:.2f}, MAPE {:.2f}%'.format(epoch+1, train_loss, train_R2, train_MAPE))
    np.savetxt('train_gnd.csv', train_y, fmt='%.2f', delimiter=',')
    np.savetxt('train_pred.csv', train_out, fmt='%.2f', delimiter=',')
    

    # Test
    test_out = []
    test_y = []
    test_mask = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            if apply_mask:
                out = out * batch.x[:,-1].reshape(-1,1)
            test_mask.append(batch.x[:,-1].reshape(-1,1))
            test_out.append(out.detach())
            test_y.append(batch.y)
    test_out, test_y, test_mask = torch.cat(test_out, dim=0).cpu(), torch.cat(test_y, dim=0).cpu(), torch.cat(test_mask, dim=0).cpu()
    test_R2 = r2_score(test_out * test_mask, test_y)
    test_mse = mean_squared_error(test_out * test_mask, test_y)
    np.savetxt('test_gnd.csv', test_y, fmt='%.2f', delimiter=',')
    np.savetxt('test_pred.csv', test_out, fmt='%.2f', delimiter=',')
    
    print('({}, {}, {}) to ({}, {}, {}):\nTrain MSE {:.4f}, R2: {:.4f}\nTest MSE {:.4f}, R2: {:.4f}\n'.format(train_idx, train_sig, train_bias, test_idx, test_sig, test_bias, train_mse, train_R2, test_mse, test_R2))
    
    return [train_mse, train_R2], [test_mse, test_R2]
    
### INDEX TRANSFER

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epoch = 500
apply_mask = True
train_set = 'small_toy_0422'
test_set = 'small_toy_0422'
train_sig = [0.01, 0.03, 0.05]
test_sig = [0.01, 0.03, 0.05]
train_bias = [0.00, 0.01]
test_bias = [0.00, 0.01]
# train_sig = [0.01]
# test_sig = [0.01]
# train_bias = [0.00]
# test_bias = [0.00]
lambda_da = 0.1
num_conv=3
da_type = 'coral'
metric = [['topo_idx', 'train_mse', 'train_R2', 'test_mse', 'test_R2']]
for test_idx in range(6):
    train_idx = [i for i in range(test_idx)] + [i for i in range(test_idx+1, 6)]
    tmp_train_metric, tmp_test_metric = index_main(train_idx=train_idx, test_idx=[test_idx], train_sig=train_sig, test_sig=test_sig, train_bias=train_bias, test_bias=test_bias, lambda_da=lambda_da, num_conv=num_conv, da_type=da_type, apply_mask=apply_mask)
    metric.append(['{}'.format(test_idx)] + ['{:.2f}'.format(ele) for ele in tmp_train_metric] + ['{:.2f}'.format(ele) for ele in tmp_test_metric])
    if apply_mask:
        if da_type is not None:
            np.savetxt('result/small_toy_0422/{}_{:.0e}_mask.csv'.format(da_type, float(lambda_da)), metric, fmt='%s', delimiter=',')
        else:
            np.savetxt('result/small_toy_0422/erm_mask.csv'.format(), metric, fmt='%s', delimiter=',')
    else:
        if da_type is not None:
            np.savetxt('result/small_toy_0422/{}_{:.0e}.csv'.format(da_type, float(lambda_da)), metric, fmt='%s', delimiter=',')
        else:
            np.savetxt('result/small_toy_0422/erm.csv'.format(), metric, fmt='%s', delimiter=',')




# ###### COST EFFICIENT
# ####### Test Case (Multi -> One, Index Transfer)
# def index_ab_main(train_idx=[1,2,3,4], test_idx=[5], train_sig=[0.01,0.03, 0.05], test_sig=[0.01,0.03,0.05], train_bias=[0.00, 0.01], test_bias=[0.00, 0.01], lambda_da=1, num_conv=1, da_type=None, apply_mask=False):
#     train_cases = [[idx, sig, bias] for idx in train_idx for sig in train_sig for bias in train_bias]
#     test_cases = [[idx, sig, bias] for idx in test_idx for sig in test_sig for bias in test_bias]
#     train_graphs, test_graphs = [], []
#     for ele in train_cases:
#         tmp_train_graph, tmp_train_nodes = load_data('small_toy_0422_ab', ele[0], ele[1], ele[2])
#         train_graphs += tmp_train_graph
#     for ele in test_cases:
#         tmp_test_graph, tmp_test_nodes = load_data('small_toy_0422_ab', ele[0], ele[1], ele[2])
#         test_graphs += tmp_test_graph
#     in_channels = train_graphs[0].x.size(-1) 
#     hidden_channels = 32
#     out_channels = train_graphs[0].y.size(-1)
#     edge_in_channels = train_graphs[0].edge_attr.size(-1)

#     train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
#     test_loader = DataLoader(test_graphs, batch_size=64, shuffle=True)

#     # Setup device, model, optimizer, and loss function.
#     model = GNNRegression(in_channels, hidden_channels, out_channels, edge_in_channels, num_conv).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
#     criterion = nn.MSELoss()

#     # Train
#     model.train()
#     for epoch in range(1, num_epoch+1):
#         train_out = []
#         train_y = []
#         train_mask = []
#         train_loss = 0
#         test_iter = iter(test_loader)
#         for batch in train_loader:
#             batch = batch.to(device)
#             optimizer.zero_grad()
#             out = model(batch)
#             if apply_mask:
#                 out = out * batch.x[:,-1].reshape(-1,1)
#             train_mask.append(batch.x[:,-1].reshape(-1,1))
#             mse_loss = criterion(out, batch.y)
#             loss = mse_loss
#             if da_type is not None:
#                 try:
#                     test_batch = next(test_iter).to(device)
#                 except StopIteration:
#                     test_iter = iter(test_loader)
#                     test_batch = next(test_iter).to(device)
#                 test_out = model(test_batch)
#                 if apply_mask:
#                     test_out = test_out * test_batch.x[:,-1].reshape(-1,1)
#                 if da_type == 'mmd':
#                     da_loss = mmd_loss(out, test_out)
#                 elif da_type == 'coral':
#                     da_loss = coral_loss(out, test_out)
#                 else:
#                     raise ValueError("Invalid domain adaptation taype")
#                 loss += lambda_da * da_loss
#             loss.backward()
#             optimizer.step()
#             train_out.append(out.detach())
#             train_y.append(batch.y)
#         train_out, train_y, train_mask = torch.cat(train_out, dim=0).cpu(), torch.cat(train_y, dim=0).cpu(), torch.cat(train_mask, dim=0).cpu()
#         train_R2 = r2_score(train_out * train_mask, train_y)
#         train_mse = mean_squared_error(train_out * train_mask, train_y)
#         # if epoch % 50 == 0:
#         #     print('Epoch {}: mse {:.2f}, R2 {:.2f}, MAPE {:.2f}%'.format(epoch+1, train_loss, train_R2, train_MAPE))
#     np.savetxt('train_gnd.csv', train_y, fmt='%.2f', delimiter=',')
#     np.savetxt('train_pred.csv', train_out, fmt='%.2f', delimiter=',')
    

#     # Test
#     test_out = []
#     test_y = []
#     test_mask = []
#     model.eval()
#     with torch.no_grad():
#         for batch in test_loader:
#             out = model(batch)
#             if apply_mask:
#                 out = out * batch.x[:,-1].reshape(-1,1)
#             test_mask.append(batch.x[:,-1].reshape(-1,1))
#             test_out.append(out.detach())
#             test_y.append(batch.y)
#     test_out, test_y, test_mask = torch.cat(test_out, dim=0).cpu(), torch.cat(test_y, dim=0).cpu(), torch.cat(test_mask, dim=0).cpu()
#     test_R2 = r2_score(test_out * test_mask, test_y)
#     test_mse = mean_squared_error(test_out * test_mask, test_y)
#     np.savetxt('test_gnd.csv', test_y, fmt='%.2f', delimiter=',')
#     np.savetxt('test_pred.csv', test_out, fmt='%.2f', delimiter=',')
    
#     print('({}, {}, {}) to ({}, {}, {}):\nTrain MSE {:.4f}, R2: {:.4f}\nTest MSE {:.4f}, R2: {:.4f}\n'.format(train_idx, train_sig, train_bias, test_idx, test_sig, test_bias, train_mse, train_R2, test_mse, test_R2))
    
#     return [train_mse, train_R2], [test_mse, test_R2]
    
# ### INDEX TRANSFER

# device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
# num_epoch = 500
# apply_mask = True
# train_set = 'small_toy_0422_ab'
# test_set = 'small_toy_0422_ab'
# train_sig = [0.01, 0.03, 0.05]
# test_sig = [0.01, 0.03, 0.05]
# train_bias = [0.00, 0.01]
# test_bias = [0.00, 0.01]
# # train_sig = [0.01]
# # test_sig = [0.01]
# # train_bias = [0.00]
# # test_bias = [0.00]
# lambda_da = 0.1
# num_conv=3
# da_type = None
# metric = [['topo_idx', 'train_mse', 'train_R2', 'test_mse', 'test_R2']]
# for test_idx in range(1,4):
#     train_idx = [i for i in range(1,test_idx)] + [i for i in range(test_idx+1, 4)]
#     tmp_train_metric, tmp_test_metric = index_ab_main(train_idx=train_idx, test_idx=[test_idx], train_sig=train_sig, test_sig=test_sig, train_bias=train_bias, test_bias=test_bias, lambda_da=lambda_da, num_conv=num_conv, da_type=da_type, apply_mask=apply_mask)
#     metric.append(['{}'.format(test_idx)] + ['{:.2f}'.format(ele) for ele in tmp_train_metric] + ['{:.2f}'.format(ele) for ele in tmp_test_metric])
#     if apply_mask:
#         if da_type is not None:
#             np.savetxt('result/small_toy_0422_ab/{}_{:.0e}_mask.csv'.format(da_type, float(lambda_da)), metric, fmt='%s', delimiter=',')
#         else:
#             np.savetxt('result/small_toy_0422_ab/erm_mask.csv'.format(), metric, fmt='%s', delimiter=',')
#     else:
#         if da_type is not None:
#             np.savetxt('result/small_toy_0422_ab/{}_{:.0e}.csv'.format(da_type, float(lambda_da)), metric, fmt='%s', delimiter=',')
#         else:
#             np.savetxt('result/small_toy_0422_ab/erm.csv'.format(), metric, fmt='%s', delimiter=',')