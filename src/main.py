import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm

from model.MLP import MLP
from loss.OurKKTLoss_MVA import OurKKTLoss
from utils.data import load_env_mat, load_csv
from metric.r2 import r2_by_metric_by_bus, r2_by_metric


def load_and_preprocess(env, args):
    n_bus = env['n_bus']
    df = load_csv(args.data_path, n_bus)

    P_d_columns = [f'p_d_{i + 1}' for i in range(n_bus)]
    Q_d_columns = [f'q_d_{i + 1}' for i in range(n_bus)]

    # P_g_columns = [f'p_g_{i + 1}' for i in range(n_bus)]
    # Q_g_columns = [f'q_g_{i + 1}' for i in range(n_bus)]

    V_r_columns = [f'v_r_{i + 1}' for i in range(n_bus)]
    V_i_columns = [f'v_i_{i + 1}' for i in range(n_bus)]

    feat_columns = P_d_columns + Q_d_columns
    label_columns = V_r_columns + V_i_columns

    all_columns = feat_columns + label_columns

    df = df[all_columns].copy()  # only use the subset

    df_tr, df_val = train_test_split(df, train_size=args.train_ratio, shuffle=False)

    # TODO: This normalization has NO GENERALIZATION at all. remember to change it.
    if args.normalize_by == 'base':
        mean = np.concatenate([
            env['P_d_base'],
            env['Q_d_base'],
            env['V_r_base'],
            env['V_i_base']
        ], axis=0)
        std = None

        df_tr = df_tr - mean
        df_val = df_val - mean

    elif args.normalize_by == 'mean':
        mean = df_tr.mean()
        std = None

        df_tr = df_tr - mean
        df_val = df_val - mean

    elif args.normalize_by == 'mean_std':

        mean = df_tr.mean()
        std = df_tr.std().replace(0, 1)

        df_tr = (df_tr - mean) / std
        df_val = (df_val - mean) / std

    else:
        mean = None
        std = None

    print(df_tr.describe())

    # All with shape (batch size, number of buses, number of metrics)
    X_tr = np.stack([df_tr[P_d_columns].values, df_tr[Q_d_columns].values], axis=2)
    X_val = np.stack([df_val[P_d_columns].values, df_val[Q_d_columns].values], axis=2)
    Y_tr = np.stack([df_tr[V_r_columns].values, df_tr[V_i_columns].values], axis=2)
    Y_val = np.stack([df_val[V_r_columns].values, df_val[V_i_columns].values], axis=2)

    dataset_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    dataset_val = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))

    return dataset_tr, dataset_val


def test_kkt_loss(args):
    # load env
    env = load_env_mat(mat_path=args.env_path, device=args.device, dtype=args.dtype)
    if 'n_bus' not in env:
        env['n_bus'] = len(env['V_max'])

    # load data, build dataloader
    dataset_tr, dataset_val = load_and_preprocess(env, args)
    dataloader_tr = DataLoader(dataset_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    unsupervised_loss_func = OurKKTLoss(env=env)

    for i, (X, Y) in enumerate(tqdm(dataloader_val)):
        X, Y = X.to(device=args.device, dtype=args.dtype), Y.to(device=args.device, dtype=args.dtype)

        V_r, V_i, P_g, Q_g = torch.unbind(Y, dim=2)  # (bs, n_node, 4) to 4 * (bs, n_node)
        P_d, Q_d = torch.unbind(X, dim=2)  # (bs, n_node, 2) to 2 * (bs, n_node)

        unsup_loss = unsupervised_loss_func(V_r, V_i, P_g, Q_g, P_d, Q_d).mean()

        print(unsup_loss)

    exit()


def main(args):
    # load env
    env = load_env_mat(mat_path=args.env_path, device=args.device, dtype=args.dtype)
    if 'n_bus' not in env:
        env['n_bus'] = len(env['V_max'])

    # load data, build dataloader
    dataset_tr, dataset_val = load_and_preprocess(env, args)
    dataloader_tr = DataLoader(dataset_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.model == 'mlp':
        model = MLP(n_nodes=env['n_bus'], n_feats=2, n_outputs=2, hidden_dim=64, n_layers=2, activation='gelu')
    elif args.model == 'mean':
        model = Mean(n_nodes=env['n_bus'], n_outputs=2)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw_wo_decay':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')

    if args.supervised_loss in ['mae', 'l1']:
        supervised_loss_func = nn.L1Loss()
    elif args.supervised_loss in ['mse', 'l2']:
        supervised_loss_func = nn.MSELoss()
    else:
        raise ValueError(f'Unknown supervised loss: {args.supervised_loss}')

    if args.unsupervised_loss == 'our_kkt_mva':
        unsupervised_loss_func = OurKKTLoss(env=env)
        args.use_unsupervised_loss = True
    elif args.unsupervised_loss == 'none':
        args.use_unsupervised_loss = False

    else:
        raise ValueError(f'Unknown unsupervised loss: {args.unsupervised_loss}')

    if args.metric == 'r2_by_metric':
        metric_func = r2_by_metric
    elif args.metric == 'r2_by_metric_by_bus':
        metric_func = r2_by_metric_by_bus
    else:
        raise ValueError(f'Unknown metric: {args.metric}')

    # if we just use training set mean
    all_Y = dataset_val.tensors[1]
    all_Y_pred = dataset_tr.tensors[1].mean(dim=0).expand(all_Y.size(0), -1, -1)
    tqdm.write('Naive Prediction')
    tqdm.write(
        f'MSE: {mean_squared_error(all_Y.reshape(-1), all_Y_pred.reshape(-1))}; {args.metric}: {metric_func(all_Y_pred, all_Y)}')

    # Train and Val
    for epoch in range(args.epochs):

        # Train
        for i, (X, Y) in enumerate(tqdm(dataloader_tr)):
            X, Y = X.to(device=args.device, dtype=args.dtype), Y.to(device=args.device, dtype=args.dtype)

            Y_pred = model(X)
            # print(Y_pred.shape, Y.shape)
            sup_loss = supervised_loss_func(Y_pred, Y)

            # print(sup_loss.shape)

            if args.use_unsupervised_loss and epoch > 100:
                V_r, V_i, P_g, Q_g = torch.unbind(Y_pred, dim=2)  # (bs, n_node, 4) to 4 * (bs, n_node)
                P_d, Q_d = torch.unbind(X, dim=2)  # (bs, n_node, 2) to 2 * (bs, n_node)

                unsup_loss = unsupervised_loss_func(V_r, V_i, P_g, Q_g, P_d, Q_d).mean()
                # print(unsup_loss.shape)

                if i == 0:
                    print(sup_loss.item(), unsup_loss.item())
                loss = sup_loss + unsup_loss

            else:
                loss = sup_loss

            # print(loss.shape)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Val
        if (epoch + 1) % args.eval_every_k_steps == 0:
            all_Y_pred = []
            all_Y = []

            with torch.no_grad():
                for i, (X, Y) in enumerate(tqdm(dataloader_val)):
                    X, Y = X.to(device=args.device, dtype=args.dtype), Y.to(device=args.device, dtype=args.dtype)
                    Y_pred = model(X)
                    all_Y_pred.append(Y_pred)
                    all_Y.append(Y)

            all_Y_pred = torch.cat(all_Y_pred, dim=0).cpu().numpy()
            all_Y = torch.cat(all_Y).cpu().numpy()

            # all_Y_pred = np.broadcast_to(all_Y.mean(axis=0), all_Y.shape)

            tqdm.write(f'Epoch {epoch + 1}/{args.epochs}')
            tqdm.write(
                f'MSE: {mean_squared_error(all_Y.reshape(-1), all_Y_pred.reshape(-1))}; {args.metric}: {metric_func(all_Y_pred, all_Y)}')


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_path', type=str, default='../data/case39/case39_env.mat')

    parser.add_argument('--data_path', type=str, default='../data/case39/results_case39_sigma0.01_bias0.00.csv')

    parser.add_argument('--normalize_by', type=str, default='base', choices=['base', 'mean'])

    parser.add_argument('--train_ratio', type=float, default=0.5)

    parser.add_argument('--model', type=str, default='mlp')

    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--optimizer', type=str, default='adamw')

    parser.add_argument('--supervised_loss', type=str, default='mse')

    parser.add_argument('--unsupervised_loss', type=str, default='our_kkt_mva')

    parser.add_argument('--metric', type=str, default='r2_by_metric')

    parser.add_argument('--eval_every_k_steps', type=int, default=10)

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--cuda', action='store_true', default=False,
                        help='whether use cuda to train')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for dataloader')

    # parser.add_argument('--dtype', type=str, default='float32')

    parser.add_argument('--num_threads', type=int, default=4,
                        help='Number of threads')

    args = parser.parse_args()

    args.env_path = os.path.expanduser(args.env_path)

    args.data_path = os.path.expanduser(args.data_path)

    args.device = torch.device('cuda') if torch.cuda.is_available() and args.cuda else torch.device('cpu')

    args.dtype = torch.float32  # TODO: do we need bfloat16?

    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = args_parser()
    setup_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    main(args)
    # test_kkt_loss(args)
