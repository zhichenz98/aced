import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
            self,
            n_nodes: int,
            n_feats: int = 4,
            n_outputs: int = 4,
            hidden_dim: int = 64,
            n_layers: int = 2,
            activation: str = 'relu',
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_feats = n_feats
        self.n_outputs = n_outputs
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        input_dim = n_nodes * n_feats
        output_dim = n_nodes * n_outputs

        if activation == 'relu':
            activation = nn.ReLU()
        elif activation == 'gelu':
            activation = nn.GELU()
        elif activation == 'tanh':
            activation = nn.Tanh()
        elif activation == 'sigmoid':
            activation = nn.Sigmoid()
        else:
            raise NotImplementedError

        assert n_layers >= 2

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation)
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1)
        out = self.net(x)
        out = out.view(B, self.n_nodes, self.n_outputs)
        return out
