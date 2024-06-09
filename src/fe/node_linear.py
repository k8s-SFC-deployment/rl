import torch
import torch.nn as nn
from gymnasium import spaces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NodeLinearFE(nn.Module):
    def __init__(self, observation_space: spaces.Dict, hidden_dims=[], state_dim=16):
        super(NodeLinearFE, self).__init__()
        self.observation_space = observation_space
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.lins = nn.Sequential()
        dims = [observation_space["node_vertex_attr"].shape[0]] + hidden_dims + [state_dim]
        for i in range(1, len(dims)):
            self.lins.add_module(f'linear{i}', nn.Linear(dims[i-1], dims[i]))
            self.lins.add_module(f'relu{i}', nn.ReLU())

    def forward(self, x):
        x = x["node_vertex_attr"].to(device)
        return self.lins(x)