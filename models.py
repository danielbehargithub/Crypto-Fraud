# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Define a simple 2-layer GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # logits

# baseline MLP- no graph data
class MLP(nn.Module):
    """Simple 3-layer MLP for node classification (ignores edges)."""
    def __init__(self, in_channels, hidden1=128, hidden2=64, out_channels=2, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None):  # edge_index ignored for compatibility
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        return x  # logits

def build_model(kind: str, in_dim: int, out_dim: int = 2):
    """Return a model instance based on kind ('GCN' or 'MLP')."""
    k = kind.upper()
    if k == "GCN":
        return GCN(in_channels=in_dim, hidden_channels=64, out_channels=out_dim)
    if k == "MLP":
        return MLP(in_channels=in_dim, hidden1=128, hidden2=64, out_channels=out_dim)
    raise ValueError(f"Unknown model kind: {kind}")
