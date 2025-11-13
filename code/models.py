import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from collections import deque
import yaml

from torch_geometric.utils import add_self_loops, subgraph, to_undirected
from torch_geometric_temporal.nn.recurrent import EvolveGCNO
from torch_geometric.nn import GATConv, GCNConv

CFG = yaml.safe_load(open("configs/config_models.yaml"))

class MLP(nn.Module):
    """Baseline: simple 3-layer MLP for node classification (ignores edges)."""

    def __init__(self, in_channels: int,
                 hidden1: int = 128,
                 hidden2: int = 64,
                 out_channels: int = 2,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        """Apply three fully-connected layers with ReLU and dropout.
        edge_index is ignored and kept only for API compatibility with GNN models.
        """
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        return x


class GCN(torch.nn.Module):
    """Simple 2-layer GCN for node classification."""

    def __init__(self, in_channels: int,
                 hidden_channels: int = 64,
                 out_channels: int = 2,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """Apply two GCNConv layers with ReLU and dropout in between."""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # logits


class EvolveGCN(nn.Module):
    """Wrapper around EvolveGCN-O with a GCN-like interface.
    The model supports two modes:
    - Static: forward(x, edge_index, time_step=None) runs a single step over the whole graph.
    - Temporal: forward(x, edge_index, time_step) builds cumulative subgraphs per time step
      and applies the recurrent GCN over time.
    """
    requires_time = True

    def __init__(self, in_channels: int,
                 hidden_channels: int = 128,
                 out_channels: int = 2,
                 dropout: float = 0.5) -> None:
        super().__init__()
        # EvolveGCNO typically keeps the same input/output dimensionality
        self.recurrent = EvolveGCNO(in_channels, in_channels)
        self.proj = nn.Linear(in_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.out_dim = out_channels
        self.history: Dict = {}

    def _step_once(self, x_t: torch.Tensor, edge_index_t: torch.Tensor) -> torch.Tensor:
        """Apply one EvolveGCN step on a single snapshot."""
        h_t = self.recurrent(x_t, edge_index_t)
        h_t = F.relu(self.proj(h_t))
        h_t = F.dropout(h_t, p=self.dropout, training=self.training)
        logits_t = self.classifier(h_t)
        return logits_t

    def reset_history(self):
        """Reset any internal temporal state if used."""
        self.history.clear()

    def forward(self, x, edge_index, time_step=None):
        """Forward pass over static or temporal graph data."""
        device = x.device
        if time_step is None:
            return self._step_once(x, edge_index)

        time_step = time_step.to(device)
        logits_full = x.new_zeros((x.size(0), self.out_dim))
        unique_steps = torch.unique(time_step, sorted=True)

        for t in unique_steps:
            node_idx_t = (time_step == t).nonzero(as_tuple=True)[0]
            if node_idx_t.numel() == 0:
                continue

            # Cumulative window: nodes with time <= t
            node_idx_leq_t = (time_step <= t).nonzero(as_tuple=True)[0]
            sub_ei_all, _ = subgraph(node_idx_leq_t, edge_index, relabel_nodes=True)

            # Map global node indices to local indices in the cumulative subgraph
            remap = -torch.ones(x.size(0), dtype=torch.long, device=device)
            remap[node_idx_leq_t] = torch.arange(node_idx_leq_t.numel(), device=device)

            x_all = x[node_idx_leq_t]
            logits_all = self._step_once(x_all, sub_ei_all)

            # Extract logits for nodes at time t
            local_idx_t = remap[node_idx_t]
            logits_t = logits_all[local_idx_t]
            logits_full[node_idx_t] = logits_t

        return logits_full


class _StructuralEncoder(nn.Module):
    """Two GAT layers per temporal snapshot."""

    def __init__(self, in_ch: int,
                 hid_ch: int,
                 out_ch: int,
                 heads: tuple[int, int] = (4, 1),
                 dropout: float = 0.5) -> None:
        super().__init__()
        h1, h2 = heads
        self.gat1 = GATConv(in_ch, hid_ch, heads=h1, dropout=dropout, concat=True)
        self.gat2 = GATConv(hid_ch * h1, out_ch, heads=h2, dropout=dropout, concat=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """Apply two GATConv layers with ELU and dropout."""
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gat2(h, edge_index)
        return h  # [N_t, out_ch]


class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to a sequence batch [B, L, D]."""
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


class _TemporalSelfAttention(nn.Module):
    """Multi-head temporal self-attention with positional encoding."""

    def __init__(self,
                 d_model: int,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 max_len: int = 128) -> None:
        super().__init__()
        self.pe = _PositionalEncoding(d_model, max_len=max_len)
        self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, key_padding_mask=None):
        """Apply temporal self-attention on sequences."""
        x = self.pe(seq)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm(x + self.dropout(attn_out))
        x = self.norm(x + self.dropout(self.ffn(x)))
        return x[:, -1, :]  #  representation of the last time step


class DySAT(nn.Module):
    """DySAT-style model: structural GAT encoder + temporal self-attention.
    The model expects temporal node data with a time_step vector that indicates
    the snapshot index for each node.
    """

    requires_time = True

    def __init__(self, in_channels: int, struct_hidden: int = 128, struct_out: int = 128,
                 temp_heads: int = 4, temp_dropout: float = 0.1, L: int = 5, out_channels: int = 2):
        super().__init__()
        self.struct = _StructuralEncoder(in_channels, struct_hidden, struct_out, heads=(4, 4), dropout=0.5)
        self.temporal = _TemporalSelfAttention(struct_out, n_heads=temp_heads, dropout=temp_dropout, max_len=128)
        self.classifier = nn.Linear(struct_out, out_channels)
        self.L = L
        self.out_dim = out_channels
        self._banks: Dict[int, deque] = {}

    def _get_bank(self, node_idx: int, D: int):
        """Return (and create if needed) the history deque for a node."""
        if node_idx not in self._banks:
            self._banks[node_idx] = deque(maxlen=self.L - 1)
        return self._banks[node_idx]

    def _build_sequences(self, global_indices: torch.Tensor, h_t: torch.Tensor):
        """Build temporal sequences for the current snapshot."""
        N, D = h_t.size(0), h_t.size(1)
        L = self.L
        seq = h_t.new_zeros((N, L, D))
        pad = torch.ones((N, L), dtype=torch.bool, device=h_t.device)
        for i, gidx in enumerate(global_indices.tolist()):
            bank = self._get_bank(gidx, D)
            hist = list(bank)
            start = max(0, L - 1 - len(hist))
            if hist:
                hist_t = [v.to(h_t.device) for v in hist]
                seq[i, start:L - 1, :] = torch.stack(hist_t[-(L - 1):], dim=0)
                pad[i, start:L - 1] = False
            seq[i, L - 1, :] = h_t[i]
            pad[i, L - 1] = False
        return seq, pad

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, time_step: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for temporal graph data."""
        if time_step is None:
            raise RuntimeError("DySAT requires data.time_step to be provided to forward().")

        device = x.device
        time_step = time_step.to(device)
        logits_full = x.new_zeros((x.size(0), self.out_dim))
        unique_steps: List[int] = torch.unique(time_step, sorted=True).tolist()

        # Reset history at the beginning of a new run
        self._banks.clear()

        for t in unique_steps:
            node_idx = (time_step == t).nonzero(as_tuple=True)[0]
            if node_idx.numel() == 0:
                continue
            sub_ei, _ = subgraph(node_idx, edge_index, relabel_nodes=True)
            x_t = x[node_idx]

            if sub_ei.numel() == 0:
                sub_ei = sub_ei.new_empty(2, 0)
            sub_ei, _ = add_self_loops(sub_ei, num_nodes=x_t.size(0))
            sub_ei = to_undirected(sub_ei, num_nodes=x_t.size(0))
            h_t = self.struct(x_t, sub_ei)  # [N_t, D]
            seq, pad = self._build_sequences(node_idx, h_t)  # pad: [N_t, L]

            # Work around potential issues on some platforms by running attention on CPU
            seq_cpu = seq.detach().cpu().to(torch.float32).contiguous()
            pad_cpu = pad.detach().to(torch.bool).cpu()
            z_t_cpu = self.temporal(seq_cpu, key_padding_mask=pad_cpu)  # [N_t, D] on CPU
            z_t = z_t_cpu.to(h_t.device)

            logits_t = self.classifier(z_t)
            logits_full[node_idx] = logits_t

            # Update history after using the current snapshot
            for i, gidx in enumerate(node_idx.tolist()):
                bank = self._get_bank(gidx, h_t.size(1))
                bank.append(h_t[i].detach())

        return logits_full




MODEL_REGISTRY = {
    "GCN": GCN,
    "MLP": MLP,
    "EVOLVEGCN": EvolveGCN,
    "DYSAT": DySAT
}


def build_model(kind: str, in_dim: int, out_dim: int = 2):
    """Factory for building a model and its default training configuration."""
    kind = kind.upper()
    ModelClass = MODEL_REGISTRY[kind]

    model_params = CFG["models"][kind]["model"].copy()
    model_params["in_channels"] = in_dim
    model_params["out_channels"] = out_dim

    model = ModelClass(**model_params)
    train_cfg = CFG["models"][kind]["training"]

    return model, train_cfg

