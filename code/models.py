import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from collections import deque
import yaml
from pathlib import Path
import math
from torch import Tensor
from torch_geometric.utils import subgraph
from torch_geometric.utils import add_self_loops, subgraph, to_undirected
from torch_geometric_temporal.nn.recurrent import EvolveGCNO
from torch_geometric.nn import GATConv, GCNConv

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(ROOT / "configs" /"config_models.yaml"))

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


# ====== GRCU / mat-GRU as in EvolveGCN-O (slightly adapted) ======


class MatGRUGate(nn.Module):
    def __init__(self, rows: int, cols: int, activation: nn.Module):
        super().__init__()
        self.activation = activation

        # As in egcn_o: W, U have shape [rows, rows], bias has shape [rows, cols]
        self.W = nn.Parameter(torch.Tensor(rows, rows))
        self.U = nn.Parameter(torch.Tensor(rows, rows))
        self.bias = nn.Parameter(torch.zeros(rows, cols))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.U.data.uniform_(-stdv, stdv)
        # bias is already initialized to zeros

    def forward(self, x: Tensor, hidden: Tensor) -> Tensor:
        # x, hidden: [rows, cols]
        out = self.activation(self.W @ x + self.U @ hidden + self.bias)
        return out


class MatGRUCell(nn.Module):
    """Evolves the GCN weight matrix Q_t over time.

    This is almost identical to mat_GRU_cell in egcn_o.py, but without TopK
    (we simply use prev_Q directly).
    """

    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.update = MatGRUGate(in_feats, out_feats, activation=nn.Sigmoid())
        self.reset_gate = MatGRUGate(in_feats, out_feats, activation=nn.Sigmoid())
        self.htilda = MatGRUGate(in_feats, out_feats, activation=nn.Tanh())

    def forward(self, prev_Q: Tensor) -> Tensor:
        # In the original egcn_o they optionally apply TopK on embeddings,
        # but in the file you provided this is disabled and they just use prev_Q.
        z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset_gate(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1.0 - update) * prev_Q + update * h_cap
        return new_Q


class GRCU(nn.Module):
    """
    Gated Recurrent Convolution Unit.

    At each time step t:
      1. Evolve a new GCN weight matrix Q_t through MatGRUCell.
      2. Apply a GCN-style update: H_t = σ(Â_t X_t Q_t).
    """

    def __init__(self, in_feats: int, out_feats: int, activation=F.relu):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.activation = activation

        self.evolve_weights = MatGRUCell(in_feats, out_feats)
        self.GCN_init_weights = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.GCN_init_weights.size(1))
        self.GCN_init_weights.data.uniform_(-stdv, stdv)

    def forward(self, A_list: List[Tensor], X_list: List[Tensor]) -> List[Tensor]:
        """
        Parameters
        ----------
        A_list : list of Tensors
            Each element is a [N_t, N_t] normalized adjacency matrix Â_t.
        X_list : list of Tensors
            Each element is a [N_t, in_feats] node feature matrix X_t.

        Returns
        -------
        out_seq : list of Tensors
            Each element is H_t of shape [N_t, out_feats] for each time step t.
        """
        Q = self.GCN_init_weights  # initial GCN weight matrix
        out_seq: List[Tensor] = []

        for A_t, X_t in zip(A_list, X_list):
            # evolve weights
            Q = self.evolve_weights(Q)  # [in_feats, out_feats]
            # apply GCN step: σ(Â_t X_t Q_t)
            # X_t: [N_t, in_feats]  -> X_t @ Q: [N_t, out_feats]
            H_t = X_t @ Q
            H_t = A_t @ H_t  # A_t: [N_t, N_t]
            H_t = self.activation(H_t)
            out_seq.append(H_t)

        return out_seq


# ====== EvolveGCN wrapper adapted to your (x, edge_index, time_step) API ======


class EvolveGCN(nn.Module):
    """EvolveGCN-O based on egcn_o.py, adapted to PyG + time_step.

    Expected input (per full temporal graph):
        x         : [num_nodes, in_channels]
        edge_index: [2, num_edges]
        time_step : [num_nodes] (discrete index for each node)

    Returns logits for all nodes across all time steps.
    """

    requires_time = True

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # As in EvolveGCN-O: two stacked GRCU layers
        self.grcu1 = GRCU(in_feats=in_channels, out_feats=hidden_channels, activation=F.relu)
        self.grcu2 = GRCU(in_feats=hidden_channels, out_feats=hidden_channels, activation=F.relu)

        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.out_dim = out_channels

    @staticmethod
    def _build_normalized_adj(edge_index_t: Tensor, num_nodes_t: int, device) -> Tensor:
        """
        Build a dense normalized adjacency matrix Â_t ∈ R^{N_t x N_t} with self-loops,
        using symmetric normalization: Â = D^{-1/2} (A + I) D^{-1/2}.

        This mimics a standard GCN normalization for a single snapshot.
        """
        if num_nodes_t == 0:
            return torch.empty((0, 0), device=device)

        A = torch.zeros((num_nodes_t, num_nodes_t), device=device)
        if edge_index_t.numel() > 0:
            src, dst = edge_index_t
            A[src, dst] = 1.0
            A[dst, src] = 1.0  # make the graph undirected, as in many EvolveGCN experiments

        # add self-loops
        A.fill_diagonal_(1.0)

        deg = A.sum(dim=1)  # [N_t]
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)

        A_hat = D_inv_sqrt @ A @ D_inv_sqrt  # [N_t, N_t]
        return A_hat

    def reset_history(self):
        """Provided for compatibility with your train() function.

        This model does not keep an explicit external temporal state:
        each forward pass starts from GCN_init_weights inside GRCU.
        """
        # No external temporal state to reset.
        pass

    def forward(self, x: Tensor, edge_index: Tensor, time_step: Tensor = None) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Node features of shape [N, in_channels].
        edge_index : LongTensor
            Edge index in COO format of shape [2, E].
        time_step : LongTensor, optional
            Time-step index for each node, shape [N]. If None, all nodes
            are treated as belonging to a single snapshot.

        Returns
        -------
        logits_full : Tensor
            Logits for all nodes, shape [N, out_dim].
        """
        device = x.device
        if time_step is None:
            # If forward is called without time_step, treat the whole graph as one snapshot.
            time_step = torch.zeros(x.size(0), dtype=torch.long, device=device)

        time_step = time_step.to(device)
        unique_steps = torch.unique(time_step, sorted=True)

        # Build A_list and X_list for each snapshot, and keep mapping from time to node indices.
        A_list: List[Tensor] = []
        X_list: List[Tensor] = []
        node_indices_per_t: List[Tensor] = []

        for t in unique_steps:
            node_idx_t = (time_step == t).nonzero(as_tuple=True)[0]
            if node_idx_t.numel() == 0:
                continue

            node_indices_per_t.append(node_idx_t)

            # Subgraph of nodes at time t only
            sub_ei_t, _ = subgraph(node_idx_t, edge_index, relabel_nodes=True)
            N_t = node_idx_t.numel()
            A_t = self._build_normalized_adj(sub_ei_t, N_t, device=device)  # [N_t, N_t]
            X_t = x[node_idx_t]  # [N_t, in_channels]

            A_list.append(A_t)
            X_list.append(X_t)

        # If there are no nodes (edge case), return zeros.
        if len(A_list) == 0:
            return x.new_zeros((x.size(0), self.out_dim))

        # Two GRCU layers in sequence, as in the original EGCN implementation.
        H_list = self.grcu1(A_list, X_list)      # list: H_t^1
        H_list = self.grcu2(A_list, H_list)      # list: H_t^2

        # Compute logits for all nodes at each time step.
        logits_full = x.new_zeros((x.size(0), self.out_dim))
        for node_idx_t, H_t in zip(node_indices_per_t, H_list):
            H_t = F.dropout(H_t, p=self.dropout, training=self.training)
            logits_t = self.classifier(H_t)
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
        
    def reset_history(self):
        self._banks.clear()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, time_step: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for temporal graph data."""
        if time_step is None:
            raise RuntimeError("DySAT requires data.time_step to be provided to forward().")

        device = x.device
        time_step = time_step.to(device)
        logits_full = x.new_zeros((x.size(0), self.out_dim))
        unique_steps: List[int] = torch.unique(time_step, sorted=True).tolist()

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
            # seq_cpu = seq.detach().cpu().to(torch.float32).contiguous()
            # pad_cpu = pad.detach().to(torch.bool).cpu()
            # z_t_cpu = self.temporal(seq_cpu, key_padding_mask=pad_cpu)  # [N_t, D] on CPU
            # z_t = z_t_cpu.to(h_t.device)

            z_t = self.temporal(seq, key_padding_mask=pad)

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

