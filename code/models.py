# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, subgraph
from typing import List, Dict
from torch_geometric_temporal.nn.recurrent import EvolveGCNO
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import add_self_loops, to_undirected
from collections import deque


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



class EvolveGCN(nn.Module):
    """
    EvolveGCN-O עטוף להתממשקות כמו GCN:
    - forward(x, edge_index, time_step=None)
    - אם time_step סופק: נבנה תת-גרף לכל צעד זמן ונריץ רצף EvolveGCN לאורך t.
    - אם time_step=None -> נריץ צעד יחיד על כל הגרף (fallback סטטי).
    """
    requires_time = True

    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 2, dropout: float = 0.5):
        super().__init__()
        # שימי לב: ברוב המימושים של EvolveGCNO המימד שווה ל-in_channels → נוסיף הקרנה
        self.recurrent = EvolveGCNO(in_channels, in_channels)
        self.proj = nn.Linear(in_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.out_dim = out_channels

    def _step_once(self, x_t: torch.Tensor, edge_index_t: torch.Tensor) -> torch.Tensor:
        h_t = self.recurrent(x_t, edge_index_t)
        h_t = F.relu(self.proj(h_t))
        h_t = F.dropout(h_t, p=self.dropout, training=self.training)
        logits_t = self.classifier(h_t)
        return logits_t

    def reset_history(self):
        # אם יש לך מבנים כמו dict/deque – אפסי כאן. אם אין – השאירי pass.
        self.history = {}  # או self.history.clear()

    def forward(self, x, edge_index, time_step=None):
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

            # --- CUMULATIVE WINDOW: nodes with time <= t ---
            node_idx_leq_t = (time_step <= t).nonzero(as_tuple=True)[0]
            sub_ei_all, _ = subgraph(node_idx_leq_t, edge_index, relabel_nodes=True)
            # מיפוי חזרה לאינדקסים של צמתי t בתוך הגרף המצטבר
            remap = -torch.ones(x.size(0), dtype=torch.long, device=device)
            remap[node_idx_leq_t] = torch.arange(node_idx_leq_t.numel(), device=device)

            x_all = x[node_idx_leq_t]
            logits_all = self._step_once(x_all, sub_ei_all)

            # נחלץ את הלוגיטים של הצמתים בזמן t
            local_idx_t = remap[node_idx_t]
            logits_t = logits_all[local_idx_t]
            logits_full[node_idx_t] = logits_t

        return logits_full


# ===== DySAT: GAT (מבני) + multihead temporal self-attention (טמפורלי) =====

class _StructuralEncoder(nn.Module):
    """שתי שכבות GAT לכל snapshot."""
    def __init__(self, in_ch, hid_ch, out_ch, heads=(4, 1), dropout=0.5):
        super().__init__()
        h1, h2 = heads
        self.gat1 = GATConv(in_ch, hid_ch, heads=h1, dropout=dropout, concat=True)
        self.gat2 = GATConv(hid_ch*h1, out_ch, heads=h2, dropout=dropout, concat=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gat2(h, edge_index)
        return h  # [N_t, out_ch]

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)

class _TemporalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1, max_len=128):
        super().__init__()
        self.pe = _PositionalEncoding(d_model, max_len=max_len)
        self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, key_padding_mask=None):
        x = self.pe(seq)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm(x + self.dropout(attn_out))
        x = self.norm(x + self.dropout(self.ffn(x)))
        return x[:, -1, :]   # הווקטור של הזמן האחרון

class DySAT(nn.Module):
    """
    DySAT עטוף להתממשקות כמו GCN:
    - forward(x, edge_index, time_step=None)
    - דורש time_step כדי לבנות היסטוריה לכל צומת לאורך L צעדים.
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
        if node_idx not in self._banks:
            self._banks[node_idx] = deque(maxlen=self.L - 1)
        return self._banks[node_idx]

    def _build_sequences(self, global_indices: torch.Tensor, h_t: torch.Tensor):
        """
        global_indices: אינדקסים גלובליים של צמתי הצעד הנוכחי (סדר = h_t)
        h_t: [N_t, D]
        מחזיר: seq [N_t, L, D], pad_mask [N_t, L] (True=pad)
        """
        N, D = h_t.size(0), h_t.size(1)
        L = self.L
        seq = h_t.new_zeros((N, L, D))
        pad = torch.ones((N, L), dtype=torch.bool, device=h_t.device)
        for i, gidx in enumerate(global_indices.tolist()):
            bank = self._get_bank(gidx, D)
            hist = list(bank)
            start = max(0, L-1 - len(hist))
            if hist:
                hist_t = [v.to(h_t.device) for v in hist]
                seq[i, start:L - 1, :] = torch.stack(hist_t[-(L - 1):], dim=0)
                pad[i, start:L - 1] = False
            seq[i, L-1, :] = h_t[i]
            pad[i, L-1] = False
        return seq, pad

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, time_step: torch.Tensor = None) -> torch.Tensor:
        if time_step is None:
            raise RuntimeError("DySAT requires data.time_step to be provided to forward().")

        device = x.device
        time_step = time_step.to(device)
        logits_full = x.new_zeros((x.size(0), self.out_dim))
        unique_steps: List[int] = torch.unique(time_step, sorted=True).tolist()

        # Empty history in the beginning of step
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
            h_t = self.struct(x_t, sub_ei)               # [N_t, D]
            seq, pad = self._build_sequences(node_idx, h_t)  # pad: [N_t, L]

            # --- SAFETY on Windows: run temporal attention on CPU ---
            seq_cpu = seq.detach().cpu().to(torch.float32).contiguous()
            pad_cpu = pad.detach().to(torch.bool).cpu()
            z_t_cpu = self.temporal(seq_cpu, key_padding_mask=pad_cpu)  # [N_t, D] on CPU
            z_t = z_t_cpu.to(h_t.device)

            logits_t = self.classifier(z_t)
            logits_full[node_idx] = logits_t

            # History update after each move
            for i, gidx in enumerate(node_idx.tolist()):
                bank = self._get_bank(gidx, h_t.size(1))
                bank.append(h_t[i].detach())

        return logits_full


def build_model(kind: str, in_dim: int, out_dim: int = 2):
    kind = kind.upper()
    if kind == "GCN":
        model = GCN(in_channels=in_dim, hidden_channels=64, out_channels=out_dim)
        train_cfg = {"lr": 2e-2, "weight_decay": 5e-4,
                     "warmup_start": 0, "scheduler_warmup": True}   # GCN: no warmup for ES, scheduler active
    elif kind == "MLP":
        model = MLP(in_channels=in_dim, hidden1=128, hidden2=64, out_channels=out_dim)
        train_cfg = {"lr": 2e-1, "weight_decay": 5e-4,
                     "warmup_start": 5, "scheduler_warmup": False} # MLP: delay ES/scheduler
    elif kind == "EVOLVEGCN":
        model = EvolveGCN(in_dim, out_dim)
        train_cfg = {"lr": 1e-3, "weight_decay": 5e-4,
                     "warmup_start": 10, "scheduler_warmup": True}
    else:
        model = DySAT(in_dim, struct_hidden=128, struct_out=128, out_channels=out_dim)
        train_cfg = {"lr": 5e-4, "weight_decay": 1e-4,
                     "warmup_start": 10, "scheduler_warmup": True}
    return model, train_cfg
