# data.py
# Purpose: Build PyG Data objects for Elliptic from the raw CSVs.
# Notes: This mirrors your current data-building logic with minimal changes.

from typing import Dict, Tuple
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce


# --- Constants / column names ---
N_FEATURES_ALL = 165
N_FEATURES_LOCAL = 94
N_CLASSES = 2  # binary: licit(0), illicit(1)

NODES_COLS = ['txId', 'time_step'] + [f'feature_{i}' for i in range(1, N_FEATURES_ALL + 1)]
FEATURE_COLS_ALL = [f'feature_{i}' for i in range(1, N_FEATURES_ALL + 1)]
FEATURE_COLS_LOCAL = FEATURE_COLS_ALL[:N_FEATURES_LOCAL]


def load_raw(
    nodes_path: str = "elliptic_bitcoin_dataset/elliptic_txs_features.csv",
    edges_path: str = "elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv",
    labels_path: str = "elliptic_bitcoin_dataset/elliptic_txs_classes.csv",
):
    """Load raw CSVs into dataframes and merge labels by txId."""
    nodes_df = pd.read_csv(nodes_path, header=None, names=NODES_COLS)
    edges_df = pd.read_csv(edges_path)
    labels_df = pd.read_csv(labels_path)
    # Merge labels (left join keeps all nodes; missing labels remain NaN)
    nodes_df = nodes_df.merge(labels_df, on='txId', how='left')
    return nodes_df, edges_df, labels_df


def build_index_map(nodes_df: pd.DataFrame):
    """Map txId to contiguous node indices [0..N-1] and attach 'idx' column."""
    txid_to_idx = {tx_id: i for i, tx_id in enumerate(nodes_df['txId'].values)}
    nodes_df = nodes_df.copy()
    nodes_df['idx'] = nodes_df['txId'].map(txid_to_idx)
    return nodes_df, txid_to_idx


def build_edge_index(edges_df: pd.DataFrame, txid_to_idx: dict, num_nodes: int, make_undirected: bool) -> torch.Tensor:
    """Map edgelist to node indices and build edge_index, optionally undirected."""
    df = edges_df.copy()
    df['src'] = df['txId1'].map(txid_to_idx)
    df['dst'] = df['txId2'].map(txid_to_idx)
    # Drop edges that could not be mapped
    df = df.dropna(subset=['src', 'dst'])
    edge_index = torch.tensor(df[['src', 'dst']].values.T, dtype=torch.long)

    # Make undirected if requested (bidirectional edges)
    if make_undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    # Clean up: remove self-loops and coalesce duplicates
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    return edge_index


def map_labels(nodes_df: pd.DataFrame) -> torch.Tensor:
    """Map 'class' to {-1,0,1} = {unknown, licit, illicit}."""
    df = nodes_df.copy()
    df['class'] = df['class'].astype(str)
    label_map = {'unknown': -1, '2': 0, '1': 1}
    df['class'] = df['class'].map(label_map)
    y = torch.tensor(df['class'].astype(int).values, dtype=torch.long)
    return y


def make_masks_temporal(nodes_df: pd.DataFrame, y_tensor: torch.Tensor):
    """Build train/val/test masks by timesteps (temporal split)."""
    train_steps = set(range(1, 35))   # 1..34
    val_steps = set(range(35, 42))  # 35..41
    test_steps = set(range(42, 50))  # 42..49

    labeled_mask = (y_tensor >= 0)
    train_time_t = torch.tensor(nodes_df['time_step'].isin(train_steps).values, dtype=torch.bool)
    val_time_t = torch.tensor(nodes_df['time_step'].isin(val_steps).values,   dtype=torch.bool)
    test_time_t = torch.tensor(nodes_df['time_step'].isin(test_steps).values,  dtype=torch.bool)

    train_mask = train_time_t & labeled_mask
    val_mask = val_time_t & labeled_mask
    test_mask = test_time_t & labeled_mask
    return train_mask, val_mask, test_mask


def make_masks_random(nodes_df: pd.DataFrame, y_tensor: torch.Tensor,
                      test_size: float = 0.3, val_frac_of_test: float = 0.5, seed: int = 42):
    """Build train/val/test masks by random split over labeled nodes (with stratify)."""
    labeled_idx = (y_tensor >= 0).nonzero(as_tuple=True)[0].cpu().numpy()
    y_labeled   = y_tensor[labeled_idx].cpu().numpy()

    # First split: train vs (val+test)
    train_idx, tmp_idx = train_test_split(
        labeled_idx, test_size=test_size, random_state=seed, stratify=y_labeled
    )
    # Second split: val vs test
    y_tmp = y_tensor[tmp_idx].cpu().numpy()
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=val_frac_of_test, random_state=seed, stratify=y_tmp
    )

    n = y_tensor.size(0)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[torch.tensor(train_idx, dtype=torch.long)] = True
    val_mask[torch.tensor(val_idx, dtype=torch.long)]     = True
    test_mask[torch.tensor(test_idx, dtype=torch.long)]   = True
    return train_mask, val_mask, test_mask


def make_data(x_tensor: torch.Tensor, edge_index: torch.Tensor, y_tensor: torch.Tensor, masks) -> Data:
    """Wrap tensors into a PyG Data object with given masks (and attach time_step for convenience)."""
    train_mask, val_mask, test_mask = masks
    data = Data(
        x=x_tensor, edge_index=edge_index, y=y_tensor,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask
    )
    return data


# data.py — drop-in replacement for get_variants()

def get_variants(
    *,
    graph_modes=("dag", "undirected"),
    feature_sets=("all", "local"),
    split_types=("temporal", "random"),
    device=None,
):
    """Return a dict keyed by (graph_mode, features_set, split_type).
    graph_mode in {"dag", "undirected"} controls whether we symmetrize edges.
    """
    # Load and prepare node/label tables once
    nodes_df, edges_df, _ = load_raw()
    nodes_df, txid_to_idx = build_index_map(nodes_df)
    num_nodes = len(nodes_df)

    # Labels (shared across modes)
    y = map_labels(nodes_df)

    # Features (shared across modes)
    x_all = torch.tensor(nodes_df[FEATURE_COLS_ALL].values,   dtype=torch.float)
    x_local = torch.tensor(nodes_df[FEATURE_COLS_LOCAL].values, dtype=torch.float)

    # Masks (shared across modes)
    masks_temporal = make_masks_temporal(nodes_df, y)
    masks_random   = make_masks_random(nodes_df, y, test_size=0.3, val_frac_of_test=0.5, seed=42)

    out: Dict[Tuple[str, str, str], Data] = {}

    for graph_mode in graph_modes:
        # Build edge_index per mode
        make_undirected = (graph_mode == "undirected")
        edge_index = build_edge_index(
            edges_df, txid_to_idx, num_nodes=num_nodes, make_undirected=make_undirected
        )

        # Assemble 4 variants for this graph_mode
        out[(graph_mode, 'all',   'temporal')] = make_data(x_all,   edge_index, y, masks_temporal)
        out[(graph_mode, 'local', 'temporal')] = make_data(x_local, edge_index, y, masks_temporal)
        out[(graph_mode, 'all',   'random')]   = make_data(x_all,   edge_index, y, masks_random)
        out[(graph_mode, 'local', 'random')]   = make_data(x_local, edge_index, y, masks_random)

    # Move all to device once at the end
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for k in list(out.keys()):
        out[k] = out[k].to(device)
        # attach time_step so temporal models can consume it:
        out[k].time_step = torch.tensor(nodes_df['time_step'].values, dtype=torch.long, device=device)

    return out
