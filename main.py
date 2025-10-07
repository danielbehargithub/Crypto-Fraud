import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

nodes_column_names = ['txId', 'time_step'] + [f'feature_{i}' for i in range(1, 166)]

# Load data
nodes_df = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_features.csv", header=None, names=nodes_column_names)
edges_df = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")
labels_df = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_classes.csv")

# Merge labels
nodes_df = nodes_df.merge(labels_df, on='txId', how='left')

# Map transaction IDs to numeric node indices
txid_to_idx = {tx_id: i for i, tx_id in enumerate(nodes_df['txId'].values)}
nodes_df['idx'] = nodes_df['txId'].map(txid_to_idx)

# Map edge endpoints to node indices
edges_df['src'] = edges_df['txId1'].map(txid_to_idx)
edges_df['dst'] = edges_df['txId2'].map(txid_to_idx)
edge_index = torch.tensor(edges_df[['src', 'dst']].values.T, dtype=torch.long)

# --- Feature subsets for two runs ---
feature_cols = [f'feature_{i}' for i in range(1, 166)]
feature_cols_all = feature_cols
feature_cols_local = feature_cols[:94]

x_all = torch.tensor(nodes_df[feature_cols_all].values, dtype=torch.float)
x_local = torch.tensor(nodes_df[feature_cols_local].values, dtype=torch.float)

# Ensure the "class" column is string, then map to integers
nodes_df['class'] = nodes_df['class'].astype(str)
label_map = {'unknown': -1, '2': 0, '1': 1}
nodes_df['class'] = nodes_df['class'].map(label_map)

# Convert label column to tensor
y = torch.tensor(nodes_df['class'].astype(int).values, dtype=torch.long)
labeled_mask_t = (y >= 0)  # torch.bool

# We'll use first 34 timesteps for training, next 7 for validation, last 8 for test
train_steps = set(range(1, 35))
val_steps   = set(range(35, 42))
test_steps  = set(range(42, 50))

# Make boolean masks based on time_step and labeled samples
train_time_t = torch.tensor(nodes_df['time_step'].isin(train_steps).values, dtype=torch.bool)
val_time_t   = torch.tensor(nodes_df['time_step'].isin(val_steps).values,   dtype=torch.bool)
test_time_t  = torch.tensor(nodes_df['time_step'].isin(test_steps).values,  dtype=torch.bool)

# Convert to torch tensors
train_mask = train_time_t & labeled_mask_t
val_mask   = val_time_t   & labeled_mask_t
test_mask  = test_time_t  & labeled_mask_t


# Combine all into a single PyG Data object
data_all = Data(
    x=x_all,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask
)

data_local = Data(
    x=x_local,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask
)

# Define a simple 2-layer GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Training function
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    return pred

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
        return x


# ==== Splitting helpers ====

def make_masks_temporal(nodes_df, y_tensor):
    """Build train/val/test masks by timesteps (temporal split)."""
    train_steps = set(range(1, 35))
    val_steps   = set(range(35, 42))
    test_steps  = set(range(42, 50))

    labeled_mask = (y_tensor >= 0)
    train_time_t = torch.tensor(nodes_df['time_step'].isin(train_steps).values, dtype=torch.bool)
    val_time_t   = torch.tensor(nodes_df['time_step'].isin(val_steps).values,   dtype=torch.bool)
    test_time_t  = torch.tensor(nodes_df['time_step'].isin(test_steps).values,  dtype=torch.bool)

    train_mask = train_time_t & labeled_mask
    val_mask   = val_time_t   & labeled_mask
    test_mask  = test_time_t  & labeled_mask
    return train_mask, val_mask, test_mask


def make_masks_random(nodes_df, y_tensor, test_size=0.3, val_frac_of_test=0.5, seed=42):
    """Build train/val/test masks by random split over labeled nodes (with stratify)."""
    labeled_idx = (y_tensor >= 0).nonzero(as_tuple=True)[0].cpu().numpy()
    y_labeled   = y_tensor[labeled_idx].cpu().numpy()

    # First split: train vs (val+test)
    train_idx, tmp_idx = train_test_split(
        labeled_idx, test_size=test_size, random_state=seed, stratify=y_labeled
    )
    # Second split: val vs test (50/50 of the remainder by default)
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


def make_data(x_tensor, edge_index, y_tensor, masks):
    """Wrap tensors into a PyG Data object with given masks."""
    train_mask, val_mask, test_mask = masks
    return Data(
        x=x_tensor, edge_index=edge_index, y=y_tensor,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask
    )

# ==== Build both split variants ====
masks_temporal = make_masks_temporal(nodes_df, y)
masks_random   = make_masks_random(nodes_df, y, test_size=0.3, val_frac_of_test=0.5, seed=42)

data_variants = {
    # (features_set, split_type) -> Data
    ('all',   'temporal'): make_data(torch.tensor(nodes_df[feature_cols_all].values, dtype=torch.float),  edge_index, y, masks_temporal),
    ('local', 'temporal'): make_data(torch.tensor(nodes_df[feature_cols_local].values, dtype=torch.float), edge_index, y, masks_temporal),
    ('all',   'random'):   make_data(torch.tensor(nodes_df[feature_cols_all].values, dtype=torch.float),  edge_index, y, masks_random),
    ('local', 'random'):   make_data(torch.tensor(nodes_df[feature_cols_local].values, dtype=torch.float), edge_index, y, masks_random),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Send to device
for key in list(data_variants.keys()):
    data_variants[key] = data_variants[key].to(device)

# ==== Runner: unchanged logic, but returns richer summary ====
def run(data, model_name, features_set, split_type):
    tag = f"{model_name.upper()} | {features_set.upper()} | {split_type.upper()}"
    print(f"\n===== RUN: {tag} =====")
    if model_name.upper() == 'GCN':
        model = GCN(in_channels=data.num_node_features, hidden_channels=64, out_channels=2).to(device)
        warmup_start = 10   # patience warmup start for GCN
        scheduler_warmup = True   # scheduler active immediately for GCN
    else:
        model = MLP(in_channels=data.num_node_features, hidden1=128, hidden2=64, out_channels=2).to(device)
        warmup_start = 50   # patience warmup start for MLP
        scheduler_warmup = False  # delay scheduler until warmup

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    EPS = 0.0002
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=0.002)

    best_val_f1 = -1.0
    best_test_f1 = 0.0
    patience = 30
    no_improve = 0
    max_epochs = 1000
    prev_lr = optimizer.param_groups[0]['lr']

    for epoch in range(max_epochs):
        loss = train(model, data, optimizer, criterion)
        pred = test(model, data)

        val_true = data.y[data.val_mask].detach().cpu().numpy()
        val_pred = pred[data.val_mask].detach().cpu().numpy()
        val_f1 = f1_score(val_true, val_pred, average='binary', pos_label=1)

        test_true = data.y[data.test_mask].detach().cpu().numpy()
        test_pred = pred[data.test_mask].detach().cpu().numpy()
        test_f1 = f1_score(test_true, test_pred, average='binary', pos_label=1)

        improved = (val_f1 - best_val_f1) >= EPS
        if improved:
            best_val_f1 = val_f1
            best_test_f1 = test_f1
            no_improve = 0
        else:
            if epoch >= warmup_start:
                no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val F1={best_val_f1:.4f}, Test F1 at best Val={best_test_f1:.4f}")
                break

        # LR scheduler: immediate for GCN, delayed for MLP
        if scheduler_warmup or epoch >= warmup_start:
            scheduler.step(val_f1)

        current_lr = optimizer.param_groups[0]['lr']
        if epoch > 0 and current_lr != prev_lr:
            print(f"🔻 LR reduced at epoch {epoch}: now {current_lr:.6f}")
        prev_lr = current_lr

        if epoch % 20 == 0 or epoch == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}")

    # Final test report (optional print)
    final_pred = test(model, data)
    test_true = data.y[data.test_mask].detach().cpu().numpy()
    test_pred = final_pred[data.test_mask].detach().cpu().numpy()
    print(f"\n[{tag}] Classification Report on Test Set:")
    print(classification_report(test_true, test_pred, target_names=['Licit','Illicit']))

    # Rich summary row
    return {
        "model": model_name.upper(),
        "features_set": features_set,     # 'local' or 'all'
        "split_type": split_type,         # 'temporal' or 'random'
        "in_channels": int(data.num_node_features),
        "best_val_f1": round(float(best_val_f1), 4),
        "test_f1_at_best": round(float(best_test_f1), 4),
        "stop_epoch": int(epoch),
        "final_lr": float(optimizer.param_groups[0]['lr']),
    }

# ==== Full grid of runs: 2 models × 2 feature sets × 2 splits ====
summaries = []
for model_name in ['GCN', 'MLP']:
    for features_set in ['all', 'local']:
        for split_type in ['temporal', 'random']:
            data_obj = data_variants[(features_set, split_type)]
            summaries.append(run(data_obj, model_name, features_set, split_type))

df = pd.DataFrame(summaries)[
    ["model", "features_set", "split_type", "in_channels", "best_val_f1", "test_f1_at_best", "stop_epoch", "final_lr"]
]

print("\n=== Summary Table (all runs) ===")
print(df.to_string(index=False))
df.to_csv("run_summary_all.csv", index=False)
print("Saved: run_summary_all.csv")






# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data_all = data_all.to(device)
# data_local = data_local.to(device)
#
# def run(data, tag, model_name='GCN'):
#     print(f"\n===== RUN: {tag} =====")
#     if model_name == 'GCN':
#         model = GCN(in_channels=data.num_node_features, hidden_channels=64, out_channels=2).to(device)
#         warmup_start = 10  # patience counter starts only after this epoch
#         scheduler_warmup = True
#     else:
#         model = MLP(in_channels=data.num_node_features, hidden1=128, hidden2=64, out_channels=2).to(device)
#         warmup_start = 50  # needs more time
#         scheduler_warmup = False
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
#     criterion = nn.CrossEntropyLoss()
#
#     EPS = 0.0002
#     scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=0.002)
#     best_val_f1 = -1.0
#     best_test_f1 = 0.0
#     patience = 30
#     no_improve = 0
#     max_epochs = 1000
#     prev_lr = optimizer.param_groups[0]['lr']
#
#     for epoch in range(0, max_epochs):
#         loss = train(model, data, optimizer, criterion)
#         pred = test(model, data)
#
#         val_true = data.y[data.val_mask].cpu().numpy()
#         val_pred = pred[data.val_mask].cpu().numpy()
#         val_f1 = f1_score(val_true, val_pred, average='binary', pos_label=1)
#
#         test_true = data.y[data.test_mask].cpu().numpy()
#         test_pred = pred[data.test_mask].cpu().numpy()
#         test_f1 = f1_score(test_true, test_pred, average='binary', pos_label=1)
#
#         improved = (val_f1 - best_val_f1) >= EPS
#         if improved:
#             best_val_f1 = val_f1
#             best_test_f1 = test_f1
#             no_improve = 0
#         else:
#             if epoch >= warmup_start:
#                 no_improve += 1
#             if no_improve >= patience:
#                 print(f"Early stopping at epoch {epoch}. Best Val F1={best_val_f1:.4f}, "
#                       f"Test F1 at best Val={best_test_f1:.4f}")
#                 break
#
#         if scheduler_warmup or epoch >= warmup_start:
#             scheduler.step(val_f1)
#         current_lr = optimizer.param_groups[0]['lr']
#         if epoch > 0 and current_lr != prev_lr:
#             print(f"🔻 LR reduced at epoch {epoch}: now {current_lr:.6f}")
#         prev_lr = current_lr
#
#         if epoch % 20 == 0 or epoch == 0:
#             print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}')
#
#
#     print(f'[{tag}] Best Val F1: {best_val_f1:.4f}, Test F1 at best Val: {best_test_f1:.4f}')
#
#     summary = {
#         "tag": tag,
#         "model": model_name.upper(),
#         "in_channels": data.num_node_features,
#         "best_val_f1": round(best_val_f1, 4),
#         "test_f1_at_best": round(best_test_f1, 4),
#         "stop_epoch": epoch,
#         "final_lr": optimizer.param_groups[0]['lr'],
#     }
#
#     # Final report
#     final_pred = test(model, data)
#     test_true = data.y[data.test_mask].cpu().numpy()
#     test_pred = final_pred[data.test_mask].cpu().numpy()
#     print(f"\n[{tag}] Classification Report on Test Set:")
#     print(classification_report(test_true, test_pred, target_names=['Licit','Illicit']))
#
#     return summary
#
# summaries = []
# summaries.append(run(data_all,   tag="GCN: ALL FEATURES (local+aggregated)", model_name='GCN'))
# summaries.append(run(data_local, tag="GCN: LOCAL-ONLY (first 94)",           model_name='GCN'))
# summaries.append(run(data_all,   tag="MLP: ALL FEATURES (local+aggregated)", model_name='MLP'))
# summaries.append(run(data_local, tag="MLP: LOCAL-ONLY (first 94)",           model_name='MLP'))
#
# df = pd.DataFrame(summaries)[
#     ["model", "tag", "in_channels", "best_val_f1", "test_f1_at_best", "stop_epoch", "final_lr"]
# ]
# print("\n=== Summary Table ===")
# print(df.to_string(index=False))
# df.to_csv("run_summary_temporal.csv", index=False)
