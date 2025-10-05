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

# Step 1: Identify indices of labeled samples (non-unknown)
labeled_mask = y >= 0
labeled_idx = labeled_mask.nonzero(as_tuple=True)[0]

# Step 2: Split only labeled samples
train_idx, test_idx = train_test_split(labeled_idx, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

# Step 3: Create boolean masks
train_mask = torch.zeros(y.size(0), dtype=torch.bool)
val_mask = torch.zeros(y.size(0), dtype=torch.bool)
test_mask = torch.zeros(y.size(0), dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_all = data_all.to(device)
data_local = data_local.to(device)

def run_one(data, tag):
    print(f"\n===== RUN: {tag} =====")
    model = GCN(in_channels=data.num_node_features, hidden_channels=64, out_channels=2).to(device)
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

    for epoch in range(0, max_epochs):
        loss = train(model, data, optimizer, criterion)
        pred = test(model, data)

        val_true = data.y[data.val_mask].cpu().numpy()
        val_pred = pred[data.val_mask].cpu().numpy()
        val_f1 = f1_score(val_true, val_pred, average='binary', pos_label=1)

        test_true = data.y[data.test_mask].cpu().numpy()
        test_pred = pred[data.test_mask].cpu().numpy()
        test_f1 = f1_score(test_true, test_pred, average='binary', pos_label=1)

        improved = (val_f1 - best_val_f1) >= EPS
        if improved:
            best_val_f1 = val_f1
            best_test_f1 = test_f1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val F1={best_val_f1:.4f}, "
                      f"Test F1 at best Val={best_test_f1:.4f}")
                break

        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        if epoch > 0 and current_lr != prev_lr:
            print(f"🔻 LR reduced at epoch {epoch}: now {current_lr:.6f}")
        prev_lr = current_lr

        if epoch % 20 == 0 or epoch == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}')

    print(f'[{tag}] Best Val F1: {best_val_f1:.4f}, Test F1 at best Val: {best_test_f1:.4f}')

    # Final report
    final_pred = test(model, data)
    test_true = data.y[data.test_mask].cpu().numpy()
    test_pred = final_pred[data.test_mask].cpu().numpy()
    print(f"\n[{tag}] Classification Report on Test Set:")
    print(classification_report(test_true, test_pred, target_names=['Licit','Illicit']))

run_one(data_all,   tag="ALL FEATURES (local+aggregated)")
run_one(data_local, tag="LOCAL-ONLY (first 94)")

