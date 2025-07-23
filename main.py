import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

nodes_column_names = ['txId', 'time_step'] + [f'feature_{i}' for i in range(1, 166)]

# Load data
nodes_df = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_features.csv", header=None, names=nodes_column_names)
edges_df = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")
labels_df = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_classes.csv")

# Merge labels
nodes_df = nodes_df.merge(labels_df, on='txId', how='left')


txid_to_idx = {tx_id: i for i, tx_id in enumerate(nodes_df['txId'].values)}
nodes_df['idx'] = nodes_df['txId'].map(txid_to_idx)

edges_df['src'] = edges_df['txId1'].map(txid_to_idx)
edges_df['dst'] = edges_df['txId2'].map(txid_to_idx)
edge_index = torch.tensor(edges_df[['src', 'dst']].values.T, dtype=torch.long)

feature_cols = [f'feature_{i}' for i in range(1, 166)]
x = torch.tensor(nodes_df[feature_cols].values, dtype=torch.float)

# ודא שהעמודה class היא string ואז המר אותה
nodes_df['class'] = nodes_df['class'].astype(str)
label_map = {'unknown': -1, '2': 0, '1': 1}
nodes_df['class'] = nodes_df['class'].map(label_map)

# ואז תוודא שהטור הוא מספרי
y = torch.tensor(nodes_df['class'].astype(int).values, dtype=torch.long)



# שלב 1: הגדר את אינדקסים של רק הדוגמאות עם תווית ידועה (לא unknown)
labeled_mask = y >= 0
labeled_idx = labeled_mask.nonzero(as_tuple=True)[0]

# שלב 2: חלוקה רק של הדוגמאות עם תווית
train_idx, test_idx = train_test_split(labeled_idx, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

# שלב 3: יצירת המסכות
train_mask = torch.zeros(y.size(0), dtype=torch.bool)
val_mask = torch.zeros(y.size(0), dtype=torch.bool)
test_mask = torch.zeros(y.size(0), dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

data = Data(
    x=x,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask
)





import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

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


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
        accs.append(acc)
    return accs  # [train_acc, val_acc, test_acc]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

model = GCN(in_channels=data.num_node_features, hidden_channels=64, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0
for epoch in range(1, 201):
    loss = train(model, data, optimizer, criterion)
    train_acc, val_acc, test_acc = test(model, data)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

print(f'Best Val Acc: {best_val_acc:.4f}, Test Acc at best Val: {best_test_acc:.4f}')
