# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# מודל טמפורלי:
from torch_geometric_temporal.nn.recurrent import EvolveGCNO  # EvolveGCN-O

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------
# 1) טעינת Elliptic + מיפויים
# --------------------------
nodes_column_names = ['txId', 'time_step'] + [f'feature_{i}' for i in range(1, 166)]
nodes_df = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_features.csv",
                       header=None, names=nodes_column_names)
edges_df = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")
labels_df = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_classes.csv")

nodes_df = nodes_df.merge(labels_df, on='txId', how='left')

# עמודות קשתות לשמות סטנדרטיים
if {'txId1','txId2'}.issubset(edges_df.columns):
    edges_df = edges_df.rename(columns={'txId1':'src','txId2':'dst'})
elif {'src','dst'}.issubset(edges_df.columns):
    pass
else:
    # שתי העמודות הראשונות ינחשו להיות src/dst
    c0, c1 = edges_df.columns[:2]
    edges_df = edges_df.rename(columns={c0:'src', c1:'dst'})

# מוודאים טיפוסים
nodes_df['txId'] = nodes_df['txId'].astype(str)
edges_df['src'] = edges_df['src'].astype(str)
edges_df['dst'] = edges_df['dst'].astype(str)
nodes_df['time_step'] = nodes_df['time_step'].astype(int)

# מיפוי לייבלים
nodes_df['class'] = nodes_df['class'].astype(str)
label_map = {'unknown': -1, '2': 0, '1': 1}
nodes_df['y'] = nodes_df['class'].map(label_map).astype(int)

# --------------------------
# 2) הכנת snapshots לפי time_step
# --------------------------
ALL_STEPS = sorted(nodes_df['time_step'].unique().tolist())
T = len(ALL_STEPS)

# פיצול בזמן: 60% אימון, 20% ולידציה, 20% טסט (אפשר לשנות)
n_train = int(0.6 * T)
n_val   = int(0.2 * T)
train_steps = ALL_STEPS[:n_train]
val_steps   = ALL_STEPS[n_train:n_train+n_val]
test_steps  = ALL_STEPS[n_train+n_val:]

# האם לבנות גרף "תוך-צעד" בלבד (קשת src/dst באותו time_step),
# או מצטבר עד t (כולל קשתות עם max(time_src, time_dst)==t)
CUMULATIVE = False

# מיפוי צומת→צעד
tx_to_t = dict(zip(nodes_df['txId'], nodes_df['time_step']))


def build_snapshot(t):
    """בונה Data עבור צעד זמן t עם אינדקסים מקומיים (רציף מ-0..n_t-1)."""
    df_t = nodes_df[nodes_df['time_step'] == t].copy()
    df_t = df_t.reset_index(drop=True)
    if df_t.empty:
        return None

    tx_local = df_t['txId'].tolist()
    local_index = {tx:i for i, tx in enumerate(tx_local)}

    # פיצ'רים X_t
    feat_cols = [f'feature_{i}' for i in range(1, 166)]
    x_t = torch.tensor(df_t[feat_cols].values, dtype=torch.float)

    # תוויות y_t ומסכות לפי ידועים/לא ידועים
    y_t = torch.tensor(df_t['y'].values, dtype=torch.long)
    known_mask_t = (y_t >= 0)

    # קשתות באותו צעד/מצטבר
    e = edges_df.copy()
    e['t_src'] = e['src'].map(tx_to_t)
    e['t_dst'] = e['dst'].map(tx_to_t)
    if CUMULATIVE:
        use_e = e[(np.maximum(e['t_src'], e['t_dst']) == t)]
    else:
        use_e = e[(e['t_src'] == t) & (e['t_dst'] == t)]

    # סינון לקשתות שנמצאות בצמתי df_t
    use_e = use_e[use_e['src'].isin(local_index) & use_e['dst'].isin(local_index)]
    if use_e.empty:
        # גם אם אין קשתות, נבנה גרף עם צמתים בלבד
        edge_index_t = torch.empty((2,0), dtype=torch.long)
    else:
        src_local = use_e['src'].map(local_index).astype(int).values
        dst_local = use_e['dst'].map(local_index).astype(int).values
        edge_index_t = torch.as_tensor(
            np.vstack((src_local, dst_local)),  # צירוף יעיל ל־NumPy אחד
            dtype=torch.long
        )
        # GCN נהנה מגרף לא-מכוון
        edge_index_t = to_undirected(edge_index_t)

    # מסכות train/val/test לפי טווחי זמן
    train_mask = torch.zeros(len(df_t), dtype=torch.bool)
    val_mask   = torch.zeros(len(df_t), dtype=torch.bool)
    test_mask  = torch.zeros(len(df_t), dtype=torch.bool)

    if t in train_steps:
        train_mask = known_mask_t.clone()
    elif t in val_steps:
        val_mask = known_mask_t.clone()
    elif t in test_steps:
        test_mask = known_mask_t.clone()

    data_t = Data(
        x=x_t, edge_index=edge_index_t, y=y_t,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask
    )
    data_t.time_step = t  # רק לצורך הדפסות
    return data_t

snapshots = [build_snapshot(t) for t in ALL_STEPS]
snapshots = [s for s in snapshots if s is not None]

train_snaps = [s for s in snapshots if s.time_step in train_steps]
val_snaps   = [s for s in snapshots if s.time_step in val_steps]
test_snaps  = [s for s in snapshots if s.time_step in test_steps]

# --------------------------
# 3) מודל: EvolveGCN-O + ראש סיווג
# --------------------------
class EvolveGCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.recurrent = EvolveGCNO(in_channels, in_channels)  # שכבה טמפורלית
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.proj = nn.Linear(in_channels, hidden_channels)

    def forward(self, x, edge_index):
        h = self.recurrent(x, edge_index)           # H_t
        h = F.relu(self.proj(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.classifier(h)                    # logits לצמתים ב-t
        return out

# ממדים
IN_CH  = 165
HID_CH = 128
OUT_CH = 2

model = EvolveGCNClassifier(IN_CH, HID_CH, OUT_CH).to(DEVICE)

# --------------------------
# 4) פונקציות אימון/הערכה טמפורליות
# --------------------------
def class_weights_from_train(train_snaps):
    ys = []
    for s in train_snaps:
        if s.train_mask.any():
            ys.append(s.y[s.train_mask].cpu().numpy())
    if not ys:
        return None
    y_all = np.concatenate(ys)
    # 0: licit, 1: illicit
    counts = np.bincount(y_all, minlength=2).astype(float)
    # החלקת משקלים
    counts[counts == 0] = 1.0
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float, device=DEVICE)

CLS_WEIGHTS = class_weights_from_train(train_snaps)
criterion = nn.CrossEntropyLoss(weight=CLS_WEIGHTS)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

def evaluate(snaps):
    """מחזיר f1/precision/recall (מחלקת המיעוט=1) על רשימת snapshots."""
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for s in snaps:
            x = s.x.to(DEVICE)
            ei = s.edge_index.to(DEVICE)
            logits = model(x, ei)
            pred = logits.argmax(dim=1).cpu().numpy()

            if s.val_mask.any():
                mask = s.val_mask.cpu().numpy()
            elif s.test_mask.any():
                mask = s.test_mask.cpu().numpy()
            else:
                continue

            y_true_all.append(s.y.cpu().numpy()[mask])
            y_pred_all.append(pred[mask])

    if not y_true_all:
        return 0.0, 0.0, 0.0
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    f1  = f1_score(y_true, y_pred, average='binary', pos_label=1)
    pre = precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    return f1, pre, rec

# --------------------------
# 5) לולאת אימון טמפורלית
# --------------------------
EPOCHS = 50
best_val_f1 = -1.0
best_state = None
best_epoch = -1

for epoch in range(1, EPOCHS+1):
    model.train()
    optimizer.zero_grad()

    # צבירת הפסדים לאורך כל צעדי הזמן של האימון
    losses = []
    for s in train_snaps:
        if not s.train_mask.any():
            continue
        x = s.x.to(DEVICE)
        ei = s.edge_index.to(DEVICE)
        y = s.y.to(DEVICE)

        logits = model(x, ei)
        loss = criterion(logits[s.train_mask], y[s.train_mask])
        losses.append(loss)

    if not losses:
        print("No labeled nodes in training range.")
        break

    # backward אחד לכל האפוק (אפשר גם mean במקום sum)
    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    optimizer.step()

    # ולידציה על צעדי ה-val (ללא גרדיאנטים)
    val_f1, val_pre, val_rec = evaluate(val_snaps)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 5 == 0 or epoch == 1:
        print(f"[Epoch {epoch:03d}] TrainLoss={total_loss.item():.4f} | "
              f"Val F1={val_f1:.4f} (P={val_pre:.4f}, R={val_rec:.4f})")

# לשחזר את המשקלים הטובים ביותר
if best_state is not None:
    model.load_state_dict(best_state)
print(f"\nBest epoch: {best_epoch}, Best Val F1: {best_val_f1:.4f}")

# --------------------------
# 6) הערכה על test (צעדי זמן מאוחרים)
# --------------------------
model.eval()
y_true_all, y_pred_all = [], []
with torch.no_grad():
    for s in test_snaps:
        x = s.x.to(DEVICE)
        ei = s.edge_index.to(DEVICE)
        logits = model(x, ei)
        pred = logits.argmax(dim=1).cpu().numpy()
        mask = s.test_mask.cpu().numpy()
        if mask.any():
            y_true_all.append(s.y.cpu().numpy()[mask])
            y_pred_all.append(pred[mask])

if y_true_all:
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    print("\nTest metrics (aggregated over time steps):")
    print("F1 (illicit=1):", f1_score(y_true, y_pred, pos_label=1))
    print("Precision:",      precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    print("Recall:",         recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["Licit","Illicit"], zero_division=0))
else:
    print("No labeled nodes in test range – nothing to evaluate.")
