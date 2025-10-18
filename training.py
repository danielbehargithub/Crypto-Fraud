
import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import GCN, MLP


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

def run(data, model_name, features_set, split_type, graph_mode):
    device = data.x.device  # ensure model/device match the data tensors
    tag = f"{model_name.upper()} | {features_set.upper()} | {split_type.upper()} | {graph_mode.upper()}"
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
        "graph_mode": graph_mode,
        "in_channels": int(data.num_node_features),
        "best_val_f1": round(float(best_val_f1), 4),
        "test_f1_at_best": round(float(best_test_f1), 4),
        "stop_epoch": int(epoch),
        "final_lr": float(optimizer.param_groups[0]['lr']),
    }
