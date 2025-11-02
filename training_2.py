
import torch
from sklearn.metrics import classification_report, precision_score, recall_score, average_precision_score, f1_score
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models_2 import build_model
import numpy as np
import torch.nn.functional as F


def _class_weights_from_train(data):
    y = data.y[data.train_mask]
    counts = torch.bincount(y, minlength=2).float()
    counts[counts == 0] = 1.0
    w = counts.sum() / (2.0 * counts)
    return w.to(data.x.device)


# Helpers that pass time_step if the model requires it
def _forward(model, data):
    if getattr(model, "requires_time", False):
        if not hasattr(data, "time_step"):
            raise RuntimeError("Model requires data.time_step but it is missing on Data.")
        return model(data.x, data.edge_index, data.time_step)
    return model(data.x, data.edge_index)


def _proba_pos(model, data):
    """מחזיר הסתברות למחלקת המיעוט (class=1) לכל הצמתים."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = _forward(model, data)          # logits: [N,2]
        proba = F.softmax(out, dim=1)[:, 1]  # p(y=1)
    if was_training:
        model.train()
    return proba.detach().cpu().numpy()

def predict_with_threshold(model, data, threshold: float):
    """החזרת חיזוי בינארי לפי סף."""
    p = _proba_pos(model, data)
    return (p >= threshold).astype(np.int64)


def find_best_threshold_on_val(model, data, thresholds=None):
    """
    סריקה על ספים ובחירת הסף שממקסם F1(Illicit) על ה-validation.
    מחזיר (best_thresh, best_f1_val).
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    val_mask = data.val_mask.detach().cpu().numpy().astype(bool)
    y_true = data.y.detach().cpu().numpy()[val_mask]
    proba = _proba_pos(model, data)[val_mask]

    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (proba >= t).astype(np.int64)
        f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# Training function
def train(model, data, optimizer, criterion):
    model.train()

    # ניתוק/איפוס מצב טמפורלי בין אפוקים
    if hasattr(model, "reset_history"):
        model.reset_history()

    optimizer.zero_grad()
    out = _forward(model, data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


# Evaluation function
@torch.no_grad()
def test(model, data):
    model.eval()
    out = _forward(model, data)
    pred = out.argmax(dim=1)
    return pred

def epoch_loop(model, data, optimizer, criterion, scheduler, *,
               EPS: float = 2e-4, lr, wd, warmup_start: int = 0, patience: int = 30, max_epochs: int = 1000,
               scheduler_warmup: bool = True,
    ):
    best_val_f1 = -1.0
    best_test_f1 = 0.0
    prev_lr = optimizer.param_groups[0]['lr']
    no_improve = 0

    for epoch in range(max_epochs):
        loss = train(model, data, optimizer, criterion)

        with torch.no_grad():
            best_t, val_f1  = find_best_threshold_on_val(model, data)
            proba_all = _proba_pos(model, data)

        test_mask = data.test_mask.detach().cpu().numpy().astype(bool)
        test_true = data.y.detach().cpu().numpy()[test_mask]
        test_proba = proba_all[test_mask]
        test_f1 = f1_score(test_true, (test_proba >= best_t).astype(int), pos_label=1)

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

        if scheduler_warmup or epoch >= warmup_start:
            scheduler.step(val_f1)

        current_lr = optimizer.param_groups[0]['lr']
        if epoch > 0 and current_lr != prev_lr:
            print(f"🔻 LR reduced at epoch {epoch}: now {current_lr:.6f}")
        prev_lr = current_lr

        if epoch % 20 == 0 or epoch == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}")

    return {
        "best_val_f1": round(float(best_val_f1), 4),
        "best_test_f1": round(float(best_test_f1), 4),
        "stop_epoch": int(epoch),
        "final_lr": float(optimizer.param_groups[0]['lr']),
    }



def run(data, model_name, features_set, split_type, graph_mode):
    device = data.x.device  # ensure model/device match the data tensors
    tag = f"{model_name.upper()} | {features_set.upper()} | {split_type.upper()} | {graph_mode.upper()}"
    print(f"\n===== RUN: {tag} =====")

    # --- Sanity: no time leakage ---
    ts = data.time_step.detach().cpu().numpy()
    train_ts = ts[data.train_mask.cpu().numpy()]
    val_ts = ts[data.val_mask.cpu().numpy()]
    test_ts = ts[data.test_mask.cpu().numpy()]
    assert train_ts.max() < val_ts.min(), "Time leakage: train overlaps/after val"
    assert val_ts.max() < test_ts.min(), "Time leakage: val overlaps/after test"

    # build any model by name:
    model, cfg = build_model(model_name, in_dim=data.num_node_features, out_dim=2)
    model = model.to(data.x.device)
    lr = cfg["lr"]
    wd = cfg["weight_decay"]
    warmup_start = cfg.get("warmup_start", 0)
    scheduler_warmup = cfg.get("scheduler_warmup", True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    weights = _class_weights_from_train(data)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=0.002)

    res = epoch_loop(
        model, data, optimizer, criterion, scheduler,
        lr=lr, wd=wd, warmup_start=warmup_start, scheduler_warmup=scheduler_warmup
    )
    best_val_f1 = res["best_val_f1"]
    best_test_f1 = res["best_test_f1"]
    stop_epoch = res["stop_epoch"]
    final_lr = res["final_lr"]


    # 1) תחזית “רגילה” (argmax) — נשמור להשוואה
    test_true = data.y[data.test_mask].detach().cpu().numpy()

    # 2) מציאת סף מיטבי על ולידציה
    best_t, best_val_f1_thr = find_best_threshold_on_val(model, data)

    # אחרי שמצאת best_t:
    test_p = _proba_pos(model, data)[data.test_mask.detach().cpu().numpy()]
    auprc = average_precision_score(test_true, test_p)
    print(f"[TEST] AUPRC (class=1): {auprc:.4f}")
    print(
        f"\n[Threshold Tuning] Best threshold on VAL (for Illicit F1): t*={best_t:.2f}, Val F1@t*={best_val_f1_thr:.4f}")

    # 3) תחזית עם הסף המיטבי על סט הבדיקה
    test_pred_thr = predict_with_threshold(model, data, best_t)[data.test_mask.detach().cpu().numpy()]

    print(f"\n[{tag}] Classification Report on Test Set (threshold tuned on VAL @ t*={best_t:.2f}):")
    print(classification_report(test_true, test_pred_thr, target_names=['Licit', 'Illicit']))

    # אפשר גם להחזיר את שני ה-F1 במילון התוצאות
    f1_test_thr = f1_score(test_true, test_pred_thr, pos_label=1)

    # Rich summary row
    return {
        "model": model_name.upper(),
        "features_set": features_set,     # 'local' or 'all'
        "split_type": split_type,         # 'temporal' or 'random'
        "graph_mode": graph_mode,
        "in_channels": int(data.num_node_features),
        "best_val_f1": round(float(best_val_f1), 4),
        "test_f1_at_best": round(float(best_test_f1), 4),
        "test_f1_tuned": round(float(f1_test_thr), 4),
        "stop_epoch": int(stop_epoch),
        "final_lr": float(optimizer.param_groups[0]['lr']),
    }
