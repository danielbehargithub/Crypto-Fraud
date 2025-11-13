import torch
from sklearn.metrics import classification_report, precision_score, recall_score, average_precision_score, f1_score
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import build_model
import numpy as np
import torch.nn.functional as F
import yaml

CFG = yaml.safe_load(open("configs/config_training.yaml"))

def compute_class_weights(y, mask, device):
    """
    Compute per-class weights based on the training subset only.

    The weights are inversely proportional to the class frequency
    (rarer class gets higher weight) and normalized so that their
    mean is 1. The returned tensor is moved to the same device as
    data.x and is intended to be used with nn.CrossEntropyLoss(weight=...).
    """
    y_masked = y[mask]
    counts = torch.bincount(y_masked, minlength=2).float()
    counts[counts == 0] = 1.0
    w = counts.sum() / (2.0 * counts)
    return w.to(device)


# Helpers that pass time_step if the model requires it
def _forward(model, data):
    """Single forward pass that is aware of temporal models."""
    if getattr(model, "requires_time", False):
        if not hasattr(data, "time_step"):
            raise RuntimeError("Model requires data.time_step but it is missing on Data.")
        return model(data.x, data.edge_index, data.time_step)
    return model(data.x, data.edge_index)


def _proba_pos(model, data):
    """Return P(y=1) for all nodes as a NumPy array.
    The model is temporarily switched to eval() mode, a forward pass is
    performed, softmax is applied over the class dimension, and the
    probability of the positive class (index 1) is returned on CPU.
    The original training/eval mode of the model is restored afterwards.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = _forward(model, data)          # logits: [N,2]
        proba = F.softmax(out, dim=1)[:, 1]  # p(y=1)
    if was_training:
        model.train()
    return proba.detach().cpu().numpy()

def predict_with_threshold(model, data, threshold: float):
    """
    Predict binary labels by thresholding the positive-class probability.
    Returns a NumPy array of shape [N] with values in {0, 1}.
    """
    p = _proba_pos(model, data)
    return (p >= threshold).astype(np.int64)


def find_best_threshold_on_val(model, data, thresholds=None):
    """
    Scan over a grid of probability thresholds and pick the one that
    maximizes F1-score on the validation set.

    Returns
    best_t : Threshold that achieved the best F1 on the validation nodes.
    best_f1 : The corresponding F1-score.
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
    """
    Perform a single training step (one epoch over the full graph).

    The model is set to train mode, history is reset if the model
    exposes `reset_history()`, a forward pass is computed, the loss
    is evaluated on the training nodes only, and an optimizer step
    is taken.

    Returns: The scalar training loss value for logging.
    """

    model.train()

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
    """ Run a forward pass and return argmax class predictions for all nodes."""
    model.eval()
    out = _forward(model, data)
    pred = out.argmax(dim=1)
    return pred

def epoch_loop(model, data, optimizer, criterion, scheduler,
               EPS: float | None = None,
               warmup_start: int = 0,
               patience: int | None = None,
               max_epochs: int | None = None,
               scheduler_warmup: bool = True
    ):
    """
    Main training loop with early stopping and LR scheduling.

    The loop repeatedly:
      1. Performs one training step on the full graph.
      2. Finds the best probability threshold on the validation set.
      3. Evaluates F1 and AUPRC on the test set using that threshold.
      4. Tracks the best validation F1 and saves the corresponding
         model state and test metrics.
      5. Applies early stopping with a patience window and an EPS
         tolerance, and steps a ReduceLROnPlateau scheduler.

             Parameters
    model : The GNN / MLP model to train.
    data : Graph data object with train/val/test masks and labels.
    optimizer : Optimizer instance (e.g. Adam).
    criterion : Loss function, typically CrossEntropy with class weights.
    scheduler : LR scheduler driven by validation F1.
    EPS : Minimal required improvement in validation F1 to be considered
        as "better" than the current best.
    warmup_start : Epoch index after which early stopping starts counting.
    patience : Number of epochs without improvement before stopping.
    max_epochs : Upper bound on the number of training epochs.
    scheduler_warmup : Whether to let the scheduler react from the very beginning or
        only after the warmup period.

    Returns
    dict
        A dictionary with summary statistics:
        best_val_f1, best_test_f1, best_threshold, auprc,
        best_epoch, stop_epoch, final_lr.
    """
    es_cfg = CFG["training"]["early_stopping"]
    if EPS is None:
        EPS = es_cfg["eps"]
    if patience is None:
        patience = es_cfg["patience"]
    if max_epochs is None:
        max_epochs = es_cfg["max_epochs"]

    best_val_f1 = -1.0
    best_test_f1 = 0.0
    best_model_state = None  # Save best model weights
    best_epoch = 0
    prev_lr = optimizer.param_groups[0]['lr']
    no_improve = 0
    best_threshold = 0.5

    test_mask = data.test_mask.detach().cpu().numpy().astype(bool)
    test_true = data.y.detach().cpu().numpy()[test_mask]

    for epoch in range(max_epochs):
        loss = train(model, data, optimizer, criterion)

        with torch.no_grad():
            best_t, val_f1  = find_best_threshold_on_val(model, data)

        improved = (val_f1 - best_val_f1) >= EPS
        if improved:

            proba_all = _proba_pos(model, data)
            test_proba = proba_all[test_mask]
            test_f1 = f1_score(test_true, (test_proba >= best_t).astype(int), pos_label=1)
            auprc = average_precision_score(test_true, test_proba)

            best_val_f1 = val_f1
            best_test_f1 = test_f1
            best_epoch = epoch
            best_threshold = best_t
            # Save model state (standard practice)
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            if epoch >= warmup_start:
                no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch})")
                break

        if scheduler_warmup or epoch >= warmup_start:
            scheduler.step(val_f1)

        current_lr = optimizer.param_groups[0]['lr']
        if epoch > 0 and current_lr != prev_lr:
            print(f"🔻 LR reduced at epoch {epoch}: now {current_lr:.6f}")
        prev_lr = current_lr

        if epoch % 20 == 0 or epoch == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}")

    # Restore best model (STANDARD PRACTICE)
    if best_model_state is not None:
        device = next(model.parameters()).device
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    return {
        "best_val_f1": round(float(best_val_f1), 4),
        "best_test_f1": round(float(best_test_f1), 4),
        "best_threshold": round(float(best_threshold), 2),
        "auprc": round(float(auprc), 4),
        "best_epoch": int(best_epoch),
        "stop_epoch": int(epoch),
        "final_lr": float(optimizer.param_groups[0]['lr']),
    }


def run(data, model_name, features_set, split_type, graph_mode):
    """
    High-level experiment runner for a single (model, data-variant) setup.

    This function:
      1. Asserts that the temporal split (train/val/test) is consistent
         and has no time leakage.
      2. Builds the requested model via `build_model`, using the
         configuration file (YAML) for hyperparameters.
      3. Instantiates optimizer, loss, and LR scheduler.
      4. Calls `epoch_loop` to train with early stopping.
      5. Prints a detailed classification report on the test set using
         the best validation-based threshold.
      6. Returns a single summary row as a Python dict for later
         aggregation into a results DataFrame.

             Parameters
    data : Graph data object (already preprocessed).
    model_name : Model identifier, e.g. "GCN", "MLP", "EVOLVEGCN", "DYSAT".
    features_set : Either "all" or "local" indicating which feature subset is used.
    split_type : Either "temporal" or "random", corresponding to the split logic.
    graph_mode : Graph construction mode, e.g. "dag" or "undirected".

    Returns: dict
        Summary metrics and tags describing this run, to be appended
        into a global experiment results table.
    """
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
    weights = compute_class_weights(data.y, data.train_mask, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    sched_cfg = CFG["training"]["scheduler"]
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=sched_cfg["factor"],
        patience=sched_cfg["patience"],
        min_lr=sched_cfg["min_lr"],
    )

    res = epoch_loop(
        model, data, optimizer, criterion, scheduler,
        warmup_start=warmup_start, scheduler_warmup=scheduler_warmup
    )
    best_val_f1 = res["best_val_f1"]
    test_f1 = res["best_test_f1"]
    best_t = res["best_threshold"]
    auprc = res["auprc"]
    best_epoch = res["best_epoch"]
    stop_epoch = res["stop_epoch"]
    final_lr = res["final_lr"]

    test_true = data.y[data.test_mask].detach().cpu().numpy()
    test_pred = predict_with_threshold(model, data, best_t)[data.test_mask.detach().cpu().numpy()]

    print(f"\n[{tag}] Classification Report on Test Set (threshold={best_t:.2f}):")
    print(classification_report(test_true, test_pred, target_names=['Licit', 'Illicit']))


    # Rich summary row
    return {
        "model": model_name.upper(),
        "features_set": features_set,
        "split_type": split_type,
        "graph_mode": graph_mode,
        "in_channels": int(data.num_node_features),
        "best_val_f1": best_val_f1,
        "test_f1": test_f1,
        "auprc": auprc,
        "best_threshold": best_t,
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "final_lr": final_lr,
    }