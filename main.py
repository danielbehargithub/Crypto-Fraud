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
from data import get_variants


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

# ===== Active Learning helpers =====

@torch.no_grad()
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Return per-node entropy (higher = more uncertain)."""
    probs = F.softmax(logits, dim=1).clamp_min(1e-12)
    return -(probs * probs.log()).sum(dim=1)

def make_dynamic_train_mask(n_nodes: int, labeled_idx: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Build a boolean train mask from given labeled indices."""
    mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    if labeled_idx.numel() > 0:
        mask[labeled_idx] = True
    return mask

def pick_by_entropy(entropy: torch.Tensor, candidate_idx: torch.Tensor, k: int) -> torch.Tensor:
    """Pick the top-k most uncertain nodes (by entropy) from the candidate set."""
    # entropy and candidate_idx expected on CPU or same device; we move to CPU numpy ordering for simplicity
    ent = entropy.detach().cpu().numpy()
    cand = candidate_idx.detach().cpu().numpy()
    order = ent[cand].argsort()[::-1]  # descending by entropy
    take = min(k, cand.shape[0])
    picked = cand[order[:take]]
    return torch.tensor(picked, dtype=torch.long, device=entropy.device)


# ==== Runner: unchanged logic, but returns richer summary ====
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


def run_active_learning(
    data: Data,
    model_name: str,
    features_set: str,
    split_type: str,
    graph_mode: str,
    seed_per_class: int = 10,
    batch_size: int = 50,
    budget: int = 200,
    max_epochs_per_round: int = 100,
    rng_seed: int = 42,
):
    """
    Minimal pool-based Active Learning with entropy uncertainty on train split.
    We simulate labeling by gradually revealing labels from the existing train set (y!=-1).
    """
    tag = f"{model_name.upper()} | {features_set.upper()} | {split_type.upper()} | {graph_mode.upper()}"
    print(f"\n===== RUN (Active Learning): {tag} =====")

    device = data.x.device
    y_all = data.y
    n_nodes = y_all.size(0)

    # Build candidate pool: only nodes that are in the original train_mask and have labels (y!=-1)
    full_train_idx = torch.nonzero(data.train_mask, as_tuple=True)[0]
    labeled_filter = (y_all >= 0)
    pool_idx = full_train_idx[labeled_filter[full_train_idx]]

    # Stratified seed: pick seed_per_class per class from pool (if possible)
    rng = torch.Generator(device='cpu').manual_seed(rng_seed)
    labeled_idx_list = []
    for cls in torch.unique(y_all[pool_idx]):
        if cls.item() < 0:  # skip unknown just in case
            continue
        cls_idx = pool_idx[(y_all[pool_idx] == cls)]
        if cls_idx.numel() == 0:
            continue
        k = min(seed_per_class, cls_idx.numel())
        choice = cls_idx[torch.randperm(cls_idx.numel(), generator=rng)[:k]]
        labeled_idx_list.append(choice)
    if len(labeled_idx_list) == 0:
        raise RuntimeError("No labeled seeds available in the training pool.")
    labeled_idx = torch.unique(torch.cat(labeled_idx_list)).to(device)

    # Keep track of which pool nodes are still unlabeled by AL
    remaining_pool = pool_idx[~torch.isin(pool_idx, labeled_idx)]

    total_acquired = 0
    round_id = 0

    best_val_f1_overall = -1.0
    best_test_f1_at_best = 0.0

    # AL outer loop
    while total_acquired < budget and remaining_pool.numel() > 0:
        round_id += 1

        # Build dynamic train mask from currently "labeled" indices
        dyn_train_mask = make_dynamic_train_mask(n_nodes, torch.sort(labeled_idx).values, device)

        # Init a fresh model each round (clean training)
        if model_name.upper() == 'GCN':
            model = GCN(in_channels=data.num_node_features, hidden_channels=64, out_channels=2).to(device)
            warmup_start = 10
            scheduler_warmup = True
        else:
            model = MLP(in_channels=data.num_node_features, hidden1=128, hidden2=64, out_channels=2).to(device)
            warmup_start = 50
            scheduler_warmup = False

        optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=0.002)

        # Train for a limited number of epochs with early stopping (like your run())
        patience = 30
        no_improve = 0
        EPS = 0.0002
        best_val_f1 = -1.0
        best_test_f1 = 0.0

        for epoch in range(max_epochs_per_round):
            # --- train step on dynamic mask ---
            model.train()
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = criterion(logits[dyn_train_mask], y_all[dyn_train_mask])
            loss.backward()
            optimizer.step()

            # --- eval on val/test ---
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                pred = logits.argmax(dim=1)

                val_true = y_all[data.val_mask].detach().cpu().numpy()
                val_pred = pred[data.val_mask].detach().cpu().numpy()
                val_f1 = f1_score(val_true, val_pred, average='binary', pos_label=1)

                test_true = y_all[data.test_mask].detach().cpu().numpy()
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
                    break

            if scheduler_warmup or epoch >= warmup_start:
                scheduler.step(val_f1)

        # Track best across rounds
        if best_val_f1 > best_val_f1_overall:
            best_val_f1_overall = best_val_f1
            best_test_f1_at_best = best_test_f1

        print(f"[AL-Round {round_id}] Labeled={labeled_idx.numel()} | Best Val F1={best_val_f1:.4f} | Test F1@best={best_test_f1:.4f}")

        # --- acquisition step: pick top-entropy nodes from remaining_pool ---
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            entropy = compute_entropy(logits)  # per-node

        k = min(batch_size, budget - total_acquired, remaining_pool.numel())
        if k <= 0:
            break

        picked = pick_by_entropy(entropy, remaining_pool, k)
        labeled_idx = torch.unique(torch.cat([labeled_idx, picked]))
        total_acquired += picked.numel()

        # Update remaining_pool
        remaining_pool = remaining_pool[~torch.isin(remaining_pool, picked)]

    # Final summary
    print(f"\n[{tag}] AL finished. Best Val F1={best_val_f1_overall:.4f}, Test F1 at best Val={best_test_f1_at_best:.4f}")
    return {
        "model": f"AL-{model_name.upper()}",
        "features_set": features_set,
        "split_type": split_type,
        "graph_mode": graph_mode,
        "in_channels": int(data.num_node_features),
        "best_val_f1": round(float(best_val_f1_overall), 4),
        "test_f1_at_best": round(float(best_test_f1_at_best), 4),
        "final_labeled": int(labeled_idx.numel()),
    }



def main():
    # Ask for both modes in one call; you can choose ("dag",) or ("undirected",) too

    # Choose subsets here:
    graph_modes  = ["dag", "undirected"]   # or ["dag"] or ["undirected"]
    model_names  = ["GCN"]                  # e.g., ["GCN"] to run only GCN. MLP available
    feature_sets = ["local"]                # e.g., ["local"] to run only LOCAL. all available
    split_types  = ["temporal"]             # e.g., ["temporal"] to run only TEMPORAL. random available
    data_variants = get_variants()

    rows = []
    for gm in graph_modes:
        for model_name in model_names:
            for fset in feature_sets:
                for split in split_types:
                    data_obj = data_variants[(gm, fset, split)]
                    row = run(data_obj, model_name, fset, split, gm)
                    rows.append(row)


    df = pd.DataFrame(rows)[
        ["graph_mode", "model", "features_set", "split_type",
         "in_channels", "best_val_f1", "test_f1_at_best", "stop_epoch", "final_lr"]
    ]
    print("\n=== Summary Table (all runs) ===")
    print(df.to_string(index=False))
    df.to_csv("run_summary.csv", index=False)
    print("Saved: run_summary.csv")

    # al_summary = run_active_learning(
    #     data=data_variants[('all', 'temporal')],
    #     model_name='GCN',
    #     features_set='all',
    #     split_type='temporal',
    #     seed_per_class=10,   # starting labeled per class
    #     batch_size=50,       # acquired per round
    #     budget=200,          # total acquired across rounds
    #     max_epochs_per_round=100,
    #     rng_seed=42
    # )
    # print("\n=== Active Learning Summary ===")
    # print(al_summary)

if __name__ == "__main__":
    main()


