from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from models import GCN, MLP
from training import train, test


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
    ent = entropy.detach().cpu().numpy()
    cand = candidate_idx.detach().cpu().numpy()
    order = ent[cand].argsort()[::-1]  # descending by entropy
    take = min(k, cand.shape[0])
    picked = cand[order[:take]]
    return torch.tensor(picked, dtype=torch.long, device=entropy.device)

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
) -> Dict:
    """Pool-based Active Learning on the training split using entropy sampling."""
    tag = f"{model_name.upper()} | {features_set.upper()} | {split_type.upper()} | {graph_mode.upper()}"
    print(f"\n===== RUN (Active Learning): {tag} =====")

    device = data.x.device
    y_all = data.y
    n_nodes = y_all.size(0)

    # Candidate pool = labeled nodes inside original train split
    full_train_idx = torch.nonzero(data.train_mask, as_tuple=True)[0]
    labeled_filter = (y_all >= 0)
    pool_idx = full_train_idx[labeled_filter[full_train_idx]]

    # Stratified seed: pick seed_per_class per class from pool (if possible)
    # attention-for now picks same 20 every time
    rng = torch.Generator(device='cpu').manual_seed(rng_seed)
    labeled_idx_list = []
    for cls in torch.unique(y_all[pool_idx]):
        if cls.item() < 0:
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

    remaining_pool = pool_idx[~torch.isin(pool_idx, labeled_idx)]
    total_acquired = 0
    round_id = 0

    best_val_f1_overall = -1.0
    best_test_f1_at_best = 0.0

    # AL loop
    while total_acquired < budget and remaining_pool.numel() > 0:
        round_id += 1
        dyn_train_mask = make_dynamic_train_mask(n_nodes, torch.sort(labeled_idx).values, device)

        # Fresh model each round
        if model_name.upper() == 'GCN':
            model = GCN(in_channels=data.num_node_features, hidden_channels=64, out_channels=2).to(device)
            warmup_start = 10; scheduler_warmup = True
        else:
            model = MLP(in_channels=data.num_node_features, hidden1=128, hidden2=64, out_channels=2).to(device)
            warmup_start = 50; scheduler_warmup = False

        optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=0.002)

        patience = 30
        no_improve = 0
        EPS = 0.0002
        best_val_f1 = -1.0
        best_test_f1 = 0.0

        for epoch in range(max_epochs_per_round):
            loss = train(model, data, optimizer, criterion, dyn_train_mask)
            pred = test(model, data)
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

        if best_val_f1 > best_val_f1_overall:
            best_val_f1_overall = best_val_f1
            best_test_f1_at_best = best_test_f1

        print(f"[AL-Round {round_id}] Labeled={labeled_idx.numel()} | Best Val F1={best_val_f1:.4f} | Test F1@best={best_test_f1:.4f}")

        # Acquisition: top-entropy from remaining pool
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            entropy = compute_entropy(logits)

        k = min(batch_size, budget - total_acquired, remaining_pool.numel())
        if k <= 0:
            break
        picked = pick_by_entropy(entropy, remaining_pool, k)

        labeled_idx = torch.unique(torch.cat([labeled_idx, picked]))
        total_acquired += picked.numel()
        remaining_pool = remaining_pool[~torch.isin(remaining_pool, picked)]

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
