import math
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, classification_report, average_precision_score
from torch_geometric.data import Data
from models import build_model
from training import _forward, epoch_loop, predict_with_threshold
import os, random
import numpy as np


def _set_all_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

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

@torch.no_grad()
def pick_umcs(
    logits: torch.Tensor,
    remaining_pool: torch.Tensor,
    labeled_idx: torch.Tensor,
    y_true: torch.Tensor,
    k: int,
    illicit_label: int = 1,
    ) -> torch.Tensor:

    device = logits.device
    probs = F.softmax(logits, dim=1).clamp_min(1e-12)
    p_illicit = probs[:, illicit_label]
    pred = (p_illicit >= 0.5).long()  # 1=illicit, 0=licit בבינארי

    s_k = labeled_idx.numel()
    if s_k == 0:
        ent = compute_entropy(logits)
        return pick_by_entropy(ent, remaining_pool, k)

    y_l = y_true[labeled_idx]
    s_illicit = int((y_l == illicit_label).sum().item())
    s_licit = s_k - s_illicit
    mu_k = s_k / 2.0

    minority_is_illicit = (s_illicit < mu_k)
    if not minority_is_illicit:
        ent = compute_entropy(logits)
        return pick_by_entropy(ent, remaining_pool, k)

    m_k = int(max(0, math.floor(mu_k - s_illicit)))
    m_k = min(m_k, k)

    pool_mask = torch.isin(torch.arange(y_true.size(0), device=device), remaining_pool)
    cand_idx = torch.nonzero(pool_mask & (pred == illicit_label), as_tuple=True)[0]

    picked_list = []

    if m_k > 0 and cand_idx.numel() > 0:
        take = min(m_k, cand_idx.numel())
        order = torch.argsort(p_illicit[cand_idx], descending=False)  # קטן->גדול
        chosen_illicit = cand_idx[order[:take]]
        picked_list.append(chosen_illicit)

    k_remaining = k - (picked_list[0].numel() if picked_list else 0)
    if k_remaining > 0:
        if picked_list:
            already = torch.unique(torch.cat(picked_list))
            fill_pool = remaining_pool[~torch.isin(remaining_pool, already)]
        else:
            fill_pool = remaining_pool

        if fill_pool.numel() > 0:
            ent = compute_entropy(logits)
            fill = pick_by_entropy(ent, fill_pool, k_remaining)
            picked_list.append(fill)

    if not picked_list:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.unique(torch.cat(picked_list))


def build_temporal_groups(
    timesteps: torch.Tensor,  # data.time_step (על ה-device)
    train_min_t: int,         # למשל 1
    train_max_t: int,         # למשל 34  (שימו לב: ב-data.py train=1..34)
    n_groups: int,
) -> list[torch.Tensor]:
    """
    מחזיר רשימה של וקטורי אינדקסים (node indices) – כל איבר הוא קבוצה רציפה בזמן.
    הקבוצות מחלקות את טווח train_min_t..train_max_t לפי הסדר.
    """
    device = timesteps.device
    n_groups = max(1, int(n_groups))

    # כל הנקודות שבטווח ה-train
    in_train = (timesteps >= train_min_t) & (timesteps <= train_max_t)
    idx_train = torch.nonzero(in_train, as_tuple=True)[0]
    ts_train = timesteps[idx_train]

    # טווח ה-timesteps (רק ערכי זמן, לא אינדקסים)
    uniq_ts = torch.arange(train_min_t, train_max_t + 1, device=device)
    total_ts = uniq_ts.numel()

    base = total_ts // n_groups
    extra = total_ts % n_groups  # הקבוצות הראשונות יקבלו +1

    groups: list[torch.Tensor] = []
    start = 0
    for g in range(n_groups):
        length = base + (1 if g < extra else 0)
        if length <= 0:
            groups.append(torch.empty(0, dtype=torch.long, device=device))
            continue
        ts_slice = uniq_ts[start:start + length]
        start += length
        # כל האינדקסים ב-train שה-time_step שלהם באחת מהתתי-ערכים שב-slice
        in_group = idx_train[torch.isin(ts_train, ts_slice)]
        groups.append(in_group)

    return groups

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
        acquisition: str = "entropy",
) -> Dict:
    """Pool-based Active Learning on the training split.
    """
    _set_all_seeds(rng_seed)

    tag = f"{model_name.upper()} | {features_set.upper()} | {split_type.upper()} | {graph_mode.upper()} | {acquisition.upper()}"
    print(f"\n===== RUN (Active Learning): {tag}\n")

    device = data.x.device
    y_all = data.y
    n_nodes = y_all.size(0)

    # Candidate pool = labeled nodes inside original train split
    full_train_idx = torch.nonzero(data.train_mask, as_tuple=True)[0]
    labeled_filter = (y_all >= 0)
    pool_idx = full_train_idx[labeled_filter[full_train_idx]]

    # Stratified seed: pick seed_per_class per class from pool (if possible)
    rng = torch.Generator(device='cpu').manual_seed(rng_seed)
    labeled_idx_list: List[torch.Tensor] = []
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

    # AL loop
    curve = []  # list of dicts: {"round": int, "n_labeled": int, "f1_pos_val": float, "auprc_val": float}
    best_t = 0.5
    n_rounds_total = max(1, budget // batch_size)
    n_groups = max(1, n_rounds_total)
    temporal_groups = None
    if acquisition == "sequential":
        temporal_groups = build_temporal_groups(
            timesteps=data.time_step,
            train_min_t=1,
            train_max_t=34,
            n_groups=n_groups,
        )

    while total_acquired <= budget and remaining_pool.numel() > 0:
        round_id += 1
        # print(f"[AL-Round {round_id}]")
        dyn_train_mask = make_dynamic_train_mask(n_nodes, torch.sort(labeled_idx).values, device)
        data.train_mask = dyn_train_mask

        y_l = y_all[labeled_idx]
        n_illicit = int((y_l == 1).sum().item())  # עדכן/י 1 אם ה"לא חוקי" אצלך זה 1; אם זה 0- החלף
        n_licit = int((y_l == 0).sum().item())
        print(
            f"[AL-Round {round_id}] Labeled dist (start): Licit={n_licit} | Illicit={n_illicit} | Total={y_l.numel()}")

        model, cfg = build_model(model_name, in_dim=data.num_node_features, out_dim=2)
        model = model.to(device)

        lr = cfg.get("lr", 2e-2)
        wd = cfg.get("weight_decay", 5e-4)
        warmup_start = cfg.get("warmup_start", 0)
        scheduler_warmup = cfg.get("scheduler_warmup", True)

        # Class weights for every round (want balance)
        def _class_weights_from_mask(y, mask, dev):
            y_tr = y[mask]
            counts = torch.bincount(y_tr, minlength=2).float()
            counts[counts == 0] = 1.0
            w = counts.sum() / (2.0 * counts)
            return w.to(dev)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss(weight=_class_weights_from_mask(y_all, dyn_train_mask, device))
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=0.002)

        res = epoch_loop(
            model, data, optimizer, criterion, scheduler,
            lr=lr, wd=wd, warmup_start=warmup_start, scheduler_warmup=scheduler_warmup, max_epochs=max_epochs_per_round
        )
        best_val_f1 = res["best_val_f1"]
        best_test_f1 = res["best_test_f1"]
        best_t = res["best_threshold"]
        auprc = res["auprc"]
        best_epoch = res["best_epoch"]
        stop_epoch = res["stop_epoch"]
        final_lr = res["final_lr"]

        curve.append({
            "round": round_id,
            "n_labeled": labeled_idx.numel(),
            "f1_pos_val": best_val_f1,
            "auprc_val": auprc,
            "best_epoch": best_epoch,
            "stop_epoch": stop_epoch,
            "final_lr": final_lr,
            "best_threshold": best_t,
        })

        print(
            f"[AL-Round {round_id}] Labeled={labeled_idx.numel()} | Best Val F1={best_val_f1:.4f} | Test F1 best={best_test_f1:.4f} "
            f"| best threshold={best_t}\n")

        k = min(batch_size, budget - total_acquired, remaining_pool.numel())
        if k <= 0:
            break
        if acquisition == "random":
            perm = torch.randperm(remaining_pool.numel(), generator=rng)
            picked_cpu = remaining_pool.detach().cpu()[perm[:k]]
            picked = picked_cpu.to(remaining_pool.device)
        else:
            model.eval()
            with torch.no_grad():
                logits = _forward(model, data)

            if acquisition == "entropy":
                entropy = compute_entropy(logits)
                picked = pick_by_entropy(entropy, remaining_pool, k)

            elif acquisition == "umcs":
                picked = pick_umcs(
                    logits=logits,
                    remaining_pool=remaining_pool,
                    labeled_idx=torch.sort(labeled_idx).values,
                    y_true=y_all,
                    k=k,
                    illicit_label=1,  
                )
            else:
                curr_gid = round_id % len(temporal_groups)
                group_idx = temporal_groups[curr_gid]
                pool_mask = torch.isin(remaining_pool, group_idx)
                pool_group = remaining_pool[pool_mask]
                picked_group = pick_umcs(
                    logits=logits,
                    remaining_pool=pool_group,
                    labeled_idx=torch.sort(labeled_idx).values,
                    y_true=y_all,
                    k=k,
                    illicit_label=1,
                )
                if picked_group.numel() == k:
                    picked = picked_group
                else:
                    k_rem = k - picked_group.numel()
                    if k_rem > 0:
                        entropy = compute_entropy(logits)
                        if picked_group.numel() > 0:
                            fill_pool = remaining_pool[~torch.isin(remaining_pool, picked_group)]
                        else:
                            fill_pool = remaining_pool
                        fill = pick_by_entropy(entropy, fill_pool, k_rem)
                        picked = torch.unique(torch.cat([picked_group, fill]))
                    else:
                        picked = picked_group

        labeled_idx = torch.unique(torch.cat([labeled_idx, picked]))
        total_acquired += picked.numel()
        remaining_pool = remaining_pool[~torch.isin(remaining_pool, picked)]

    test_true = data.y[data.test_mask].detach().cpu().numpy()
    test_pred = predict_with_threshold(model, data, best_t)[data.test_mask.detach().cpu().numpy()]

    print(f"\n[{tag}] Classification Report on Test Set (threshold={best_t:.2f}):")
    print(classification_report(test_true, test_pred, target_names=['Licit', 'Illicit']))

    return {
        "model": f"AL-{model_name.upper()}",
        "acquisition": acquisition,
        "features_set": features_set,
        "split_type": split_type,
        "graph_mode": graph_mode,
        "in_channels": int(data.num_node_features),
        "best_val_f1": best_val_f1,
        "test_f1": best_test_f1,
        "auprc": auprc,
        "best_threshold": best_t,
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "final_lr": final_lr,
        "final_labeled": int(labeled_idx.numel()),
        "curve": curve,
    }