import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import networkx as nx
from typing import Dict, Optional, Tuple, List
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG_VIS = yaml.safe_load(open(ROOT / "configs" / "config_visual.yaml"))
CFG_VIS = CFG_VIS["visual"]

DATASET_DIR = ROOT / CFG_VIS["dataset_dir"]
RESULTS_DIR = ROOT / CFG_VIS["results_dir"]
VIS_DIR     = ROOT / CFG_VIS["vis_dir"]

AL_VIS_DIR   = VIS_DIR / CFG_VIS.get("al_vis_subdir", "AL_visualizations")
AL_PLOTS_DIR = AL_VIS_DIR / CFG_VIS.get("al_plots_subdir", "plots")
AL_RATIO_DIR = AL_VIS_DIR / CFG_VIS.get("al_ratio_subdir", "illicit_ratio")

FILES = CFG_VIS["files"]

# -------------------------------
# 1) Data preview
def plot_data():
    """
    Quick NetworkX visualization of an Elliptic subgraph, color-coded by label.
    Expects the original CSVs under 'elliptic_bitcoin_dataset/'.
    """
    nodes_column_names = ['txId', 'time_step'] + [f'feature_{i}' for i in range(1, 166)]

    nodes_df = pd.read_csv(
        DATASET_DIR / "elliptic_txs_features.csv",
        header=None,
        names=nodes_column_names,
    )
    edges_df = pd.read_csv(DATASET_DIR / "elliptic_txs_edgelist.csv")
    labels_df = pd.read_csv(DATASET_DIR / "elliptic_txs_classes.csv")

    # Attach labels
    nodes_df = nodes_df.merge(labels_df, on='txId', how='left')

    # Simple id mapping
    txid_to_idx = {tx_id: i for i, tx_id in enumerate(nodes_df['txId'].values)}
    nodes_df['idx'] = nodes_df['txId'].map(txid_to_idx)

    edges_df['src'] = edges_df['txId1'].map(txid_to_idx)
    edges_df['dst'] = edges_df['txId2'].map(txid_to_idx)
    _ = torch.tensor(edges_df[['src', 'dst']].values.T, dtype=torch.long)  # kept for parity

    # Build NetworkX graph for a subcomponent
    G_nx = nx.from_pandas_edgelist(edges_df, 'txId1', 'txId2')
    components = list(nx.connected_components(G_nx))
    subgraph_nodes = list(components[1])[:]  # pick a component (as in your original)
    subgraph = G_nx.subgraph(subgraph_nodes)

    # Color nodes by class
    label_dict = nodes_df.set_index('txId')['class'].to_dict()
    node_colors = []
    for node in subgraph.nodes():
        label = label_dict.get(node, -1)
        if label == '1':
            node_colors.append('red')      # illicit
        elif label == '2':
            node_colors.append('green')    # licit
        else:
            node_colors.append('gray')     # unknown

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw(subgraph, pos, node_color=node_colors, with_labels=False,
            node_size=50, edge_color='lightgray')
    plt.title("Elliptic Subgraph (Red=Illicit, Green=Licit, Gray=Unknown)")
    plt.savefig(VIS_DIR / FILES["elliptic_subgraph"])



# -------------------------------
# Paired DELTAs utilities (GRAPH_MODE by MODEL, FEATURES_SET by MODEL, etc.)
# -------------------------------

def compute_paired_deltas_from_csv(
    csv_path: str,
    factor_col: str = "graph_mode",   # column with two levels to compare (e.g., 'dag' vs 'undirected')
    a: str = "dag",
    b: str = "undirected",
    metric: str = "test_f1",
    group_cols: Optional[List[str]] = None,   # identity columns that must match within a pair
    filters: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of paired deltas: delta = metric(a) - metric(b).
    A delta is created only when both a and b exist for the same identity (group_cols).
    """
    df = pd.read_csv(csv_path)

    if filters:
        for k, v in filters.items():
            df = df[df[k] == v]

    if group_cols is None:
        # default identity for passive runs
        group_cols = ["model", "features_set", "split_type", "in_channels"]


    sub = df[df[factor_col].isin([a, b])].copy()
    if sub.empty:
        return pd.DataFrame(columns=group_cols + [f"{metric}@{a}", f"{metric}@{b}", "delta"])

    pv = sub.pivot_table(index=group_cols, columns=factor_col, values=metric, aggfunc="max")

    needed = [x for x in [a, b] if x in pv.columns]
    if len(needed) < 2:
        return pd.DataFrame(columns=group_cols + [f"{metric}@{a}", f"{metric}@{b}", "delta"])

    pv = pv.dropna(subset=needed, how="any").copy()
    if pv.empty:
        return pd.DataFrame(columns=group_cols + [f"{metric}@{a}", f"{metric}@{b}", "delta"])

    pv["delta"] = pv[a] - pv[b]
    out = pv.reset_index().rename(columns={a: f"{metric}@{a}", b: f"{metric}@{b}"})
    return out  # columns: group_cols + [metric@a, metric@b, delta]


def delta_boxplot_from_csv(
    csv_path: str,
    factor_col: str = "graph_mode",
    a: str = "dag",
    b: str = "undirected",
    metric: str = "auto",
    group_cols: Optional[List[str]] = None,
    filters: Optional[Dict[str, object]] = None,
    x_col: Optional[str] = None,              # optional category to split boxes, e.g. "model"
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 4.5),
    showfliers: bool = False,
    save_path: str | Path = None,
    show: bool = False,
):
    """
    Draw a boxplot over paired deltas.
    If x_col is provided (e.g., 'model'), draw one box per category; otherwise a single box.
    """
    deltas = compute_paired_deltas_from_csv(
        csv_path=csv_path,
        factor_col=factor_col, a=a, b=b,
        metric=metric, group_cols=group_cols, filters=filters
    )
    if deltas.empty:
        return None

    if save_path is None:
        raise ValueError("save_path must be provided by the caller.")

    dcol = "delta"
    cats = ["ALL"]
    data = [deltas[dcol].dropna().values]

    if x_col is not None and x_col in deltas.columns:
        cats = sorted(deltas[x_col].dropna().unique().tolist())
        data = [deltas.loc[deltas[x_col] == c, dcol].dropna().values for c in cats]

    fig = plt.figure(figsize=figsize)
    bp = plt.boxplot(
        data, tick_labels=[str(c) for c in cats],
        showfliers=showfliers, patch_artist=True, widths=0.3
    )
    for patch in bp['boxes']:
        patch.set_alpha(0.5)

    plt.axhline(0.0, linestyle="--", linewidth=1, alpha=0.6)
    plt.ylabel(f"ΔF1")
    plt.xlabel("Category" if x_col is None else x_col)
    if title is None:
        title = f"Effect of {factor_col} (ΔF1 = {a} − {b})"
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=160)
    plt.close()
    return fig


def compute_leaderboard(
    csv_path: str,
    metric: str = "test_f1",
    group_col: str = "model",
    output_path: str | Path = None,
):
    """
    Create a unified leaderboard table combining:
      - mean rank (lower = better)
      - mean metric (raw)
      - mean scaled metric (0–1 range)
    """
    df = pd.read_csv(csv_path)

    df["rank"] = df[metric].rank(ascending=False, method="min")

    rank_summary = (
        df.groupby(group_col)[["rank", metric]]
        .agg(
            mean_rank=("rank", "mean"),
            mean_metric=(metric, "mean"),
        )
        .reset_index()
    )

    df["scaled"] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
    scaled_summary = (
        df.groupby(group_col)["scaled"]
        .mean()
        .reset_index()
        .rename(columns={"scaled": f"proportional_mean_{metric}"})
    )

    leaderboard = pd.merge(rank_summary, scaled_summary, on=group_col)
    leaderboard = leaderboard.rename(
        columns={
            "mean_rank": f"mean_rank",
            "mean_metric": f"mean_{metric}",
        }
    )
    for col in leaderboard.select_dtypes(include="number").columns:
        leaderboard[col] = leaderboard[col].round(3)

    leaderboard = leaderboard.sort_values(f"mean_rank").reset_index(drop=True)
    if output_path is not None:
        leaderboard.to_csv(output_path, index=False)
    return leaderboard

##
## AL plots
##

def plot_al_by_model(
    csv_path: str,
    passive_csv_path: str,
    facet_col: str = "model",                # facet = one panel per model
    line_col: str = "method",           # line = AL strategy
    metric: str = "test_f1",                    # auto prefers test if present
    label_col: str = "n_labeled",
    filters: Optional[Dict[str, object]] = None,  # optional filtering (e.g., DAG/local only)
    agg: str = "mean",                       # aggregate F1 across seeds/runs at each n_labeled
    ci: Optional[str] = "sem",               # "std"/"sem"/None error ribbon
    figsize: Tuple[int,int] = (8,5),
):
    """
    Plot Active Learning performance curves per model.

    For each value of `facet_col` (e.g. each model), this function:
      - aggregates AL runs (from `csv_path`) across seeds/configs
        to produce one mean curve per AL method (line_col),
      - optionally adds an uncertainty band (std or sem),
      - overlays a horizontal passive baseline taken from
        `passive_csv_path` (same facet, same filters),
      - saves each panel as a PNG under `visualizations/AL_visualizations/plots/`.
    """
    df = pd.read_csv(csv_path)
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")

    # apply optional filters (e.g., {"graph_mode":"dag","features_set":"local"})
    if filters:
        for k, v in filters.items():
            df = df[df[k] == v]

    passive_df = pd.read_csv(passive_csv_path)

    facet_values = sorted(df[facet_col].dropna().unique())

    for fv in facet_values:
        sub = df[df[facet_col] == fv].copy()
        if sub.empty:
            continue

        # aggregate across seeds/configs: one mean curve per AL method
        grp = sub.groupby([line_col, label_col], as_index=False).agg(
            f1=(metric, agg),
            count=(metric, "count"),
            std=(metric, "std")
        )

        plt.figure(figsize=figsize)

        # --- draw AL curves ---
        for lc, g in grp.groupby(line_col):
            g = g.sort_values(label_col)
            x = g[label_col].values
            y = g["f1"].values
            plt.plot(x, y, marker="o", label=str(lc))

            # optional ribbon
            if ci in ("std", "sem"):
                err = g["std"].fillna(0).values
                if ci == "sem":
                    n = g["count"].clip(lower=1).values
                    err = err / np.sqrt(n)
                plt.fill_between(x, y - err, y + err, alpha=0.15)

        # --- passive baseline line ---
        base_model = fv.replace("AL-", "").replace(" ", "")
        passive_sub = passive_df[passive_df[facet_col] == base_model].copy()
        if filters:
            for k, v in filters.items():
                if k in passive_sub.columns:
                    passive_sub = passive_sub[passive_sub[k] == v]
        passive_val = np.nan
        if not passive_sub.empty:
            if metric in passive_sub.columns:
                passive_val = passive_sub[metric].mean()
            elif "test_f1" in passive_sub.columns:
                passive_val = passive_sub["test_f1"].mean()
        if not np.isnan(passive_val):
            x_min = grp[label_col].min()
            x_max = grp[label_col].max()
            plt.hlines(
                passive_val,
                xmin=x_min,
                xmax=x_max,
                linestyles="--",
                linewidth=1.5,
                label="Passive baseline"
            )

        plt.title(f"{fv} F1 vs Number of Labeled Samples")
        plt.xlabel("Number of labeled samples")
        plt.ylabel("F1 score")
        plt.grid(True, alpha=0.3)
        plt.legend(title=line_col)
        plt.tight_layout()

        save_name = AL_PLOTS_DIR / f"plot_{fv}.png".replace(" ", "_")
        plt.savefig(save_name, dpi=200, bbox_inches="tight")
        plt.close()


def plot_illicit_ratio(
    csv_path: str,
    facet_col: str = "model",
    line_col: str = "method",
    label_col: str = "n_labeled",
    illicit_col: str = "n_illicit",
    figsize: tuple = (8, 5)
):
    """
    Plot ratio of illicit labels (n_illicit / n_labeled) per AL strategy, for each model.
    """
    df = pd.read_csv(csv_path)
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df[illicit_col] = pd.to_numeric(df[illicit_col], errors="coerce")

    # compute ratio
    df["ratio_illicit"] = df[illicit_col] / df[label_col]
    df = df.dropna(subset=["ratio_illicit"])

    facet_values = sorted(df[facet_col].dropna().unique())

    for fv in facet_values:
        sub = df[df[facet_col] == fv].copy()
        if sub.empty:
            continue

        grp = sub.groupby([line_col, label_col], as_index=False).agg(
            ratio=("ratio_illicit", "mean")
        )

        plt.figure(figsize=figsize)
        for lc, g in grp.groupby(line_col):
            g = g.sort_values(label_col)
            plt.plot(g[label_col], g["ratio"], marker="o", label=str(lc))

        plt.title(f"{fv} Ratio of Illicit Labels")
        plt.xlabel("Number of labeled samples")
        plt.ylabel("ratio")
        plt.grid(True, alpha=0.3)
        plt.legend(title=line_col)
        plt.tight_layout()

        save_name = AL_RATIO_DIR / f"illicit_ratio_{fv}.png".replace(" ", "_")
        plt.savefig(save_name, dpi=200, bbox_inches="tight")
        plt.close()


def _partial_corr_1d(x, y, z):
    """
    partial corr between x and y controlling for z (all 1D arrays, same length).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    # design matrix for z (with bias)
    Xz = np.column_stack([np.ones_like(z), z])

    # regress x on z
    bx, _, _, _ = np.linalg.lstsq(Xz, x, rcond=None)
    rx = x - Xz @ bx

    # regress y on z
    by, _, _, _ = np.linalg.lstsq(Xz, y, rcond=None)
    ry = y - Xz @ by

    # correlation between residuals
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return np.corrcoef(rx, ry)[0, 1]


def compute_balance_partial_corr(
    csv_path: str,
    model_col: str = "model",
    al_col: str = "method",
    label_col: str = "n_labeled",
    illicit_col: str = "n_illicit",
    f1_col: str = "best_test_f1",
    output_path: str = "visualizations/AL_visualizations/illicit_balance_effect.csv",
):
    """
    For each (model, method), compute:
      - corr_raw: Pearson corr between balance and F1
      - corr_partial: partial corr between balance and F1 controlling for n_labeled
    Balance is defined from n_illicit / n_labeled, maximal at perfect 0.5 balance.
    """
    df = pd.read_csv(csv_path)

    # make sure numeric
    for c in [label_col, illicit_col, f1_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows with missing core fields
    df = df.dropna(subset=[label_col, illicit_col, f1_col])

    # ratio + balance
    df["ratio_illicit"] = df[illicit_col] / df[label_col]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio_illicit"])

    df["balance"] = 1.0 - 2.0 * np.abs(df["ratio_illicit"] - 0.5)

    rows = []
    group_cols = [model_col, al_col]

    for keys, sub in df.groupby(group_cols):
        if len(sub) < 3:
            continue

        b = sub["balance"].values
        f1 = sub[f1_col].values
        nlab = sub[label_col].values

        # raw corr(B, F1)
        if np.std(b) == 0 or np.std(f1) == 0:
            corr_raw = np.nan
        else:
            corr_raw = np.corrcoef(b, f1)[0, 1]

        # partial corr(B, F1 | n_labeled)
        corr_part = _partial_corr_1d(b, f1, nlab)

        row = {
            model_col: keys[0],
            al_col: keys[1],
            "n_points": len(sub),
            "corr_partial": corr_part,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    # round for readability
    for c in out.select_dtypes(include="number").columns:
        out[c] = out[c].round(3)

    out.to_csv(output_path, index=False)
    return out

def run_all_visualizations():
    """
    Convenience entry point to recompute all visual artifacts
    (figures, CSV summaries) for the current experiment results.
    """
    passive_csv = RESULTS_DIR / FILES["summary_passive"]
    active_csv = RESULTS_DIR / FILES["summary_active"]
    curves_csv = RESULTS_DIR / FILES["run_curves"]


    plot_data()

    delta_boxplot_from_csv(
        csv_path=passive_csv,
        factor_col="graph_mode", a="dag", b="undirected",
        x_col="model", metric="test_f1",
        group_cols=["model", "features_set", "split_type", "in_channels"],
        save_path=VIS_DIR / FILES["boxplot_graph_mode"],
    )

    delta_boxplot_from_csv(
        csv_path=passive_csv,
        factor_col="features_set", a="all", b="local",
        x_col="model", metric="test_f1",
        group_cols=["model", "graph_mode", "split_type"],
        save_path=VIS_DIR / FILES["boxplot_features_set"],
    )

    compute_leaderboard(
        csv_path=passive_csv,
        metric="test_f1",
        group_col="model",
        output_path=VIS_DIR / FILES["leaderboard_passive"],
    )

    compute_balance_partial_corr(
        csv_path=curves_csv,
        output_path=AL_VIS_DIR / FILES["illicit_balance_effect"],
    )

    plot_illicit_ratio(
        csv_path=curves_csv,
        facet_col="model",
        line_col="method",
    )

    compute_leaderboard(
        csv_path=active_csv,
        metric="test_f1",
        group_col="method",
        output_path=AL_VIS_DIR / FILES["leaderboard_active"],
    )


    plot_al_by_model(
        csv_path=curves_csv,
        passive_csv_path=passive_csv,
        facet_col="model",
        line_col="method",
        metric="best_test_f1",
        label_col="n_labeled",
        filters={"graph_mode": "dag", "features_set": "local"},
    )

if __name__ == "__main__":
    run_all_visualizations()
