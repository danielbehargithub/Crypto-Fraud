import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, Tuple, List


def plot_data():
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




    # נבנה את הגרף כ־networkx בשביל ויזואליזציה
    G_nx = nx.from_pandas_edgelist(edges_df, 'txId1', 'txId2')

    # נבחר תת-גרף ראשון (הקומפוננטה הראשונה)
    components = list(nx.connected_components(G_nx))
    subgraph_nodes = list(components[1])[:]  # 100 צמתים ראשונים מתוך הרכיב
    subgraph = G_nx.subgraph(subgraph_nodes)

    # נצבע לפי label
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

    # ציור
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(subgraph, seed=42)  # סידור אוטומטי
    nx.draw(subgraph, pos, node_color=node_colors, with_labels=False, node_size=50, edge_color='lightgray')
    plt.title("Elliptic Subgraph Visualization (Red=Illicit, Green=Licit, Gray=Unknown)")
    plt.show()





def plot_al_by_model(
    csv_path: str = "run_curves.csv",
    facet_col: str = "model",           # פאנל=מודל
    line_col: str = "acquisition",      # קו=שיטת AL
    metric: str = "auto",               # auto = בוחר test אם קיים
    label_col: str = "n_labeled",
    filters: Optional[Dict[str, object]] = None,  # סינון לקונפיגורציה מייצגת
    agg: str = "mean",                  # ממוצע/חציון של F1 בכל n_labeled
    ci: Optional[str] = "sem",          # שגיאה (std/sem) או None
    figsize: Tuple[int,int] = (8,5)
):
    # טען CSV
    df = pd.read_csv(csv_path)

    # קביעת מטריקה
    if metric == "auto":
        metric = "best_test_f1" if "best_test_f1" in df.columns else "best_val_f1"

    # טיפולים בסיסיים
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")

    # החלת פילטרים אם רוצים להשוות רק קונפיגורציה אחת (למשל רק DAG/local)
    if filters:
        for k, v in filters.items():
            df = df[df[k] == v]

    # ערכי הפאנל (למשל כל המודלים)
    facet_values = sorted(df[facet_col].dropna().unique())

    for fv in facet_values:
        sub = df[df[facet_col] == fv].copy()
        if sub.empty:
            continue

        # אגרגציה: ממוצע/חציון לכל (AL method, n_labeled)
        grp = sub.groupby([line_col, label_col], as_index=False).agg(
            f1=(metric, agg),
            count=(metric, "count"),
            std=(metric, "std")
        )

        # ציור
        plt.figure(figsize=figsize)

        for lc, g in grp.groupby(line_col):
            g = g.sort_values(label_col)
            x = g[label_col].values
            y = g["f1"].values
            plt.plot(x, y, marker="o", label=str(lc))

            # סרטי שגיאה (רשות)
            if ci in ("std", "sem"):
                err = g["std"].fillna(0).values
                if ci == "sem":
                    n = g["count"].clip(lower=1).values
                    err = err / np.sqrt(n)
                plt.fill_between(x, y - err, y + err, alpha=0.15)

        plt.title(f"{facet_col} = {fv}")
        plt.xlabel("Number of labeled samples")
        plt.ylabel("F1 score")
        plt.grid(True, alpha=0.3)
        plt.legend(title=line_col)
        plt.tight_layout()
        plt.show()

plot_al_by_model("run_curves.csv")

plot_al_by_model(
    "run_curves.csv",
    filters={
        "graph_mode": "dag",
        "features_set": "local",
        "split_type": "temporal",
        "in_channels": 94,
    }
)
