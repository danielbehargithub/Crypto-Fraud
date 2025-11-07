import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt

def plot_f1_minority_vs_labels(df_curves, title="F1 (minority) vs. #Labeled", save_path="al_f1_curve.png"):
    if df_curves.empty:
        print("[plot] No curve data to plot.")
        return

    # ממיינים כדי לקבל קווים חלקים
    dfc = df_curves.sort_values(["acquisition", "n_labeled"]).copy()

    plt.figure(figsize=(8, 5.5))
    for acq, g in dfc.groupby("acquisition"):
        plt.plot(g["n_labeled"], g["f1_pos_val"], marker="o", linewidth=2, label=str(acq))

    plt.xlabel("Number of labeled samples (cumulative)")
    plt.ylabel("F1 (minority class)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Acquisition")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    print(f"[plot] Saved: {save_path}")

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
