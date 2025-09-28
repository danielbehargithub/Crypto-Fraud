# -*- coding: utf-8 -*-
"""
Temporal Graph Diagnostics for Elliptic (Node Classification setting)

Outputs:
- metrics/graph_temporal_metrics.csv  # פיצ'רים לכל צעד זמן
- plots/*.png                          # גרפים: סדרות זמן, ACF, FFT
- console summary                      # "המלצת מודל" לפי דפוסים אמפיריים

ת櫲
- מתאים ל-50 snapshots (כמו Elliptic). ברירת מחדל: מנתח רק קשתות תוך-צעדיות.
- אם יש צורך לכלול קשתות בין-צעדים (t→t+1), ראו הערה בסוף הסקריפט.
"""

import os
import math
import random
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import spearmanr, zscore
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL
import ruptures as rpt
import matplotlib.pyplot as plt

# =========================
# CONFIG (אפשר לשנות בקלות)
# =========================
RANDOM_SEED = 42
OUT_DIR_METRICS = "metrics"
OUT_DIR_PLOTS = "plots"

# חישובים "כבדים" – אפשר לכבות אם איטי:
COMPUTE_COMMUNITIES = True  # Greedy modularity (על גרף לא-מכוון)
COMPUTE_COMMUNITY_SIM = True  # ARI/NMI בין צעדים עוקבים (דורש קהילות)
APPROX_APL_DIAMETER = False  # חישוב מקורב של מרחק ממוצע/דיאמטר (עלול להיות איטי)
SAMPLE_NODES_FOR_APL = 30  # גודל דגימה למרחקים אם מאופשר

# מחזוריות/תנודתיות – אילו פיצ'רים לבדוק
FEATURES_FOR_PERIODICITY = [
    "n_nodes", "n_edges", "density_undirected",
    "avg_clustering", "mean_degree", "modularity",
    "jaccard_nodes", "jaccard_edges"
]

# כלל אצבע למחזוריות
ACF_PEAK_MIN = 0.30  # סף שיאה ב-ACF (ללא לג אפס)
MIN_FEATURES_WITH_SAME_LAG = 3  # כמה פיצ'רים צריכים להסכים על אותו לג

# כלל אצבע ל-churn גבוה
NEWV_RATE_HIGH = 0.30  # יחס צמתים חדשים גבוה
JV_LOW = 0.40  # חפיפה נמוכה בין צעדים

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ======================================================
# 1) קריאת נתונים (את כבר סיפקת — משלב כאן לשחזור מלא)
# ======================================================
nodes_column_names = ['txId', 'time_step'] + [f'feature_{i}' for i in range(1, 166)]

nodes_df = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_features.csv",
                       header=None, names=nodes_column_names)
edges_df = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")
labels_df = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_classes.csv")

nodes_df = nodes_df.merge(labels_df, on='txId', how='left')

# תיקון שמות עמודות ב-edges אם נדרש
if len(edges_df.columns) >= 2:
    src_col, dst_col = edges_df.columns[:2]
    edges_df = edges_df.rename(columns={src_col: "src", dst_col: "dst"})
else:
    raise ValueError("edges_df must have at least 2 columns.")

# לוודא טיפוסים
nodes_df['time_step'] = nodes_df['time_step'].astype(int)
nodes_df['txId'] = nodes_df['txId'].astype(str)
edges_df['src'] = edges_df['src'].astype(str)
edges_df['dst'] = edges_df['dst'].astype(str)

# מיפוי צומת→צעד זמן
tx_to_time = dict(zip(nodes_df['txId'], nodes_df['time_step']))

# ======================================================
# 2) בניית Snapshots וסטטיסטיקות לכל זמן t
# ======================================================
os.makedirs(OUT_DIR_METRICS, exist_ok=True)
os.makedirs(OUT_DIR_PLOTS, exist_ok=True)

unique_steps = sorted(nodes_df['time_step'].unique().tolist())

# הכנה: ל-label drift
label_col = None
for c in nodes_df.columns:
    if c.lower() in ("class", "label", "y", "target"):
        label_col = c
        break


def get_known_illicit_ratio(df_t):
    """יחס הדוגמאות ה'מוכרות' + יחס ה-illicit מתוך המוכרות."""
    if label_col is None:
        return np.nan, np.nan
    s = df_t[label_col].astype(str)
    # אליפטיק: בדרך כלל '1' illicit, '2' licit, 'unknown'/'-1'/NaN אם אין
    known_mask = s.isin(['1', '2', 'illicit', 'licit'])
    known_count = known_mask.sum()
    total = len(s)
    known_ratio = known_count / total if total > 0 else np.nan

    illicit_mask = s.isin(['1', 'illicit'])
    illicit_ratio_known = (illicit_mask & known_mask).sum() / known_count if known_count > 0 else np.nan
    return known_ratio, illicit_ratio_known


def approx_apl_and_diameter(Gu, k=SAMPLE_NODES_FOR_APL):
    """חישוב מקורב של מרחק ממוצע ודיאמטר ע"י דגימה."""
    if not APPROX_APL_DIAMETER:
        return np.nan, np.nan
    if Gu.number_of_nodes() == 0:
        return np.nan, np.nan
    nodes = list(Gu.nodes())
    k = min(k, len(nodes))
    sample = np.random.choice(nodes, size=k, replace=False)
    all_dists = []
    max_ecc = 0
    for v in sample:
        lengths = nx.single_source_shortest_path_length(Gu, v)
        if len(lengths) > 1:
            dists = list(lengths.values())
            all_dists.extend(dists)
            max_ecc = max(max_ecc, max(dists))
    apl = float(np.mean(all_dists)) if all_dists else np.nan
    diameter = float(max_ecc) if max_ecc > 0 else np.nan
    return apl, diameter


# לאחסון תוצאות לכל t
rows = []

# לשימוש חוזר עבור churn / ARI/NMI
prev_nodes = None
prev_edges = None
prev_deg_map = None
prev_comm_assign = None  # dict: node -> community id

for t in unique_steps:
    V_t = set(nodes_df.loc[nodes_df['time_step'] == t, 'txId'].astype(str))
    # קשתות תוך-צעדיות בלבד (src, dst ב-V_t ובאותו t)
    # אפשרות: לכלול רק קשתות שבהן לשני הצמתים אותו time_step.
    # לחלופין (לא כאן): לכלול קשתות עם max(time_src, time_dst) == t כדי "לצבור" מידע.
    edges_t = edges_df.copy()
    edges_t['t_src'] = edges_t['src'].map(tx_to_time)
    edges_t['t_dst'] = edges_t['dst'].map(tx_to_time)
    edges_t = edges_t[(edges_t['t_src'] == t) & (edges_t['t_dst'] == t)]
    E_t = set(map(tuple, edges_t[['src', 'dst']].values.tolist()))

    # גרף מכוון + גרף לא-מכוון (למדדים שונים)
    Gd = nx.DiGraph()
    Gd.add_nodes_from(V_t)
    Gd.add_edges_from(E_t)

    Gu = nx.Graph()
    Gu.add_nodes_from(V_t)
    Gu.add_edges_from(E_t)

    n = Gu.number_of_nodes()
    m = Gu.number_of_edges()

    density_u = nx.density(Gu) if n > 1 else np.nan
    avg_clust = nx.average_clustering(Gu) if n > 1 else np.nan

    # אסורטטיביות – על לא-מכוון (פשוט יותר להערכה מהירה)
    assort = np.nan
    try:
        if n > 2 and m > 0:
            assort = nx.degree_assortativity_coefficient(Gu)
    except Exception:
        assort = np.nan

    # סטטיסטיקות דרגות
    degs_u = np.array([d for _, d in Gu.degree()]) if n > 0 else np.array([])
    mean_deg = float(degs_u.mean()) if degs_u.size else np.nan
    std_deg = float(degs_u.std()) if degs_u.size else np.nan

    indeg = np.array([d for _, d in Gd.in_degree()]) if n > 0 else np.array([])
    outdeg = np.array([d for _, d in Gd.out_degree()]) if n > 0 else np.array([])
    mean_in = float(indeg.mean()) if indeg.size else np.nan
    mean_out = float(outdeg.mean()) if outdeg.size else np.nan

    # APL/דיאמטר מקורבים (גדול? כבוי כברירת מחדל)
    apl, diam = approx_apl_and_diameter(Gu)

    # קהילות + מודולריות
    num_comms = np.nan
    modularity = np.nan
    comm_assign = None
    if COMPUTE_COMMUNITIES and n > 0 and m > 0:
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            from networkx.algorithms.community.quality import modularity as nx_modularity

            comms = list(greedy_modularity_communities(Gu))
            num_comms = len(comms)
            modularity = nx_modularity(Gu, comms)

            # שיוך צומת→מספר קהילה לצורך ARI/NMI
            comm_assign = {}
            for idx, cset in enumerate(comms):
                for node in cset:
                    comm_assign[node] = idx
        except Exception:
            num_comms = np.nan
            modularity = np.nan
            comm_assign = None

    # Label drift (יחס ידועים + שיעור illicit מתוך הידועים)
    df_t = nodes_df.loc[nodes_df['time_step'] == t]
    known_ratio, illicit_ratio_known = get_known_illicit_ratio(df_t)

    # --- Churn מול t-1 ---
    jaccard_nodes = np.nan
    newV_rate = np.nan
    jaccard_edges = np.nan
    spearman_deg_corr = np.nan
    ari = np.nan
    nmi = np.nan

    if prev_nodes is not None:
        inter_nodes = V_t & prev_nodes
        union_nodes = V_t | prev_nodes
        if len(union_nodes) > 0:
            jaccard_nodes = len(inter_nodes) / len(union_nodes)
        if len(V_t) > 0:
            newV_rate = len(V_t - prev_nodes) / len(V_t)

        # Jaccard לקשתות (כאן על לא-מכוון נורמליזציה פשוטה)
        E_prev = prev_edges if prev_edges is not None else set()
        # כדי לא לערב כיוון, נוריד כיוון לקשתות
        und_t = set(tuple(sorted(e)) for e in E_t)
        und_prev = set(tuple(sorted(e)) for e in E_prev)
        inter_e = und_t & und_prev
        union_e = und_t | und_prev
        if len(union_e) > 0:
            jaccard_edges = len(inter_e) / len(union_e)

        # יציבות דרגות (Spearman) על הצמתים המשותפים
        if len(inter_nodes) > 1:
            deg_map_t = dict(Gu.degree())
            deg_map_prev = prev_deg_map if prev_deg_map is not None else {}
            d1 = []
            d2 = []
            for v in inter_nodes:
                d1.append(deg_map_prev.get(v, 0))
                d2.append(deg_map_t.get(v, 0))
            if len(d1) > 1 and len(d2) > 1:
                rho, _ = spearmanr(d1, d2)
                spearman_deg_corr = float(rho)

        # דמיון קהילתי בין צעדים (אם חושב)
        if COMPUTE_COMMUNITIES and COMPUTE_COMMUNITY_SIM and (prev_comm_assign is not None) and \
                (comm_assign is not None):
            common = [v for v in inter_nodes if v in prev_comm_assign and v in comm_assign]
            if len(common) > 5:
                y_prev = [prev_comm_assign[v] for v in common]
                y_curr = [comm_assign[v] for v in common]
                ari = adjusted_rand_score(y_prev, y_curr)
                nmi = normalized_mutual_info_score(y_prev, y_curr)

    # שמירה לשורה
    rows.append({
        "time_step": t,
        "n_nodes": n,
        "n_edges": m,
        "density_undirected": density_u,
        "avg_clustering": avg_clust,
        "assortativity": assort,
        "mean_degree": mean_deg,
        "std_degree": std_deg,
        "mean_in_degree": mean_in,
        "mean_out_degree": mean_out,
        "approx_avg_path_len": apl,
        "approx_diameter": diam,
        "num_communities": num_comms,
        "modularity": modularity,
        "known_label_ratio": known_ratio,
        "illicit_ratio_known": illicit_ratio_known,
        "jaccard_nodes": jaccard_nodes,
        "newV_rate": newV_rate,
        "jaccard_edges": jaccard_edges,
        "spearman_deg_corr": spearman_deg_corr,
        "comm_ARI_prev": ari,
        "comm_NMI_prev": nmi,
    })

    # עדכון לזיכרון לצעד הבא
    prev_nodes = V_t
    prev_edges = E_t
    prev_deg_map = dict(Gu.degree())
    prev_comm_assign = comm_assign

# מסגרת התוצאות
metrics_df = pd.DataFrame(rows).sort_values("time_step").reset_index(drop=True)

# שמירה ל-CSV
metrics_path = os.path.join(OUT_DIR_METRICS, "graph_temporal_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"[✓] Saved metrics to: {metrics_path}")


# ======================================================
# 3) מחזוריות (ACF) + FFT + STL + שרטוטים
# ======================================================
def plot_series(x, y, title, fname):
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel("time_step")
    plt.tight_layout()
    outp = os.path.join(OUT_DIR_PLOTS, fname)
    plt.savefig(outp, dpi=160)
    plt.close()


def plot_acf_series(y, title, fname, nlags=12):
    try:
        series = pd.Series(y).astype(float)
        series = series.replace([np.inf, -np.inf], np.nan).interpolate().bfill().ffill()
        vals = acf(series, nlags=min(nlags, len(series) - 2), fft=True)
        plt.figure(figsize=(10, 4))
        plt.stem(range(len(vals)), vals, use_line_collection=True)
        plt.title(f"ACF: {title}")
        plt.xlabel("lag")
        plt.tight_layout()
        outp = os.path.join(OUT_DIR_PLOTS, fname)
        plt.savefig(outp, dpi=160)
        plt.close()
        return vals
    except Exception:
        return None


def plot_fft_series(y, title, fname):
    try:
        series = pd.Series(y).astype(float)
        series = series.replace([np.inf, -np.inf], np.nan).interpolate().bfill().ffill()
        Y = np.fft.rfft(series - series.mean())
        freqs = np.fft.rfftfreq(len(series), d=1.0)
        power = (Y.real ** 2 + Y.imag ** 2)
        plt.figure(figsize=(10, 4))
        plt.plot(freqs[1:], power[1:])  # מתעלמים מתדר 0
        plt.title(f"FFT power: {title}")
        plt.xlabel("frequency")
        plt.tight_layout()
        outp = os.path.join(OUT_DIR_PLOTS, fname)
        plt.savefig(outp, dpi=160)
        plt.close()
    except Exception:
        pass


# צייר סדרות זמן בסיסיות
for col in ["n_nodes", "n_edges", "density_undirected", "avg_clustering",
            "modularity", "jaccard_nodes", "jaccard_edges", "newV_rate"]:
    if col in metrics_df:
        plot_series(metrics_df["time_step"], metrics_df[col], f"{col} over time", f"{col}_over_time.png")

# ACF + FFT לפיצ'רים שנבחרו
acf_lag_votes = defaultdict(int)
for col in FEATURES_FOR_PERIODICITY:
    if col not in metrics_df.columns:
        continue
    vals = metrics_df[col].values
    acf_vals = plot_acf_series(vals, col, f"acf_{col}.png")
    plot_fft_series(vals, col, f"fft_{col}.png")

    if acf_vals is not None and len(acf_vals) > 1:
        # חפש לג (>=1) עם מקסימום מעל סף
        lags = np.arange(1, len(acf_vals))
        peaks = lags[acf_vals[1:] >= ACF_PEAK_MIN]
        if len(peaks) > 0:
            # קחי את הלג עם הערך הגבוה ביותר
            best_lag = lags[1:][np.argmax(acf_vals[2:])] if len(acf_vals) > 3 else peaks[0]
            # אם best_lag לא עבר את הסף, קחי את הראשון שעבר
            if acf_vals[best_lag] < ACF_PEAK_MIN:
                best_lag = peaks[0]
            acf_lag_votes[int(best_lag)] += 1

# STL על 2–3 פיצ'רים מייצגים
for col in ["n_nodes", "density_undirected", "modularity"]:
    if col in metrics_df.columns:
        series = metrics_df[col].astype(float)
        series = series.replace([np.inf, -np.inf], np.nan).interpolate().bfill().ffill()
        try:
            stl = STL(series, period=7, robust=True).fit()
            plt.figure(figsize=(10, 6))
            plt.subplot(3, 1, 1);
            plt.plot(series);
            plt.title(f"{col} - original")
            plt.subplot(3, 1, 2);
            plt.plot(stl.trend);
            plt.title("trend")
            plt.subplot(3, 1, 3);
            plt.plot(stl.seasonal);
            plt.title("seasonal")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR_PLOTS, f"stl_{col}.png"), dpi=160)
            plt.close()
        except Exception:
            pass

# ======================================================
# 4) Change-point detection (רב-משתני)
# ======================================================
# משתמשים בכמה עמודות מבניות כמ矲
cp_cols = ["n_nodes", "n_edges", "density_undirected", "avg_clustering", "modularity"]
cp_used = [c for c in cp_cols if c in metrics_df.columns]
if len(cp_used) >= 2:
    X = metrics_df[cp_used].copy()
    X = X.apply(lambda s: pd.to_numeric(s, errors="coerce")).replace([np.inf, -np.inf], np.nan)
    X = X.interpolate().bfill().ffill()
    Xz = X.apply(zscore, nan_policy='omit').replace([np.inf, -np.inf], 0.0).values

    # PELT עם cost='rbf' – בחרי penalty לפי טעם; כאן ערך מתון
    model = rpt.Pelt(model="rbf", min_size=3, jump=1)
    model.fit(Xz)
    change_idx = model.predict(pen=5)  # החזרת אינדקסים אחרי נק' שינוי
    change_steps = [metrics_df.loc[i - 1, "time_step"] for i in change_idx if (i - 1) in metrics_df.index]

    metrics_df["is_change_point"] = metrics_df["time_step"].isin(change_steps).astype(int)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[✓] Change-points (heuristic): steps {change_steps}")

# ======================================================
# 5) סיכום והמלצת מודל (EvolveGCN-O vs DySAT)
# ======================================================
# בדיקת מחזוריות: האם יש לג דומיננטי שמופיע אצל >= MIN_FEATURES_WITH_SAME_LAG פיצ'רים?
dominant_lag = None
if len(acf_lag_votes):
    lag, votes = max(acf_lag_votes.items(), key=lambda kv: kv[1])
    if votes >= MIN_FEATURES_WITH_SAME_LAG:
        dominant_lag = lag

# מדדי churn גלובליים
median_jv = metrics_df["jaccard_nodes"].median(skipna=True)
high_newv_ratio = (metrics_df["newV_rate"] >= NEWV_RATE_HIGH).mean()  # שיעור צעדים עם newV גבוה

print("\n========== SUMMARY ==========")
print(f"Dominant periodic lag (ACF): {dominant_lag}  "
      f"(votes≥{MIN_FEATURES_WITH_SAME_LAG}? {'YES' if dominant_lag is not None else 'NO'})")
print(f"Median Jaccard (nodes): {median_jv:.3f}   (low if < {JV_LOW})")
print(f"Share of steps with newV_rate ≥ {NEWV_RATE_HIGH:.2f}: {high_newv_ratio:.2f}")
print(f"Metrics CSV: {metrics_path}")
print(f"Plots dir:  {OUT_DIR_PLOTS}")

recommendation = None
reasons = []

# כללי החלטה פשוטים:
if (dominant_lag is not None) and (median_jv is not None) and (median_jv >= JV_LOW):
    recommendation = "DySAT"
    reasons.append("מחזוריות/עונתיות מזוהה בכמה פיצ'רים, וחפיפת צמתים בין צעדים לא נמוכה במיוחד.")
elif (median_jv is not None and median_jv < JV_LOW) or (high_newv_ratio is not None and high_newv_ratio >= 0.4):
    recommendation = "EvolveGCN-O"
    reasons.append("Churn גבוה (חפיפה נמוכה/יחס צמתים חדשים גבוה) – סט הצמתים משתנה משמעותית לאורך זמן.")
else:
    # ברירת מחדל: להתחיל עם EvolveGCN-O כקו בסיס אינדוקטיבי, להשוות מול DySAT
    recommendation = "EvolveGCN-O (start) → compare with DySAT"
    reasons.append("לא זוהתה מחזוריות חזקה; כדאי להתחיל במודל אינדוקטיבי ולבדוק השוואה.")

print(f"\nModel recommendation: {recommendation}")
for r in reasons:
    print(f" - {r}")

# ===============================
# הערה: איך לכלול קשתות בין-צעדים
# ===============================
# במקום לסנן edges_t על t_src==t_dst==t, אפשר להגדיר "חלון" ולכלול קשתות שעבורן
# max(t_src, t_dst) == t  (או בתוך חלון {t-w, ..., t}), כדי לייצג הצטברות היסטורית.
# לשם כך, החליפי את הסינון ל:
#
# edges_t = edges_df.copy()
# edges_t['t_src'] = edges_t['src'].map(tx_to_time)
# edges_t['t_dst'] = edges_t['dst'].map(tx_to_time)
# edges_t = edges_t[np.maximum(edges_t['t_src'], edges_t['t_dst']) == t]
# V_t = set(edges_t['src']).union(edges_t['dst'])  # נגזר מהקשתות בחלון
#
# ואז המשיכי כרגיל. זה שימושי אם תרצי "גרף מצטבר" (cumulative) לכל צעד.
