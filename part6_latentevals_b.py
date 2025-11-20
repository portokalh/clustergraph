#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PART 6 — LATENT EVALUATION FOR A GIVEN K
========================================
Clean + debugged version (2025-11-20)

Evaluates latent spaces for:
    • CORR latent
    • EUCLID latent
    • WASS latent

Outputs:
    plots/
        UMAP_*.png
        PCA_*.png
        latent_trait_corr_heatmap.png
        cluster_trait_bars_*.png

    tables/
        latent_trait_correlations.csv
        knn_classification_results.csv
        cluster_trait_composition_kmeans.csv
        cluster_trait_composition_consensus.csv
"""

# ================================================================
# IMPORTS
# ================================================================
import os
import numpy as np
import pandas as pd
import torch
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Silence pandas downcasting warning (your earlier issue)
pd.set_option("future.no_silent_downcasting", True)

# ================================================================
# CONFIGURATION
# ================================================================
GRAPH_K = 25
METRICS = ["CORR", "EUCLID", "WASS"]
FOCUS_METRIC = "EUCLID"

ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation"
COLUMNS_ROOT = os.path.join(ROOT, "columns4gaudi111825", "columna-analyses111925")
MDATA_PATH = os.path.join(ROOT, "processed_graph_data", "metadata_with_PCs.xlsx")

EVAL_ROOT = os.path.join(COLUMNS_ROOT, f"latent_eval_K{GRAPH_K}")
PLOT_DIR = os.path.join(EVAL_ROOT, "plots")
TABLE_DIR = os.path.join(EVAL_ROOT, "tables")

for d in [EVAL_ROOT, PLOT_DIR, TABLE_DIR]:
    os.makedirs(d, exist_ok=True)

RES_DIR = os.path.join(COLUMNS_ROOT, f"results_K{GRAPH_K}")

TRAIT_COLS_CONT = ["age", "BMI"]
TRAIT_COLS_CAT = ["APOE", "risk_for_ad", "sex"]

# ================================================================
# HELPERS
# ================================================================
def compute_umap(Z, n_neighbors=15, min_dist=0.05):
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, Z.shape[0] - 1),
        min_dist=min_dist,
        spread=1.0,
        n_components=2,
        random_state=42,
    )
    return reducer.fit_transform(Z)


def knn_latent_classification(Z, labels, n_neighbors=5):
    labels = np.asarray(labels)

    mask = labels != "NA"
    Z_valid = Z[mask]
    y = labels[mask]

    if len(np.unique(y)) < 2 or len(y) < 10:
        return np.nan, np.nan

    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z_valid)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []

    for train, test in skf.split(Zs, y):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(Zs[train], y[train])
        y_pred = clf.predict(Zs[test])
        accs.append(accuracy_score(y[test], y_pred))
        f1s.append(f1_score(y[test], y_pred, average="macro"))

    return float(np.mean(accs)), float(np.mean(f1s))


def safe_pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 5:
        return np.nan
    try:
        r, _ = pearsonr(x[mask], y[mask])
        return float(r)
    except Exception:
        return np.nan


# ================================================================
# LOAD METADATA AND ALIGN TO GRAPHS
# ================================================================
print("Loading metadata from:", MDATA_PATH)
df_all = pd.read_excel(MDATA_PATH)
df_all["MRI_Exam"] = df_all["MRI_Exam"].astype(str).str.zfill(5)

GRAPHS_PT = os.path.join(
    COLUMNS_ROOT, "graphs_knn", f"k{GRAPH_K}",
    f"md_shared_knn_k{GRAPH_K}_corr_euclid_wass.pt"
)

print("Loading graphs from:", GRAPHS_PT)
graphs = torch.load(GRAPHS_PT, map_location="cpu")
subject_ids = [str(g.subject_id).zfill(5) for g in graphs]

df_sub = df_all[df_all["MRI_Exam"].isin(subject_ids)].copy()
df_sub["__order"] = df_sub["MRI_Exam"].apply(lambda s: subject_ids.index(s))
df_sub = df_sub.sort_values("__order").reset_index(drop=True)
df_sub.drop(columns="__order", inplace=True)
df_sub["MRI_Exam"] = df_sub["MRI_Exam"].astype(str)

print("Aligned metadata shape:", df_sub.shape)

# ================================================================
# LOAD LATENTS
# ================================================================
Z_dict = {}
for metric in METRICS:
    path = os.path.join(
        COLUMNS_ROOT, f"latent_k{GRAPH_K}",
        f"latent_epochs_Joint_{metric}",
        f"latent_final_Joint_{metric}.npy"
    )
    print(f"Loading {metric} latents:", path)
    Z = np.load(path)

    keep = np.isin(subject_ids, df_sub["MRI_Exam"].values)
    Z = Z[keep]
    assert Z.shape[0] == df_sub.shape[0]

    Z_dict[metric] = Z

latent_dim = Z_dict[FOCUS_METRIC].shape[1]


# ================================================================
# PART 1 — UMAP + PCA PLOTS
# ================================================================
def plot_umap(Z, df, metric, factor):
    emb = compute_umap(StandardScaler().fit_transform(Z))
    out = os.path.join(PLOT_DIR, f"UMAP_{metric}_{factor}.png")

    fig, ax = plt.subplots(figsize=(5,4), dpi=250)

    if factor in TRAIT_COLS_CONT:
        vals = pd.to_numeric(df[factor], errors="coerce")
        sc = ax.scatter(emb[:,0], emb[:,1], c=vals, cmap="viridis",
                        s=40, edgecolor="none", alpha=0.9)
        plt.colorbar(sc, ax=ax)
    else:
        labels = df[factor].astype(str)
        cats = pd.Categorical(labels)
        sc = ax.scatter(emb[:,0], emb[:,1], c=cats.codes, cmap="tab10",
                        s=40, edgecolor="k", linewidth=0.2)
        handles = [plt.Line2D([0],[0], marker='o',
                              color=plt.get_cmap("tab10")(i),
                              linestyle='', label=str(cat))
                   for i, cat in enumerate(cats.categories)]
        ax.legend(handles=handles, title=factor)

    ax.set_title(f"K={GRAPH_K}, {metric} UMAP by {factor}")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out, dpi=250)
    plt.close()
    print("Saved", out)


def plot_pca(Z, df, metric, factor):
    Zs = StandardScaler().fit_transform(Z)
    pcs = PCA(n_components=2).fit_transform(Zs)
    out = os.path.join(PLOT_DIR, f"PCA_{metric}_{factor}.png")

    fig, ax = plt.subplots(figsize=(5,4), dpi=250)

    if factor in TRAIT_COLS_CONT:
        vals = pd.to_numeric(df[factor], errors="coerce")
        sc = ax.scatter(pcs[:,0], pcs[:,1], c=vals, cmap="viridis",
                        s=40, edgecolor="none", alpha=0.9)
        plt.colorbar(sc, ax=ax)
    else:
        labels = df[factor].astype(str)
        cats = pd.Categorical(labels)
        sc = ax.scatter(pcs[:,0], pcs[:,1], c=cats.codes, cmap="tab10",
                        s=40, edgecolor="k", linewidth=0.2)

    ax.set_title(f"K={GRAPH_K}, {metric} PCA by {factor}")
    plt.tight_layout()
    plt.savefig(out, dpi=250)
    plt.close()
    print("Saved", out)


for metric, Z in Z_dict.items():
    for fac in ["APOE", "risk_for_ad", "age", "BMI"]:
        plot_umap(Z, df_sub, metric, fac)
        plot_pca(Z, df_sub, metric, fac)


# ================================================================
# PART 2 — kNN CLASSIFICATION
# ================================================================
rows = []
for metric, Z in Z_dict.items():
    for fac in ["APOE", "risk_for_ad"]:
        labels = df_sub[fac].astype(str).replace({"nan":"NA","NaN":"NA"})
        acc,f1 = knn_latent_classification(StandardScaler().fit_transform(Z), labels)
        rows.append([GRAPH_K, metric, fac, acc, f1])

df_knn = pd.DataFrame(rows, columns=["K","Metric","Factor","Accuracy","MacroF1"])
df_knn.to_csv(os.path.join(TABLE_DIR, "knn_classification_results.csv"), index=False)
print("Saved kNN classification table.")


# ================================================================
# PART 3 — LATENT–TRAIT CORRELATIONS
# ================================================================
Zf = StandardScaler().fit_transform(Z_dict[FOCUS_METRIC])
corr_rows = []

for d in range(latent_dim):
    vec = Zf[:,d]

    row = {
        "K":GRAPH_K,
        "Metric":FOCUS_METRIC,
        "Latent_dim":d+1,
        "r_age":safe_pearson(vec, df_sub["age"]),
        "r_BMI":safe_pearson(vec, df_sub["BMI"]),
    }

    # APOE → numeric
    ap_bin = df_sub["APOE"].replace({"E4-":0,"E4+":1}).astype(float)
    row["r_APOE_bin"] = safe_pearson(vec, ap_bin)

    risk_ord = df_sub["risk_for_ad"].astype(str).replace(
        {"0":0,"1":1,"2":2,"3":3}).astype(float)
    row["r_risk_ord"] = safe_pearson(vec, risk_ord)

    corr_rows.append(row)

df_corr = pd.DataFrame(corr_rows)
df_corr.to_csv(os.path.join(TABLE_DIR, "latent_trait_correlations.csv"), index=False)

plt.figure(figsize=(6,6), dpi=250)
sns.heatmap(df_corr.set_index("Latent_dim").iloc[:,2:].abs(), cmap="viridis")
plt.title(f"Latent–Trait |r| heatmap (K={GRAPH_K}, {FOCUS_METRIC})")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "latent_trait_corr_heatmap.png"), dpi=250)
plt.close()


# ================================================================
# PART 4 — CLUSTER COMPOSITION (K-means + Consensus)
# ================================================================
def cluster_trait_composition(df_labels, cluster_col, name_prefix):
    df_labels["MRI_Exam"] = df_labels["MRI_Exam"].astype(str)
    df_m = pd.merge(df_sub, df_labels, on="MRI_Exam", how="inner")

    results = []
    for cl in sorted(df_m[cluster_col].unique()):
        sub = df_m[df_m[cluster_col] == cl]
        row = {
            "cluster": cl,
            "n": len(sub),
            "prefix": name_prefix
        }
        for fac in ["APOE","risk_for_ad","sex"]:
            vc = sub[fac].astype(str).value_counts(normalize=True)
            for cat,p in vc.items():
                row[f"{fac}_{cat}"] = round(float(p),3)
        results.append(row)

    return pd.DataFrame(results)


# -------- Load cluster assignments --------
kmeans_csv = os.path.join(RES_DIR, f"kmeans_cluster_assignments_K{GRAPH_K}.csv")
consensus_csv = os.path.join(RES_DIR, f"consensus_clusters_K{GRAPH_K}.csv")

df_kmeans = pd.read_csv(kmeans_csv)
df_kmeans["MRI_Exam"] = df_kmeans["MRI_Exam"].astype(str)

df_cons = pd.read_csv(consensus_csv)
df_cons["MRI_Exam"] = df_cons["MRI_Exam"].astype(str)

# ---- K-means composition ----
out = []
for metric in METRICS:
    sub = df_kmeans[df_kmeans["Distance"] == metric][["MRI_Exam","Cluster"]]
    if sub.empty: continue
    sub = sub.rename(columns={"Cluster":"ClusterID"})
    out.append(cluster_trait_composition(sub, "ClusterID", f"kmeans_{metric}"))

df_ck = pd.concat(out)
df_ck.to_csv(os.path.join(TABLE_DIR,"cluster_trait_composition_kmeans.csv"), index=False)

# ---- Consensus composition ----
df_cons2 = df_cons.rename(columns={"ConsensusCluster":"ClusterID"})[["MRI_Exam","ClusterID"]]
df_cc = cluster_trait_composition(df_cons2, "ClusterID", "consensus")
df_cc.to_csv(os.path.join(TABLE_DIR,"cluster_trait_composition_consensus.csv"), index=False)

print("\n======================")
print(" DONE — Part 6")
print("Outputs:", EVAL_ROOT)
print("======================")
