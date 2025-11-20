#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PART 6 - LATENT EVALUATION & CLUSTER STATS FOR K=25
===================================================

This script evaluates GAUDI latent spaces and clustering for K=25
across three distance metrics: CORR, EUCLID, WASS.

It produces:

  - UMAP / PCA by traits (APOE, risk_for_ad, age, BMI)
  - kNN classification performance (APOE, risk_for_ad)
  - Latent–trait correlation heatmap (age, BMI, APOE-bin, risk_for_ad-ordinal)
  - Cluster trait composition (APOE, risk_for_ad, sex) for:
        * KMeans clusters per metric
        * Consensus clusters (all metrics)
  - WASS cluster enrichment barplots (APOE, sex, risk_for_ad)
  - APOE × Cluster contingency heatmaps
  - "Nature grid" UMAPs with KDE background (metrics × traits)
  - Family dendrogram in CORR latent space (if 'Family' present)
  - 1000-bootstrap AUC for APOE, sex, risk_for_ad across metrics
  - PCA & UMAP with KMeans clusters overlaid
  - Cluster quality metrics (silhouette, Dunn, CH, DBI, within-cluster distances)
  - Multinomial logistic regression coefficients for cluster prediction
  - Chi-square + Cramer's V tests for cluster × trait association
        * KMeans (per metric)
        * Consensus (all metrics pooled)

Outputs:

  latent_eval_K25/
      plots/
      tables/

Edit ROOT and paths below if your folder layout differs.
"""

# ================================================================
# IMPORTS
# ================================================================
import os
import math
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from scipy.stats import pearsonr, gaussian_kde, chi2_contingency
from scipy.cluster.hierarchy import linkage, dendrogram

# pandas future warning
pd.set_option("future.no_silent_downcasting", True)
sns.set(style="whitegrid")

# ================================================================
# CONFIGURATION
# ================================================================
GRAPH_K = 25
METRICS = ["CORR", "EUCLID", "WASS"]
FOCUS_METRIC = "EUCLID"   # for latent–trait correlations

# ---- Base paths: adjust if needed ----
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
TRAIT_COLS_CAT  = ["APOE", "risk_for_ad", "sex"]

# Extra traits for "Nature grid" if available in metadata
GRID_TRAITS = [
    "APOE", "genotype", "sex", "risk_for_ad", "ethnicity", "Family", "age", "BMI"
]

# ================================================================
# HELPER FUNCTIONS
# ================================================================
def compute_umap(Z, n_neighbors=15, min_dist=0.05, random_state=42):
    """2D UMAP embedding with conservative defaults."""
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, max(2, Z.shape[0] - 1)),
        min_dist=min_dist,
        spread=1.0,
        n_components=2,
        random_state=random_state,
    )
    return reducer.fit_transform(Z)


def knn_latent_classification(Z, labels, n_neighbors=5):
    """5-fold CV kNN classification for categorical labels (strings)."""
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
    """Pearson r with NaN-safe masking and minimum sample count."""
    x, y = np.asarray(x), np.asarray(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 5:
        return np.nan
    try:
        r, _ = pearsonr(x[mask], y[mask])
        return float(r)
    except Exception:
        return np.nan


def kde_background(ax, emb, levels=10):
    """Draw Gaussian KDE contour background behind scatter."""
    x, y = emb[:, 0], emb[:, 1]
    if len(x) < 10:
        return
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except Exception:
        return
    xmin, xmax = x.min() - 0.5, x.max() + 0.5
    ymin, ymax = y.min() - 0.5, y.max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 80),
        np.linspace(ymin, ymax, 80),
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contour(
        xx,
        yy,
        zz,
        levels=levels,
        colors="lightgray",
        linewidths=0.7,
        alpha=0.7,
    )


def bootstrap_auc(Z, labels, n_boot=1000, random_state=42):
    """
    Bootstrap ROC-AUC with logistic regression on latent Z.

    labels: binary (0/1) or string convertible to two classes.
    Returns mean AUC, lower/upper 95% CI, and all bootstrap values.
    """
    rng = np.random.default_rng(random_state)

    y = np.asarray(labels)
    # Encode as 0/1 if not numeric
    if y.dtype.kind not in ("i", "u", "f"):
        cats = pd.Categorical(y)
        if len(cats.categories) != 2:
            return np.nan, np.nan, np.nan, np.array([])
        y = (cats.codes == 1).astype(int)
    else:
        unique_vals = np.unique(y[~pd.isna(y)])
        if len(unique_vals) != 2:
            return np.nan, np.nan, np.nan, np.array([])
        y_bin = np.zeros_like(y, dtype=int)
        y_bin[y == unique_vals.max()] = 1
        y = y_bin

    mask = ~np.isnan(y)
    Z = Z[mask]
    y = y[mask]

    if len(np.unique(y)) < 2 or len(y) < 10:
        return np.nan, np.nan, np.nan, np.array([])

    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z)

    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)   # resample with replacement
        Xb, yb = Zs[idx], y[idx]

        clf = LogisticRegression(
            solver="liblinear",
            max_iter=200,
            class_weight="balanced",
        )
        clf.fit(Xb, yb)
        y_prob = clf.predict_proba(Zs)[:, 1]   # evaluate on full set
        aucs.append(roc_auc_score(y, y_prob))

    aucs = np.array(aucs)
    mean_auc = float(np.mean(aucs))
    lo, hi = np.percentile(aucs, [2.5, 97.5])

    return mean_auc, float(lo), float(hi), aucs


# ================================================================
# LOAD METADATA AND ALIGN TO GRAPHS
# ================================================================
print("Loading metadata from:", MDATA_PATH)
df_all = pd.read_excel(MDATA_PATH)
df_all["MRI_Exam"] = df_all["MRI_Exam"].astype(str).str.zfill(5)

GRAPHS_PT = os.path.join(
    COLUMNS_ROOT,
    "graphs_knn",
    f"k{GRAPH_K}",
    f"md_shared_knn_k{GRAPH_K}_corr_euclid_wass.pt",
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
        COLUMNS_ROOT,
        f"latent_k{GRAPH_K}",
        f"latent_epochs_Joint_{metric}",
        f"latent_final_Joint_{metric}.npy",
    )
    print(f"Loading {metric} latents:", path)
    Z = np.load(path)

    keep = np.isin(subject_ids, df_sub["MRI_Exam"].values)
    Z = Z[keep]
    assert Z.shape[0] == df_sub.shape[0]

    Z_dict[metric] = Z

latent_dim = Z_dict[FOCUS_METRIC].shape[1]

# ================================================================
# PART 1 — UMAP + PCA BY TRAITS
# ================================================================
def plot_umap(Z, df, metric, factor):
    emb = compute_umap(StandardScaler().fit_transform(Z))
    out = os.path.join(PLOT_DIR, f"UMAP_{metric}_{factor}.png")

    fig, ax = plt.subplots(figsize=(5, 4), dpi=250)
    kde_background(ax, emb)

    if factor in TRAIT_COLS_CONT:
        vals = pd.to_numeric(df[factor], errors="coerce")
        sc = ax.scatter(
            emb[:, 0],
            emb[:, 1],
            c=vals,
            cmap="viridis",
            s=40,
            edgecolor="none",
            alpha=0.9,
        )
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label(factor)
    else:
        labels = df[factor].astype(str)
        cats = pd.Categorical(labels)
        ax.scatter(
            emb[:, 0],
            emb[:, 1],
            c=cats.codes,
            cmap="tab10",
            s=40,
            edgecolor="k",
            linewidth=0.2,
        )
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=plt.get_cmap("tab10")(i),
                markersize=6,
                label=str(cat),
            )
            for i, cat in enumerate(cats.categories)
        ]
        ax.legend(handles=handles, title=factor, fontsize=7)

    ax.set_title(f"K={GRAPH_K}, {metric} UMAP by {factor}")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out, dpi=250)
    plt.close()
    print("Saved", out)


def plot_pca(Z, df, metric, factor):
    Zs = StandardScaler().fit_transform(Z)
    pcs = PCA(n_components=2).fit_transform(Zs)
    out = os.path.join(PLOT_DIR, f"PCA_{metric}_{factor}.png")

    fig, ax = plt.subplots(figsize=(5, 4), dpi=250)

    if factor in TRAIT_COLS_CONT:
        vals = pd.to_numeric(df[factor], errors="coerce")
        sc = ax.scatter(
            pcs[:, 0],
            pcs[:, 1],
            c=vals,
            cmap="viridis",
            s=40,
            edgecolor="none",
            alpha=0.9,
        )
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label(factor)
    else:
        labels = df[factor].astype(str)
        cats = pd.Categorical(labels)
        ax.scatter(
            pcs[:, 0],
            pcs[:, 1],
            c=cats.codes,
            cmap="tab10",
            s=40,
            edgecolor="k",
            linewidth=0.2,
        )

    ax.set_title(f"K={GRAPH_K}, {metric} PCA by {factor}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out, dpi=250)
    plt.close()
    print("Saved", out)


for metric, Z in Z_dict.items():
    for fac in ["APOE", "risk_for_ad", "age", "BMI"]:
        if fac in df_sub.columns:
            plot_umap(Z, df_sub, metric, fac)
            plot_pca(Z, df_sub, metric, fac)

# ================================================================
# PART 2 — kNN CLASSIFICATION
# ================================================================
rows = []
for metric, Z in Z_dict.items():
    for fac in ["APOE", "risk_for_ad"]:
        if fac not in df_sub.columns:
            continue
        labels = df_sub[fac].astype(str).replace({"nan": "NA", "NaN": "NA"})
        acc, f1 = knn_latent_classification(
            StandardScaler().fit_transform(Z), labels
        )
        rows.append([GRAPH_K, metric, fac, acc, f1])

df_knn = pd.DataFrame(
    rows, columns=["K", "Metric", "Factor", "Accuracy", "MacroF1"]
)
df_knn.to_csv(
    os.path.join(TABLE_DIR, "knn_classification_results.csv"), index=False
)
print("Saved kNN classification table.")

# ================================================================
# PART 3 — LATENT–TRAIT CORRELATIONS (FOCUS_METRIC)
# ================================================================
Zf = StandardScaler().fit_transform(Z_dict[FOCUS_METRIC])
corr_rows = []

for d in range(latent_dim):
    vec = Zf[:, d]

    row = {
        "K": GRAPH_K,
        "Metric": FOCUS_METRIC,
        "Latent_dim": d + 1,
        "r_age": safe_pearson(vec, df_sub["age"]),
        "r_BMI": safe_pearson(vec, df_sub["BMI"]),
    }

    # APOE → numeric 0/1 if available
    if "APOE" in df_sub.columns:
        ap_bin = df_sub["APOE"].replace({"E4-": 0, "E4+": 1}).astype(float)
        row["r_APOE_bin"] = safe_pearson(vec, ap_bin)
    else:
        row["r_APOE_bin"] = np.nan

    if "risk_for_ad" in df_sub.columns:
        risk_ord = (
            df_sub["risk_for_ad"]
            .astype(str)
            .replace({"0": 0, "1": 1, "2": 2, "3": 3})
            .astype(float)
        )
        row["r_risk_ord"] = safe_pearson(vec, risk_ord)
    else:
        row["r_risk_ord"] = np.nan

    corr_rows.append(row)

df_corr = pd.DataFrame(corr_rows)
df_corr.to_csv(
    os.path.join(TABLE_DIR, "latent_trait_correlations.csv"), index=False
)

plt.figure(figsize=(6, 6), dpi=250)
sns.heatmap(
    df_corr.set_index("Latent_dim")
    .loc[:, ["r_age", "r_BMI", "r_APOE_bin", "r_risk_ord"]]
    .abs(),
    cmap="viridis",
)
plt.title(f"Latent–Trait |r| heatmap (K={GRAPH_K}, {FOCUS_METRIC})")
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_DIR, "latent_trait_corr_heatmap.png"), dpi=250
)
plt.close()

# ================================================================
# LOAD CLUSTER ASSIGNMENTS (KMeans + Consensus)
# ================================================================
kmeans_csv = os.path.join(
    RES_DIR, f"kmeans_cluster_assignments_K{GRAPH_K}.csv"
)
consensus_csv = os.path.join(RES_DIR, f"consensus_clusters_K{GRAPH_K}.csv")

df_kmeans = pd.read_csv(kmeans_csv)
df_kmeans["MRI_Exam"] = df_kmeans["MRI_Exam"].astype(str)

df_cons = pd.read_csv(consensus_csv)
df_cons["MRI_Exam"] = df_cons["MRI_Exam"].astype(str)

# ================================================================
# PART 4 — CLUSTER COMPOSITION (KMeans + Consensus)
# ================================================================
def cluster_trait_composition(df_labels, cluster_col, name_prefix):
    df_labels = df_labels.copy()
    df_labels["MRI_Exam"] = df_labels["MRI_Exam"].astype(str)
    df_m = pd.merge(df_sub, df_labels, on="MRI_Exam", how="inner")

    results = []
    for cl in sorted(df_m[cluster_col].unique()):
        sub = df_m[df_m[cluster_col] == cl]
        row = {"cluster": cl, "n": len(sub), "prefix": name_prefix}
        for fac in ["APOE", "risk_for_ad", "sex"]:
            if fac not in sub.columns:
                continue
            vc = sub[fac].astype(str).value_counts(normalize=True)
            for cat, p in vc.items():
                row[f"{fac}_{cat}"] = round(float(p), 3)
        results.append(row)

    return pd.DataFrame(results)


# ---- K-means composition ----
out = []
for metric in METRICS:
    sub = df_kmeans[df_kmeans["Distance"] == metric][["MRI_Exam", "Cluster"]]
    if sub.empty:
        continue
    sub = sub.rename(columns={"Cluster": "ClusterID"})
    out.append(cluster_trait_composition(sub, "ClusterID", f"kmeans_{metric}"))

df_ck = pd.concat(out)
df_ck.to_csv(
    os.path.join(TABLE_DIR, "cluster_trait_composition_kmeans.csv"),
    index=False,
)

# ---- Consensus composition ----
df_cons2 = df_cons.rename(columns={"ConsensusCluster": "ClusterID"})[
    ["MRI_Exam", "ClusterID"]
]
df_cc = cluster_trait_composition(df_cons2, "ClusterID", "consensus")
df_cc.to_csv(
    os.path.join(TABLE_DIR, "cluster_trait_composition_consensus.csv"),
    index=False,
)

# ================================================================
# PART 5 — ENRICHMENT BARS (WASS, KMeans) + APOE×CLUSTER HEATMAPS
# ================================================================
def plot_wass_enrichment_bars(df_kmeans, df_sub):
    """
    Barplots for WASS (KMeans) clusters: APOE, sex, risk_for_ad.
    Robust to column naming issues; always ensures a 'Cluster' column.
    """
    df_w = df_kmeans[df_kmeans["Distance"] == "WASS"].copy()
    if df_w.empty:
        print("No WASS rows in kmeans assignments.")
        return

    df_w = df_w.rename(columns={
        "ClusterID": "Cluster",
        "cluster": "Cluster",
        "Label": "Cluster",
    })
    if "Cluster" not in df_w.columns:
        print("ERROR: Could not find a 'Cluster' column in WASS subset:", df_w.columns)
        return

    df_w["MRI_Exam"] = df_w["MRI_Exam"].astype(str)
    df_m = df_sub.merge(df_w["MRI_Exam Cluster".split()], on="MRI_Exam", how="inner")
    df_m["Cluster"] = df_m["Cluster"].astype(str)

    factors = ["APOE", "sex", "risk_for_ad"]
    titles = {
        "APOE": "APOE enrichment across clusters (WASS, KMeans)",
        "sex": "sex enrichment across clusters (WASS, KMeans)",
        "risk_for_ad": "risk_for_ad enrichment across clusters (WASS, KMeans)",
    }

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), dpi=250)

    for ax, fac in zip(axes, factors):
        if fac not in df_m.columns:
            ax.axis("off")
            continue

        tmp = df_m[["Cluster", fac]].copy()
        tmp[fac] = tmp[fac].astype(str)

        counts = (
            tmp.groupby([fac, "Cluster"])["Cluster"]
            .count()
            .rename("n")
            .reset_index()
        )
        counts["Proportion"] = counts.groupby(fac)["n"].transform(
            lambda x: x / x.sum()
        )

        sns.barplot(
            data=counts,
            x=fac,
            y="Proportion",
            hue="Cluster",
            ax=ax,
            palette="tab10",
        )
        ax.set_ylim(0, 1.05)
        ax.set_title(titles[fac])
        ax.legend(title="Cluster", fontsize=7)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "kmeans_cluster_bars_WASS.png")
    plt.savefig(out, dpi=250)
    plt.close()
    print("Saved", out)


def plot_apoe_cluster_heatmaps(df_kmeans, df_sub):
    """
    APOE × Cluster heatmaps for CORR, EUCLID, WASS (KMeans).
    This version is fully safe against empty merges or plotting errors.
    """

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 10), dpi=250)

    for ax, metric in zip(axes, METRICS):

        # Subset to this metric
        sub = df_kmeans[df_kmeans["Distance"] == metric][["MRI_Exam", "Cluster"]].copy()
        sub["MRI_Exam"] = sub["MRI_Exam"].astype(str).str.zfill(5)

        # Merge with APOE
        df_m = pd.merge(
            df_sub[["MRI_Exam", "APOE"]],
            sub,
            on="MRI_Exam",
            how="inner"
        )

        # If no rows, skip
        if df_m.empty:
            ax.set_title(f"{metric}: No merge between APOE and cluster")
            ax.axis("off")
            continue

        # Crosstab
        ct = pd.crosstab(df_m["Cluster"], df_m["APOE"])

        # Skip empty crosstab
        if ct.empty or ct.shape[0] == 0 or ct.shape[1] == 0:
            ax.set_title(f"{metric}: Empty crosstab")
            ax.axis("off")
            continue

        # Plot heatmap
        sns.heatmap(
            ct,
            annot=True,
            fmt="d",
            cmap="viridis",
            cbar=True,
            ax=ax
        )

        ax.set_title(f"{metric} — APOE × Cluster (KMeans)")
        ax.set_xlabel("APOE")
        ax.set_ylabel("Cluster")

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "kmeans_cluster_heatmap_APOE.png")
    plt.savefig(out, dpi=250)
    plt.close()
    print("Saved", out)



plot_wass_enrichment_bars(df_kmeans, df_sub)
plot_apoe_cluster_heatmaps(df_kmeans, df_sub)

# ================================================================
# PART 6 — “NATURE GRID” UMAPs WITH KDE BACKGROUND
# ================================================================
def plot_umap_nature_grid(Z_dict, df_sub):
    cols = [fac for fac in GRID_TRAITS if fac in df_sub.columns]
    if not cols:
        return

    n_cols = len(cols)
    n_rows = len(METRICS)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(3 * n_cols, 3 * n_rows),
        dpi=250,
        squeeze=False,
    )

    for r, metric in enumerate(METRICS):
        Z = Z_dict[metric]
        emb = compute_umap(StandardScaler().fit_transform(Z))

        for c, fac in enumerate(cols):
            ax = axes[r, c]
            kde_background(ax, emb)

            if fac in TRAIT_COLS_CONT:
                vals = pd.to_numeric(df_sub[fac], errors="coerce")
                ax.scatter(
                    emb[:, 0],
                    emb[:, 1],
                    c=vals,
                    cmap="viridis",
                    s=20,
                    edgecolor="none",
                )
            else:
                labels = df_sub[fac].astype(str)
                cats = pd.Categorical(labels)
                ax.scatter(
                    emb[:, 0],
                    emb[:, 1],
                    c=cats.codes,
                    cmap="tab10",
                    s=20,
                    edgecolor="k",
                    linewidth=0.2,
                )

            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{metric} — {fac}", fontsize=8)
            if c == 0:
                ax.set_ylabel(metric, fontsize=8)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "UMAP_NatureGrid_CORR_EUCLID_WASS.png")
    plt.savefig(out, dpi=250)
    plt.close()
    print("Saved", out)


plot_umap_nature_grid(Z_dict, df_sub)

# ================================================================
# PART 7 — FAMILY DENDROGRAM (CORR LATENT CENTROIDS)
# ================================================================
def plot_family_dendrogram(Z_corr, df_sub):
    if "Family" not in df_sub.columns:
        return
    Zs = StandardScaler().fit_transform(Z_corr)
    df_tmp = df_sub[["MRI_Exam", "Family"]].copy()
    df_tmp["Family"] = df_tmp["Family"].astype(str)

    fams = df_tmp["Family"].unique()
    centroids = []
    labels = []

    for fam in fams:
        idx = df_tmp["Family"] == fam
        if idx.sum() < 2:
            continue
        centroids.append(Zs[idx].mean(axis=0))
        labels.append(fam)

    if len(centroids) < 2:
        return

    X = np.vstack(centroids)
    Z_link = linkage(X, method="ward")

    plt.figure(figsize=(6, 5), dpi=250)
    dendrogram(Z_link, labels=labels, leaf_rotation=45)
    plt.ylabel("Euclidean distance (CORR latent)")
    plt.title("Family dendrogram (CORR latent centroids)")
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "Family_dendrogram_CORR.png")
    plt.savefig(out, dpi=250)
    plt.close()
    print("Saved", out)


plot_family_dendrogram(Z_dict["CORR"], df_sub)

# ================================================================
# PART 8 — 1000-BOOTSTRAP AUC ACROSS METRICS
# ================================================================
def compute_bootstrap_auc_all(Z_dict, df_sub, n_boot=1000):
    rows = []
    for metric in METRICS:
        Z = Z_dict[metric]
        Zs = StandardScaler().fit_transform(Z)

        for fac in ["APOE", "sex", "risk_for_ad"]:
            if fac not in df_sub.columns:
                continue

            labels = df_sub[fac].astype(str)
            if fac == "APOE":
                labels = labels.replace({"E4-": 0, "E4+": 1})
            elif fac == "sex":
                labels = labels.replace({"F": 0, "M": 1})

            mean_auc, lo, hi, _ = bootstrap_auc(Zs, labels, n_boot=n_boot)
            rows.append({
                "Metric": metric,
                "Factor": fac,
                "AUC": mean_auc,
                "CI_low": lo,
                "CI_high": hi,
                "n_boot": n_boot,
            })

    df_auc_boot = pd.DataFrame(rows)
    df_auc_boot.to_csv(
        os.path.join(TABLE_DIR, "auc_bootstrap_results.csv"),
        index=False,
    )
    print("Saved bootstrap AUC to auc_bootstrap_results.csv")

    # summary plot
    plt.figure(figsize=(8, 3), dpi=250)
    for fac, color in zip(
        ["APOE", "sex", "risk_for_ad"],
        ["tab:green", "tab:purple", "tab:orange"],
    ):
        sub = df_auc_boot[df_auc_boot["Factor"] == fac]
        if sub.empty:
            continue
        plt.errorbar(
            sub["Metric"],
            sub["AUC"],
            yerr=[
                sub["AUC"] - sub["CI_low"],
                sub["CI_high"] - sub["AUC"],
            ],
            marker="o",
            capsize=4,
            label=fac,
            color=color,
        )
    plt.ylim(0.2, 1.0)
    plt.ylabel("AUC (1000-bootstrap)")
    plt.xlabel("Distance Metric")
    plt.title("AUC (1000-bootstrap) across CORR / EUCLID / WASS")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "auc_bootstrap_comparison.png")
    plt.savefig(out, dpi=250)
    plt.close()
    print("Saved", out)


compute_bootstrap_auc_all(Z_dict, df_sub, n_boot=1000)

# ================================================================
# PART 9 — PCA & UMAP BY KMEANS CLUSTERS
# ================================================================
def plot_pca_kmeans_clusters(Z_dict, df_kmeans):
    fig, axes = plt.subplots(
        nrows=1, ncols=len(METRICS), figsize=(12, 3), dpi=250
    )

    for ax, metric in zip(axes, METRICS):
        Z = Z_dict[metric]
        Zs = StandardScaler().fit_transform(Z)

        pcs = PCA(n_components=2).fit_transform(Zs)

        sub = df_kmeans[df_kmeans["Distance"] == metric][["MRI_Exam", "Cluster"]].copy()
        if sub.empty:
            ax.axis("off")
            continue

        sub["MRI_Exam"] = sub["MRI_Exam"].astype(str)
        order = df_sub[["MRI_Exam"]].copy()
        df_lab = order.merge(sub, on="MRI_Exam", how="left")
        labels = df_lab["Cluster"].values

        for cl in sorted(np.unique(labels[~pd.isna(labels)])):
            mask = labels == cl
            ax.scatter(
                pcs[mask, 0],
                pcs[mask, 1],
                s=30,
                label=f"Cluster {int(cl)}",
            )

        ax.set_title(f"{metric}: PCA by cluster (KMeans)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    axes[-1].legend()
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "PCA_kmeans_clusters_CORR_EUCLID_WASS.png")
    plt.savefig(out, dpi=250)
    plt.close()
    print("Saved", out)


def plot_umap_kmeans_clusters(Z_dict, df_kmeans):
    fig, axes = plt.subplots(
        nrows=1, ncols=len(METRICS), figsize=(12, 3), dpi=250
    )

    for ax, metric in zip(axes, METRICS):
        Z = Z_dict[metric]
        emb = compute_umap(StandardScaler().fit_transform(Z))

        kde_background(ax, emb)

        sub = df_kmeans[df_kmeans["Distance"] == metric][["MRI_Exam", "Cluster"]].copy()
        if sub.empty:
            ax.axis("off")
            continue

        sub["MRI_Exam"] = sub["MRI_Exam"].astype(str)
        order = df_sub[["MRI_Exam"]].copy()
        df_lab = order.merge(sub, on="MRI_Exam", how="left")
        labels = df_lab["Cluster"].values

        for cl in sorted(np.unique(labels[~pd.isna(labels)])):
            mask = labels == cl
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=30,
                label=f"Cluster {int(cl)}",
            )

        ax.set_title(f"{metric}: KMeans clusters (UMAP)")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[-1].legend()
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "UMAP_kmeans_clusters_CORR_EUCLID_WASS.png")
    plt.savefig(out, dpi=250)
    plt.close()
    print("Saved", out)


plot_pca_kmeans_clusters(Z_dict, df_kmeans)
plot_umap_kmeans_clusters(Z_dict, df_kmeans)

# ================================================================
# PART 10 — CLUSTER QUALITY & DISTANCE STATS (KMeans)
# ================================================================
def compute_cluster_distance_stats(Z_dict, df_kmeans):
    metric_rows = []
    cluster_rows = []

    for metric in METRICS:
        Z = Z_dict[metric]
        Zs = StandardScaler().fit_transform(Z)

        sub = df_kmeans[df_kmeans["Distance"] == metric][["MRI_Exam", "Cluster"]].copy()
        if sub.empty:
            continue
        sub["MRI_Exam"] = sub["MRI_Exam"].astype(str)

        order = df_sub[["MRI_Exam"]].copy()
        df_lab = order.merge(sub, on="MRI_Exam", how="left")
        labels = df_lab["Cluster"].values

        mask = ~pd.isna(labels)
        labels = labels[mask].astype(int)
        Zs_valid = Zs[mask]

        if len(np.unique(labels)) < 2:
            continue

        sil_global = silhouette_score(Zs_valid, labels)
        ch = calinski_harabasz_score(Zs_valid, labels)
        db = davies_bouldin_score(Zs_valid, labels)

        # within-cluster + Dunn
        centroids = {}
        max_intra = 0.0
        for cl in np.unique(labels):
            idx = labels == cl
            Xc = Zs_valid[idx]
            centroids[cl] = Xc.mean(axis=0)
            if Xc.shape[0] > 1:
                dists = np.linalg.norm(
                    Xc[:, None, :] - Xc[None, :, :], axis=-1
                )
                max_intra = max(max_intra, float(dists.max()))
                triu = dists[np.triu_indices_from(dists, k=1)]
                within_mean = float(triu.mean()) if triu.size > 0 else 0.0
            else:
                within_mean = 0.0

            cluster_rows.append({
                "Metric": metric,
                "Cluster": int(cl),
                "n": int(Xc.shape[0]),
                "within_mean_dist": within_mean,
            })

        min_inter = np.inf
        for c1, c2 in combinations(centroids.keys(), 2):
            d = np.linalg.norm(centroids[c1] - centroids[c2])
            min_inter = min(min_inter, float(d))

        dunn = float(min_inter / max(max_intra, 1e-6))

        metric_rows.append({
            "Metric": metric,
            "silhouette_global": sil_global,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "dunn_index": dunn,
        })

    df_metric = pd.DataFrame(metric_rows)
    df_cluster = pd.DataFrame(cluster_rows)

    df_metric.to_csv(
        os.path.join(TABLE_DIR, "cluster_metric_stats_kmeans.csv"),
        index=False,
    )
    df_cluster.to_csv(
        os.path.join(TABLE_DIR, "cluster_within_stats_kmeans.csv"),
        index=False,
    )
    print("Saved cluster stats (metric + within-cluster).")

    if not df_metric.empty:
        fig, ax1 = plt.subplots(figsize=(6, 3), dpi=250)
        ax1.bar(
            df_metric["Metric"],
            df_metric["silhouette_global"],
            alpha=0.7,
            label="Silhouette",
        )
        ax1.set_ylabel("Silhouette")
        ax2 = ax1.twinx()
        ax2.plot(
            df_metric["Metric"],
            df_metric["dunn_index"],
            marker="o",
            color="tab:red",
            label="Dunn index",
        )
        ax2.set_ylabel("Dunn index")
        fig.legend(loc="upper right")
        plt.title("Cluster quality (KMeans)")
        plt.tight_layout()
        out = os.path.join(PLOT_DIR, "kmeans_cluster_quality.png")
        plt.savefig(out, dpi=250)
        plt.close()
        print("Saved", out)


compute_cluster_distance_stats(Z_dict, df_kmeans)

# ================================================================
# PART 11 — MULTINOMIAL LOGISTIC REGRESSION COEFFICIENTS (KMeans)
# ================================================================
def compute_logreg_cluster_coeffs(Z_dict, df_kmeans):
    rows = []
    for metric in METRICS:
        Z = Z_dict[metric]
        Zs = StandardScaler().fit_transform(Z)

        sub = df_kmeans[df_kmeans["Distance"] == metric][["MRI_Exam", "Cluster"]].copy()
        if sub.empty:
            continue
        sub["MRI_Exam"] = sub["MRI_Exam"].astype(str)
        order = df_sub[["MRI_Exam"]].copy()
        df_lab = order.merge(sub, on="MRI_Exam", how="left")
        y = df_lab["Cluster"].values

        mask = ~pd.isna(y)
        Zs_valid = Zs[mask]
        y = y[mask].astype(int)

        if len(np.unique(y)) < 2:
            continue

        clf = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=500,
        )
        clf.fit(Zs_valid, y)

        coef = clf.coef_  # (n_classes, latent_dim)
        for cls_idx, cls in enumerate(clf.classes_):
            for d in range(coef.shape[1]):
                rows.append({
                    "Metric": metric,
                    "Cluster": int(cls),
                    "Latent_dim": d + 1,
                    "coef": float(coef[cls_idx, d]),
                    "abs_coef": float(abs(coef[cls_idx, d])),
                })

    df_coef = pd.DataFrame(rows)
    df_coef.to_csv(
        os.path.join(TABLE_DIR, "logreg_cluster_coeffs_kmeans.csv"),
        index=False,
    )
    print("Saved logistic regression cluster coefficients.")


compute_logreg_cluster_coeffs(Z_dict, df_kmeans)

# ================================================================
# PART 12 — CHI-SQUARE + CRAMER'S V (KMeans + Consensus)
# ================================================================
def compute_chisq_for_clusters_kmeans(df_kmeans, df_sub):
    results = []
    for metric in METRICS:
        sub = df_kmeans[df_kmeans["Distance"] == metric][["MRI_Exam", "Cluster"]].copy()
        if sub.empty:
            continue
        sub["MRI_Exam"] = sub["MRI_Exam"].astype(str)
        df_m = df_sub.merge(sub, on="MRI_Exam", how="inner")
        df_m["Cluster"] = df_m["Cluster"].astype(str)

        for fac in ["APOE", "risk_for_ad", "sex"]:
            if fac not in df_m.columns:
                continue

            tab = pd.crosstab(df_m["Cluster"], df_m[fac])
            if tab.shape[0] < 2 or tab.shape[1] < 2:
                continue

            chi2, p, dof, exp = chi2_contingency(tab)
            n = tab.to_numpy().sum()
            r, c = tab.shape
            phi2 = chi2 / n
            cramers_v = math.sqrt(phi2 / max(1, min(r - 1, c - 1)))

            results.append({
                "cluster_type": "kmeans",
                "metric": metric,
                "factor": fac,
                "chi2": chi2,
                "dof": dof,
                "p": p,
                "cramers_v": cramers_v,
                "n": int(n),
            })
    return results


def compute_chisq_for_clusters_consensus(df_cons, df_sub):
    results = []
    df_l = df_cons.rename(columns={"ConsensusCluster": "ClusterID"})[
        ["MRI_Exam", "ClusterID"]
    ].copy()
    df_l["MRI_Exam"] = df_l["MRI_Exam"].astype(str)
    df_m = df_sub.merge(df_l, on="MRI_Exam", how="inner")
    df_m["ClusterID"] = df_m["ClusterID"].astype(str)

    for fac in ["APOE", "risk_for_ad", "sex"]:
        if fac not in df_m.columns:
            continue

        tab = pd.crosstab(df_m["ClusterID"], df_m[fac])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue

        chi2, p, dof, exp = chi2_contingency(tab)
        n = tab.to_numpy().sum()
        r, c = tab.shape
        phi2 = chi2 / n
        cramers_v = math.sqrt(phi2 / max(1, min(r - 1, c - 1)))

        results.append({
            "cluster_type": "consensus",
            "metric": "ALL",
            "factor": fac,
            "chi2": chi2,
            "dof": dof,
            "p": p,
            "cramers_v": cramers_v,
            "n": int(n),
        })
    return results


chisq_rows = []
chisq_rows += compute_chisq_for_clusters_kmeans(df_kmeans, df_sub)
chisq_rows += compute_chisq_for_clusters_consensus(df_cons, df_sub)

df_chisq = pd.DataFrame(chisq_rows)
df_chisq.to_csv(
    os.path.join(TABLE_DIR, "cluster_chisq_tests.csv"),
    index=False,
)
print("Saved χ² tests to cluster_chisq_tests.csv")

# ================================================================
# DONE
# ================================================================
print("\n======================")
print(" DONE — Part 6 (latent evaluation + cluster statistics)")
print("Outputs in:", EVAL_ROOT)
print("======================")
