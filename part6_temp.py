#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PART 6 — LATENT EVALUATION (MERGED)
===================================

This script evaluates latent spaces for K=25 using CORR / EUCLID / WASS
GAUDI latents.

It produces:

  plots/
    UMAP_*_NatureGrid_withLegends.png
    PCA_*_NatureGrid_withLegends.png
    UMAP_kmeans_clusters_CORR_EUCLID_WASS.png
    PCA_kmeans_clusters_CORR_EUCLID_WASS.png
    latent_trait_corr_heatmap.png
    auc_bootstrap_comparison.png
    Family_dendrogram_CORR.png
    kmeans_cluster_bars_WASS.png

  tables/
    knn_classification_results.csv
    latent_trait_correlations.csv
    kmeans_chisq_tests.csv
    silhouette_scores.csv
    auc_bootstrap_summary.csv
    cluster_trait_composition_kmeans.csv

Paths are set for:
  /mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from scipy.stats import pearsonr, chi2_contingency
from scipy.cluster.hierarchy import linkage, dendrogram

# Make pandas warnings quiet
pd.set_option("future.no_silent_downcasting", True)

sns.set(style="whitegrid")

# ================================================================
# CONFIGURATION
# ================================================================
GRAPH_K = 25
METRICS = ["CORR", "EUCLID", "WASS"]
FOCUS_METRIC = "EUCLID"

ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation"
COLUMNS_ROOT = os.path.join(
    ROOT, "columns4gaudi111825", "columna-analyses111925"
)
MDATA_PATH = os.path.join(ROOT, "processed_graph_data", "metadata_with_PCs.xlsx")

EVAL_ROOT = os.path.join(COLUMNS_ROOT, f"latent_eval_K{GRAPH_K}")
PLOT_DIR = os.path.join(EVAL_ROOT, "plots")
TABLE_DIR = os.path.join(EVAL_ROOT, "tables")
for d in [EVAL_ROOT, PLOT_DIR, TABLE_DIR]:
    os.makedirs(d, exist_ok=True)

RES_DIR = os.path.join(COLUMNS_ROOT, f"results_K{GRAPH_K}")

TRAIT_COLS_CONT = ["age", "BMI"]
TRAIT_COLS_BIN = ["APOE", "sex", "risk_for_ad"]

# ------------------------------------------------
# Palette A — global categorical colors
# ------------------------------------------------
PALETTE_APOE = {"E4+": "#7B1FA2", "E4-": "#43A047"}          # purple / green
PALETTE_SEX = {"F": "#7B1FA2", "M": "#43A047"}               # purple / green
PALETTE_RISK = {"0": "#43A047", "1": "#7B1FA2"}              # green / purple
PALETTE_CLUSTERS = {0: "#1b9e77", 1: "#d95f02", 2: "#7570b3"}  # teal / orange / violet

# ================================================================
# HELPERS
# ================================================================
def compute_umap(Z, n_neighbors=15, min_dist=0.05):
    """2D UMAP embedding with fixed seed."""
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, Z.shape[0] - 1),
        min_dist=min_dist,
        spread=1.0,
        n_components=2,
        random_state=42,
    )
    return reducer.fit_transform(Z)


def knn_latent_classification(Z, labels, n_neighbors=5):
    """5-fold CV kNN classification in latent space."""
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
    """NaN-safe Pearson correlation."""
    x, y = np.asarray(x), np.asarray(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 5:
        return np.nan
    try:
        r, _ = pearsonr(x[mask], y[mask])
        return float(r)
    except Exception:
        return np.nan


def get_palette_for_factor(factor, cats):
    """Return categorical color mapping using palette A where possible."""
    cats = list(cats)
    if factor == "APOE":
        return {c: PALETTE_APOE.get(c, "#999999") for c in cats}
    if factor == "sex":
        return {c: PALETTE_SEX.get(c, "#999999") for c in cats}
    if factor == "risk_for_ad":
        return {c: PALETTE_RISK.get(str(c), "#999999") for c in cats}
    # fallback
    cmap = sns.color_palette("tab10", n_colors=len(cats))
    return {c: cmap[i] for i, c in enumerate(cats)}


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
# PART 1 — UMAP / PCA NATURE GRIDS WITH LEGENDS + CONTOURS
# ================================================================
NATURE_FACTORS = [
    "APOE",
    "genotype",
    "sex",
    "risk_for_ad",
    "ethnicity",
    "Family",
    "age",
    "BMI",
]


def make_nature_grid(Z_dict, df, mode="UMAP"):
    """
    Make 3x8 grid (metrics x factors) with shared background contours,
    palette A for binary factors, legends + colorbars.
    """
    assert mode in ["UMAP", "PCA"]
    fig, axes = plt.subplots(
        nrows=len(METRICS),
        ncols=len(NATURE_FACTORS),
        figsize=(18, 6),
        dpi=300,
        sharex=False,
        sharey=False,
    )

    for i, metric in enumerate(METRICS):
        Z = Z_dict[metric]
        Zs = StandardScaler().fit_transform(Z)
        if mode == "UMAP":
            emb = compute_umap(Zs)
        else:
            emb = PCA(n_components=2).fit_transform(Zs)

        # pre-compute density for level sets
        x, y = emb[:, 0], emb[:, 1]
        # global contour extents
        for j, factor in enumerate(NATURE_FACTORS):
            ax = axes[i, j]

            # background contours (always grey)
            try:
                sns.kdeplot(
                    x=x,
                    y=y,
                    levels=10,
                    linewidths=0.5,
                    color="#D0D0D0",
                    fill=False,
                    ax=ax,
                )
            except Exception:
                pass

            if factor in TRAIT_COLS_CONT:
                vals = pd.to_numeric(df[factor], errors="coerce")
                sc = ax.scatter(
                    x,
                    y,
                    c=vals,
                    s=16,
                    cmap="viridis",
                    edgecolor="none",
                    alpha=0.95,
                )
                cbar = plt.colorbar(sc, ax=ax)
                cbar.ax.tick_params(labelsize=6)
                cbar.set_label(factor, fontsize=7)
            else:
                labels = df[factor].astype(str)
                cats = pd.Categorical(labels)
                pal = get_palette_for_factor(factor, cats.categories)
                colors = [pal[c] for c in cats]
                ax.scatter(
                    x,
                    y,
                    c=colors,
                    s=16,
                    edgecolor="k",
                    linewidth=0.2,
                    alpha=0.95,
                )

                # legend (small)
                if j == 0:  # only once per row for binary-ish factors
                    handles = [
                        plt.Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor=pal[c],
                            markersize=4,
                            label=str(c),
                        )
                        for c in cats.categories
                    ]
                    ax.legend(
                        handles=handles,
                        title=factor,
                        fontsize=5,
                        title_fontsize=6,
                        frameon=True,
                        framealpha=0.9,
                        loc="upper right",
                    )

            if i == 0:
                ax.set_title(f"{metric} — {factor}", fontsize=8)
            if j == 0:
                ax.set_ylabel(metric, fontsize=8)
            else:
                ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    fname = f"{mode}_NatureGrid_CORR_EUCLID_WASS_withLegends.png"
    out = os.path.join(PLOT_DIR, fname)
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved grid:", out)


print("\n=== PART 1: UMAP/PCA Nature grids ===")
make_nature_grid(Z_dict, df_sub, mode="UMAP")
make_nature_grid(Z_dict, df_sub, mode="PCA")

# ================================================================
# PART 2 — kNN CLASSIFICATION (APOE, sex, risk_for_ad)
# ================================================================
print("\n=== PART 2: kNN classification in latent space ===")
rows = []
for metric, Z in Z_dict.items():
    Zs = StandardScaler().fit_transform(Z)
    for fac in ["APOE", "sex", "risk_for_ad"]:
        labels = df_sub[fac].astype(str).replace({"nan": "NA", "NaN": "NA"})
        acc, f1 = knn_latent_classification(Zs, labels)
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
print("\n=== PART 3: Latent–trait correlations (FOCUS_METRIC=EUCLID) ===")
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

    ap_bin = (
        df_sub["APOE"]
        .replace({"E4-": 0, "E4+": 1})
        .astype("float")
    )
    row["r_APOE_bin"] = safe_pearson(vec, ap_bin)

    risk_ord = (
        df_sub["risk_for_ad"]
        .astype(str)
        .replace({"0": 0, "1": 1, "2": 2, "3": 3})
        .astype(float)
    )
    row["r_risk_ord"] = safe_pearson(vec, risk_ord)
    corr_rows.append(row)

df_corr = pd.DataFrame(corr_rows)
df_corr.to_csv(
    os.path.join(TABLE_DIR, "latent_trait_correlations.csv"), index=False
)

plt.figure(figsize=(5, 5), dpi=300)
sns.heatmap(
    df_corr.set_index("Latent_dim").iloc[:, 2:].abs(),
    cmap="viridis",
    cbar_kws={"label": "|r|"},
)
plt.title(f"Latent–Trait |r| heatmap (K={GRAPH_K}, {FOCUS_METRIC})")
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_DIR, "latent_trait_corr_heatmap.png"), dpi=300
)
plt.close()

# ================================================================
# PART 4 — KMEANS CLUSTERS, χ², SILHOUETTE, ENRICHMENT, DENDROGRAM
# ================================================================
print("\n=== PART 4: KMeans clusters, stats & enrichment ===")

# --------- Load KMeans cluster assignments ----------
kmeans_csv = os.path.join(RES_DIR, f"kmeans_cluster_assignments_K{GRAPH_K}.csv")
df_kmeans = pd.read_csv(kmeans_csv)
df_kmeans["MRI_Exam"] = df_kmeans["MRI_Exam"].astype(str)

# Helper to align clusters with subject order
def get_cluster_labels_for_metric(metric):
    sub = df_kmeans[df_kmeans["Distance"] == metric][
        ["MRI_Exam", "Cluster"]
    ].copy()
    if sub.empty:
        return None
    sub["MRI_Exam"] = sub["MRI_Exam"].astype(str)
    df_m = pd.merge(
        df_sub[["MRI_Exam"]], sub, on="MRI_Exam", how="left"
    )  # preserve order
    return df_m["Cluster"].to_numpy()


# ---------- UMAP / PCA colored by KMeans clusters ----------
def plot_kmeans_cluster_scatter(Z_dict, df_sub, df_kmeans):
    fig, axes = plt.subplots(
        nrows=1, ncols=len(METRICS), figsize=(12, 3), dpi=300
    )

    for i, metric in enumerate(METRICS):
        ax = axes[i]
        Z = Z_dict[metric]
        Zs = StandardScaler().fit_transform(Z)
        emb = compute_umap(Zs)
        x, y = emb[:, 0], emb[:, 1]

        clusters = get_cluster_labels_for_metric(metric)
        if clusters is None or np.all(pd.isna(clusters)):
            print(f"[WARN] No clusters for metric {metric}")
            continue

        # contours
        try:
            sns.kdeplot(
                x=x,
                y=y,
                levels=10,
                linewidths=0.5,
                color="#D0D0D0",
                fill=False,
                ax=ax,
            )
        except Exception:
            pass

        colors = [PALETTE_CLUSTERS.get(int(c), "#999999") for c in clusters]
        ax.scatter(
            x,
            y,
            c=colors,
            s=20,
            edgecolor="k",
            linewidth=0.2,
            alpha=0.95,
            label=None,
        )

        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PALETTE_CLUSTERS.get(c, "#999999"),
                markeredgecolor="k",
                markersize=5,
                label=f"Cluster {c}",
            )
            for c in sorted(np.unique(clusters[~pd.isna(clusters)]))
        ]
        ax.legend(
            handles=handles,
            title="Cluster",
            fontsize=6,
            title_fontsize=7,
            frameon=True,
            loc="upper right",
        )
        ax.set_title(f"{metric}: KMeans clusters (UMAP)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "UMAP_kmeans_clusters_CORR_EUCLID_WASS.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

    # PCA version
    fig, axes = plt.subplots(
        nrows=1, ncols=len(METRICS), figsize=(12, 3), dpi=300
    )
    for i, metric in enumerate(METRICS):
        ax = axes[i]
        Z = Z_dict[metric]
        Zs = StandardScaler().fit_transform(Z)
        pcs = PCA(n_components=2).fit_transform(Zs)
        x, y = pcs[:, 0], pcs[:, 1]

        clusters = get_cluster_labels_for_metric(metric)
        if clusters is None or np.all(pd.isna(clusters)):
            continue
        colors = [PALETTE_CLUSTERS.get(int(c), "#999999") for c in clusters]

        ax.scatter(
            x,
            y,
            c=colors,
            s=20,
            edgecolor="k",
            linewidth=0.2,
            alpha=0.95,
        )
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PALETTE_CLUSTERS.get(c, "#999999"),
                markeredgecolor="k",
                markersize=5,
                label=f"Cluster {c}",
            )
            for c in sorted(np.unique(clusters[~pd.isna(clusters)]))
        ]
        ax.legend(
            handles=handles,
            title="Cluster",
            fontsize=6,
            title_fontsize=7,
            frameon=True,
            loc="upper right",
        )
        ax.set_title(f"{metric}: PCA by cluster", fontsize=9)
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "PCA_kmeans_clusters_CORR_EUCLID_WASS.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


plot_kmeans_cluster_scatter(Z_dict, df_sub, df_kmeans)

# ---------- χ² tests for Cluster × [APOE, sex, risk_for_ad] ----------
chi_rows = []
for metric in METRICS:
    clusters = get_cluster_labels_for_metric(metric)
    if clusters is None or np.all(pd.isna(clusters)):
        continue
    df_m = df_sub.copy()
    df_m["Cluster"] = clusters

    for fac in ["APOE", "sex", "risk_for_ad"]:
        ct = pd.crosstab(df_m["Cluster"], df_m[fac])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, dof, _ = chi2_contingency(ct)
        chi_rows.append(
            {
                "K": GRAPH_K,
                "Metric": metric,
                "Factor": fac,
                "chi2": chi2,
                "dof": dof,
                "p_value": p,
            }
        )

df_chi = pd.DataFrame(chi_rows)
df_chi.to_csv(
    os.path.join(TABLE_DIR, "kmeans_chisq_tests.csv"), index=False
)
print("Saved χ² tests.")

# ---------- Silhouette scores ----------
sil_rows = []
for metric in METRICS:
    clusters = get_cluster_labels_for_metric(metric)
    if clusters is None or np.any(pd.isna(clusters)):
        continue
    Z = Z_dict[metric]
    Zs = StandardScaler().fit_transform(Z)
    if len(np.unique(clusters)) < 2:
        continue
    score = silhouette_score(Zs, clusters)
    sil_rows.append(
        {"K": GRAPH_K, "Metric": metric, "silhouette_score": score}
    )

df_sil = pd.DataFrame(sil_rows)
df_sil.to_csv(
    os.path.join(TABLE_DIR, "silhouette_scores.csv"), index=False
)
print("Saved silhouette scores.")

# ---------- Cluster composition & WASS bars ----------
def cluster_trait_composition(df_labels, df_meta, metric_name):
    df_labels["MRI_Exam"] = df_labels["MRI_Exam"].astype(str)
    df_m = pd.merge(df_meta, df_labels, on="MRI_Exam", how="inner")

    results = []
    for cl in sorted(df_m["Cluster"].dropna().unique()):
        sub = df_m[df_m["Cluster"] == cl]
        row = {"Metric": metric_name, "cluster": cl, "n": len(sub)}
        for fac in ["APOE", "sex", "risk_for_ad"]:
            vc = sub[fac].astype(str).value_counts(normalize=True)
            for cat, p in vc.items():
                row[f"{fac}_{cat}"] = round(float(p), 3)
        results.append(row)
    return pd.DataFrame(results), df_m


comp_all = []
df_m_WASS = None

for metric in METRICS:
    sub = df_kmeans[df_kmeans["Distance"] == metric][
        ["MRI_Exam", "Cluster"]
    ].copy()
    if sub.empty:
        continue
    comp, df_m = cluster_trait_composition(sub, df_sub, metric)
    comp_all.append(comp)
    if metric == "WASS":
        df_m_WASS = df_m

if comp_all:
    df_comp_all = pd.concat(comp_all, ignore_index=True)
    df_comp_all.to_csv(
        os.path.join(TABLE_DIR, "cluster_trait_composition_kmeans.csv"),
        index=False,
    )
    print("Saved cluster composition table.")

# WASS barplots (only if WASS clusters exist)
if df_m_WASS is not None and not df_m_WASS.empty:
    print("Making WASS cluster-enrichment barplots...")
    factors = ["APOE", "sex", "risk_for_ad"]
    fig, axes = plt.subplots(
        nrows=len(factors), ncols=1, figsize=(6, 8), dpi=300
    )

    for ax, fac in zip(axes, factors):
        rows = []
        for cl in sorted(df_m_WASS["Cluster"].dropna().unique()):
            sub = df_m_WASS[df_m_WASS["Cluster"] == cl]
            vc = sub[fac].astype(str).value_counts(normalize=True)
            for cat, p in vc.items():
                rows.append(
                    {
                        fac: cat,
                        "Cluster": f"{int(cl)}",
                        "proportion": float(p),
                    }
                )
        df_plot = pd.DataFrame(rows)
        pal = get_palette_for_factor(fac, sorted(df_plot[fac].unique()))
        sns.barplot(
            data=df_plot,
            x=fac,
            y="proportion",
            hue="Cluster",
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion")
        ax.set_title(f"{fac} enrichment across clusters (WASS, KMeans)")
        ax.legend(title="Cluster", fontsize=7, title_fontsize=8)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "kmeans_cluster_bars_WASS.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)
else:
    print("[WARN] No WASS clusters for barplots.")

# ---------- Family dendrogram (CORR latent centroids) ----------
print("Building family dendrogram (CORR latent centroids)...")
Z_corr = StandardScaler().fit_transform(Z_dict["CORR"])
families = df_sub["Family"].astype(str)

family_centroids = {}
for fam in sorted(families.unique()):
    if fam.lower() == "nan":
        continue
    mask = families == fam
    if mask.sum() < 2:
        continue
    family_centroids[fam] = Z_corr[mask].mean(axis=0)

if len(family_centroids) >= 2:
    fam_names = list(family_centroids.keys())
    mat = np.vstack([family_centroids[f] for f in fam_names])
    linkage_mat = linkage(mat, method="ward")
    plt.figure(figsize=(5, 4), dpi=300)
    dendrogram(linkage_mat, labels=fam_names, leaf_rotation=45)
    plt.ylabel("Euclidean distance (CORR latent)")
    plt.title("Family dendrogram (CORR latent centroids)")
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "Family_dendrogram_CORR.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)
else:
    print("[WARN] Not enough families with ≥2 members for dendrogram.")

# ================================================================
# PART 5 — 1000× BOOTSTRAP AUC COMPARISON (APOE, sex, risk_for_ad)
# ================================================================
print("\n=== PART 5: 1000× bootstrap AUC across metrics ===")

def bootstrap_auc_for_metric_factor(Z, labels, n_boot=1000):
    labels = np.asarray(labels)
    # require binary labels
    uniq = np.unique(labels)
    if len(uniq) != 2:
        return None

    Zs = StandardScaler().fit_transform(Z)
    n = Zs.shape[0]
    aucs = []

    for b in range(n_boot):
        rng = np.random.default_rng(seed=42 + b)
        idx = rng.integers(0, n, size=n)
        Zb = Zs[idx]
        yb = labels[idx]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + b)
        fold_aucs = []
        for train, test in skf.split(Zb, yb):
            clf = LogisticRegression(
                max_iter=1000, solver="lbfgs"
            )
            clf.fit(Zb[train], yb[train])
            prob = clf.predict_proba(Zb[test])[:, 1]
            if len(np.unique(yb[test])) < 2:
                continue
            fold_aucs.append(roc_auc_score(yb[test], prob))
        if fold_aucs:
            aucs.append(np.mean(fold_aucs))

    if not aucs:
        return None
    aucs = np.array(aucs)
    return float(aucs.mean()), float(aucs.std())


auc_rows = []
summary_for_plot = {}

for fac in ["APOE", "sex", "risk_for_ad"]:
    y_raw = df_sub[fac].astype(str)
    # binarize where needed
    if fac == "APOE":
        y = y_raw.replace({"E4-": 0, "E4+": 1}).astype(int).to_numpy()
    elif fac == "sex":
        y = y_raw.replace({"F": 0, "M": 1}).astype(int).to_numpy()
    else:  # risk_for_ad (0 vs 1, ignoring >1 if present)
        y = y_raw.replace({"0": 0, "1": 1, "2": 1, "3": 1}).astype(int).to_numpy()

    summary_for_plot[fac] = {"Metric": [], "mean_auc": [], "std_auc": []}

    for metric in METRICS:
        Z = Z_dict[metric]
        res = bootstrap_auc_for_metric_factor(Z, y, n_boot=1000)
        if res is None:
            continue
        mean_auc, std_auc = res
        auc_rows.append(
            {
                "K": GRAPH_K,
                "Metric": metric,
                "Factor": fac,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
            }
        )
        summary_for_plot[fac]["Metric"].append(metric)
        summary_for_plot[fac]["mean_auc"].append(mean_auc)
        summary_for_plot[fac]["std_auc"].append(std_auc)

df_auc = pd.DataFrame(auc_rows)
df_auc.to_csv(
    os.path.join(TABLE_DIR, "auc_bootstrap_summary.csv"), index=False
)
print("Saved AUC bootstrap summary.")

# Plot
plt.figure(figsize=(8, 3), dpi=300)
x_positions = np.arange(len(METRICS))
metric_to_x = {m: i for i, m in enumerate(METRICS)}

colors_fac = {"APOE": "#2ca02c", "sex": "#9467bd", "risk_for_ad": "#ff7f0e"}

for fac, col in colors_fac.items():
    data = summary_for_plot.get(fac, None)
    if data is None or not data["Metric"]:
        continue
    xs = [metric_to_x[m] for m in data["Metric"]]
    means = data["mean_auc"]
    stds = data["std_auc"]
    plt.errorbar(
        xs,
        means,
        yerr=stds,
        fmt="o-",
        label=fac,
        color=col,
        capsize=3,
    )

plt.xticks(x_positions, METRICS, fontsize=10)
plt.ylim(0.2, 1.0)
plt.ylabel("AUC (1000-bootstrap)", fontsize=11)
plt.xlabel("Distance Metric", fontsize=11)
plt.title(
    "AUC (1000-bootstrap) across CORR / EUCLID / WASS",
    fontsize=13,
)
plt.legend(title="", fontsize=9)
plt.tight_layout()
out = os.path.join(PLOT_DIR, "auc_bootstrap_comparison.png")
plt.savefig(out, dpi=300)
plt.close()
print("Saved:", out)

# ================================================================
print("\n======================")
print(" DONE — Merged PART 6")
print("Outputs in:", EVAL_ROOT)
print("======================")
