#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAUDI Latent Space Analysis and Consensus Clustering
====================================================

Sections:
1.  Imports and global paths
2.  Load metadata and align with graphs
3.  Load CORR / EUCLID / WASS latents (K=10 Joint MD+QSM)
4.  Clean metadata (categorical & continuous)
5.  UMAP helper + KDE contours
6.  Latent metrics: silhouette + 5-fold CV AUC
7.  Family clustering: silhouette + permutation p + dendrogram
8.  Nature-style 3×8 UMAP grid (CORR/EUCLID/WASS × labels)
9.  K-means clustering per metric + χ² tests + enrichment + heatmaps
10. Multi-metric consensus clustering (CORR+EUCLID+WASS)
"""

# ================================================================
# 1. IMPORTS AND GLOBAL PATHS
# ================================================================
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.neighbors import KernelDensity
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans, SpectralClustering

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import chi2_contingency

import umap
import torch
import seaborn as sns
import plotly.graph_objects as go

from torch_geometric.data import Data, Batch

# -------------------------
# Paths (adapt as needed)
# -------------------------
ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation"

# this is your analysis folder with graphs_knn and latent_k10 inside
COLUMNS_ROOT = os.path.join(ROOT, "columns4gaudi111825", "columna-analyses111925")

# graph base for K=10
GRAPHS_DIR = os.path.join(COLUMNS_ROOT, "graphs_knn", "k10")
MD_PT      = os.path.join(GRAPHS_DIR, "md_shared_knn_k10_corr_euclid_wass.pt")

# latents from your part1_gauditraining.py runs for K=10, Joint MD+QSM
LATENT_ROOT_K10 = os.path.join(COLUMNS_ROOT, "latent_k10")
latent_paths = {
    "CORR":   os.path.join(LATENT_ROOT_K10, "latent_epochs_Joint_CORR",   "latent_final_Joint_CORR.npy"),
    "EUCLID": os.path.join(LATENT_ROOT_K10, "latent_epochs_Joint_EUCLID", "latent_final_Joint_EUCLID.npy"),
    "WASS":   os.path.join(LATENT_ROOT_K10, "latent_epochs_Joint_WASS",   "latent_final_Joint_WASS.npy"),
}

# metadata path (adapt if needed)
MDATA_PATH = os.path.join(ROOT, "processed_graph_data", "metadata_with_PCs.xlsx")

# All outputs from this script will go here
OUTDIR = os.path.join(COLUMNS_ROOT, "umap_figures_nature")
os.makedirs(OUTDIR, exist_ok=True)

# visualization columns
categorical_columns = ["APOE", "genotype", "sex", "risk_for_ad", "ethnicity", "Family"]
continuous_columns  = ["age", "BMI"]
label_columns       = categorical_columns + continuous_columns


# ================================================================
# 2. LOAD METADATA AND ALIGN WITH GRAPHS
# ================================================================
print("Loading metadata from:", MDATA_PATH)
df = pd.read_excel(MDATA_PATH)
df["MRI_Exam"] = df["MRI_Exam"].astype(str).str.zfill(5)
print("Metadata rows:", len(df))

print("Loading MD graphs from:", MD_PT)
graphs = torch.load(MD_PT, map_location="cpu")
subject_ids = [str(getattr(g, "subject_id")).zfill(5) for g in graphs]
print("Found subject IDs in graphs:", len(subject_ids))
print("Example IDs:", subject_ids[:5])

# align metadata to graph order
df_sub = df[df["MRI_Exam"].isin(subject_ids)].copy()
if len(df_sub) != len(subject_ids):
    missing = sorted(set(subject_ids) - set(df_sub["MRI_Exam"]))
    if missing:
        print("⚠ WARNING: metadata does not fully match graphs.")
        print("⚠ Missing metadata for:", missing)
else:
    missing = []

df_sub["__order"] = df_sub["MRI_Exam"].apply(lambda s: subject_ids.index(s))
df_sub = df_sub.sort_values("__order").reset_index(drop=True)
df_sub.drop(columns="__order", inplace=True)

print("Aligned metadata shape:", df_sub.shape)
n_samples = df_sub.shape[0]


# ================================================================
# 3. LOAD LATENTS AND ALIGN TO METADATA
# ================================================================
keep_mask = np.isin(subject_ids, df_sub["MRI_Exam"].values)

latents = {}
for mode, path in latent_paths.items():
    print(f"\nLoading latents for {mode} from:", path)
    Z = np.load(path)
    print(f"{mode}: original latent shape {Z.shape}")

    if Z.shape[0] != len(subject_ids):
        raise ValueError(
            f"{mode}: latent rows ({Z.shape[0]}) != graphs ({len(subject_ids)})."
        )
    Z = Z[keep_mask]
    print(f"{mode}: latent aligned → {Z.shape}")

    latents[mode] = Z

for mode in latents:
    if latents[mode].shape[0] != n_samples:
        raise ValueError(
            f"{mode}: latent rows {latents[mode].shape[0]} != metadata rows {n_samples}"
        )


# ================================================================
# 4. CLEAN METADATA (CATEGORICAL & CONTINUOUS)
# ================================================================
na_tokens = {"", "nan", "NaN", "None", "NA"}

for col in categorical_columns:
    if col in df_sub.columns:
        s = df_sub[col].astype(str).str.strip()
        s = s.replace({val: "NA" for val in na_tokens})
        df_sub[col] = s

if "Family" in df_sub.columns:
    fam_vals = set(df_sub["Family"].unique())
    non_na_fams = fam_vals - {"NA"}
    if len(non_na_fams) == 0:
        print("Dropping 'Family' — no non-NA entries found.")
        categorical_columns = [c for c in categorical_columns if c != "Family"]
        label_columns = categorical_columns + continuous_columns
    else:
        print("Family categories (including NA):", fam_vals)


# ================================================================
# 5. UMAP HELPER + KDE CONTOURS
# ================================================================
def compute_umap(Z):
    reducer = umap.UMAP(
        n_neighbors=min(15, Z.shape[0] - 1),
        min_dist=0.1,
        spread=1.0,
        n_components=2,
        random_state=42,
    )
    return reducer.fit_transform(Z)


def add_kde_contour(ax, emb, color="k"):
    if emb.shape[0] < 5:
        return
    kde = KernelDensity(bandwidth=0.4)
    kde.fit(emb)
    x, y = np.mgrid[
        emb[:, 0].min():emb[:, 0].max():200j,
        emb[:, 1].min():emb[:, 1].max():200j
    ]
    grid = np.vstack([x.ravel(), y.ravel()]).T
    z = np.exp(kde.score_samples(grid)).reshape(200, 200)
    ax.contour(x, y, z, levels=6, linewidths=0.6, colors=color, alpha=0.25)


# Palettes
categorical_palettes = {
    "APOE":        "Dark2",
    "genotype":    "Set2",
    "sex":         "Set1",
    "risk_for_ad": "tab10",
    "ethnicity":   "tab20",
    "Family":      "Paired",
}
continuous_cmaps = {
    "age": "viridis",
    "BMI": "plasma",
}

def get_palette(col, n):
    """
    Return color array for n categories.
    If n == 2: fixed green/purple (E4-/E4+ style).
    """
    if n == 2:
        return np.array([
            [27/255, 158/255, 119/255],   # green
            [117/255, 112/255, 179/255],  # purple
        ])
    cmap_name = categorical_palettes.get(col, "tab10")
    cmap = plt.get_cmap(cmap_name)
    return cmap(np.linspace(0, 1, n))


# ================================================================
# 6. LATENT METRICS: SILHOUETTE + 5-FOLD CV AUC
# ================================================================
def preprocess_binary(labels, positive):
    """Convert labels to 0/1: positive vs other."""
    return np.array([1 if l == positive else 0 for l in labels])


def compute_auc(Z, y):
    """5-fold CV AUC."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    Zs = StandardScaler().fit_transform(Z)

    for train, test in skf.split(Zs, y):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Zs[train], y[train])
        prob = clf.predict_proba(Zs[test])[:, 1]
        aucs.append(roc_auc_score(y[test], prob))

    return np.mean(aucs), np.std(aucs)


binary_factors = {
    "APOE": ("E4+", "E4-"),
    "sex": ("F", "M"),
    "risk_for_ad": ("1", "0"),  # cast as string
}

metric_rows = []

for mode, Z in latents.items():
    print(f"\n=== Metrics on raw latents: {mode} ===")
    for factor, (pos_label, neg_label) in binary_factors.items():
        if factor not in df_sub.columns:
            continue

        labels = df_sub[factor].astype(str).values
        mask_valid = labels != "NA"
        labels_valid = labels[mask_valid]
        Z_valid = Z[mask_valid]

        if len(np.unique(labels_valid)) < 2 or Z_valid.shape[0] < 5:
            sil = np.nan
            auc_mean = np.nan
            auc_std = np.nan
        else:
            # Silhouette
            try:
                sil = silhouette_score(Z_valid, labels_valid)
            except Exception:
                sil = np.nan

            # AUC
            try:
                y = preprocess_binary(labels_valid, pos_label)
                if len(np.unique(y)) < 2:
                    auc_mean, auc_std = np.nan, np.nan
                else:
                    auc_mean, auc_std = compute_auc(Z_valid, y)
            except Exception:
                auc_mean, auc_std = np.nan, np.nan

        metric_rows.append([mode, factor, sil, auc_mean, auc_std])

df_metrics = pd.DataFrame(
    metric_rows,
    columns=["Distance", "Factor", "Silhouette", "AUC_mean", "AUC_std"]
)
metrics_csv = os.path.join(OUTDIR, "distance_metrics.csv")
df_metrics.to_csv(metrics_csv, index=False)
print("\nSaved metrics to:", metrics_csv)
print(df_metrics)


# plot metric comparisons
bright_green = "#2ECC71"
vivid_purple = "#9B59B6"
third_color  = "#E67E22"

def plot_metric(df_res, metric, ylabel, fname):
    plt.figure(figsize=(8, 4))
    colors = {
        "APOE": bright_green,
        "sex": vivid_purple,
        "risk_for_ad": third_color,
    }
    for factor in binary_factors.keys():
        sub = df_res[df_res["Factor"] == factor]
        if sub.empty:
            continue
        plt.plot(
            sub["Distance"],
            sub[metric],
            marker="o",
            markersize=8,
            linewidth=3,
            label=factor,
            color=colors.get(factor, "#1F77B4"),
        )

    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Distance Metric", fontsize=12)
    plt.title(f"{ylabel} across CORR / EUCLID / WASS", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(OUTDIR, fname)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)

plot_metric(df_metrics, "Silhouette", "Silhouette Score", "silhouette_comparison.png")
plot_metric(df_metrics, "AUC_mean", "AUC (mean, 5-fold CV)", "auc_comparison.png")


# ================================================================
# 7. FAMILY CLUSTERING STATS + DENDROGRAM
# ================================================================
def family_stats(Z, fam_labels_raw, n_perm=5000):
    fam_series = pd.Series(fam_labels_raw).astype(str)
    drop_tokens = {"NA", "nan", "NaN", "None", "0", ""}
    mask = ~fam_series.isin(drop_tokens)

    fam_valid = fam_series[mask]
    Z_valid = Z[mask]

    if len(Z_valid) < 5 or fam_valid.nunique() < 2:
        return np.nan, np.nan

    fams = pd.Categorical(fam_valid)
    try:
        sil_obs = silhouette_score(Z_valid, fams.codes)
    except Exception:
        return np.nan, np.nan

    rng = np.random.default_rng(42)
    perm_scores = []
    for _ in range(n_perm):
        shuffled = rng.permutation(fams.codes)
        try:
            perm_scores.append(silhouette_score(Z_valid, shuffled))
        except Exception:
            perm_scores.append(np.nan)

    perm_scores = np.array(perm_scores)
    perm_scores = perm_scores[~np.isnan(perm_scores)]
    if len(perm_scores) == 0:
        return sil_obs, np.nan

    pval = np.mean(perm_scores >= sil_obs)
    return sil_obs, pval


family_rows = []
if "Family" in df_sub.columns:
    fam_labels_all = df_sub["Family"].astype(str).values
    for mode, Z in latents.items():
        sil_fam, p_fam = family_stats(Z, fam_labels_all, n_perm=5000)
        family_rows.append([mode, sil_fam, p_fam])

    df_family = pd.DataFrame(
        family_rows,
        columns=["Distance", "Family_silhouette", "Family_pvalue"]
    )
    fam_csv = os.path.join(OUTDIR, "family_clustering_stats.csv")
    df_family.to_csv(fam_csv, index=False)
    print("\nFamily clustering stats saved to:", fam_csv)
    print(df_family)
else:
    df_family = None
    print("\nNo 'Family' column found for clustering stats.")


# Family dendrogram for CORR latents
if "Family" in df_sub.columns:
    fam_series = df_sub["Family"].astype(str)
    drop_tokens = {"NA", "nan", "NaN", "None", "0", ""}
    mask_fam = ~fam_series.isin(drop_tokens)

    fam_valid = fam_series[mask_fam]
    Z_corr = latents["CORR"][mask_fam]

    fam_cats = pd.Categorical(fam_valid)
    families = list(fam_cats.categories)

    if len(families) >= 2:
        centroids = []
        for fam in families:
            idx = np.where(fam_valid.values == fam)[0]
            centroids.append(Z_corr[idx].mean(axis=0))
        centroids = np.vstack(centroids)

        dist = pdist(centroids, metric="euclidean")
        link = linkage(dist, method="average")

        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        dendrogram(link, labels=families, ax=ax, leaf_rotation=90)
        ax.set_ylabel("Euclidean distance (CORR latent)")
        ax.set_title("Family dendrogram (CORR latent centroids)")

        for ext in ["png", "pdf"]:
            out_path = os.path.join(OUTDIR, f"Family_dendrogram_CORR.{ext}")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print("Saved:", out_path)
        plt.close()
    else:
        print("Not enough distinct families for dendrogram.")
else:
    print("Skipping dendrogram — no 'Family' column.")


# ================================================================
# 8. NATURE-STYLE UMAP GRID (3×8)
# ================================================================
modes = ["CORR", "EUCLID", "WASS"]
nrows = len(modes)
ncols = len(label_columns)

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(3.6 * ncols, 3.6 * nrows),
    dpi=300
)

if nrows == 1:
    axes = np.expand_dims(axes, axis=0)
if ncols == 1:
    axes = np.expand_dims(axes, axis=1)

for i, mode in enumerate(modes):
    Z = latents[mode]
    print(f"Computing UMAP for {mode}...")
    emb = compute_umap(Z)

    for j, col in enumerate(label_columns):
        ax = axes[i, j]

        # continuous
        if col in continuous_columns:
            if col not in df_sub.columns:
                ax.set_title(f"{mode} — {col} (missing)", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            vals = pd.to_numeric(df_sub[col], errors="coerce")
            mask = ~vals.isna()

            if mask.sum() == 0:
                ax.set_title(f"{mode} — {col} (no data)", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            sc = ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=vals[mask],
                cmap=plt.get_cmap(continuous_cmaps.get(col, "viridis")),
                s=30,
                alpha=0.9,
                linewidth=0.0,
                edgecolor="none",
            )
            add_kde_contour(ax, emb)

            ax.set_title(f"{mode} — {col}", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])

            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(col, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            continue

        # categorical
        if col not in df_sub.columns:
            ax.set_title(f"{mode} — {col} (missing)", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        labels_series = df_sub[col].astype(str)

        # Family: drop NA-like values only in that panel
        if col == "Family":
            drop_tokens = {"NA", "nan", "NaN", "0", "None", ""}
            mask_valid = ~labels_series.isin(drop_tokens)
            labels = labels_series[mask_valid].values
            emb_used = emb[mask_valid, :]
        else:
            labels = labels_series.values
            emb_used = emb

        if len(labels) == 0:
            ax.set_title(f"{mode} — {col} (no valid labels)", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        cats = pd.Categorical(labels)
        codes = cats.codes
        n_classes = len(cats.categories)

        colors = get_palette(col, n_classes)
        cmap = ListedColormap(colors)

        sc = ax.scatter(
            emb_used[:, 0], emb_used[:, 1],
            c=codes,
            cmap=cmap,
            s=30,
            alpha=0.9,
            linewidth=0.2,
            edgecolor="k",
        )

        add_kde_contour(ax, emb)

        ax.set_title(f"{mode} — {col}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

        handles = [
            plt.Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                color=colors[k],
                label=str(cat),
                markersize=5,
            )
            for k, cat in enumerate(cats.categories)
        ]
        leg = ax.legend(
            handles=handles,
            title=col,
            fontsize=7,
            title_fontsize=8,
            loc="best",
            frameon=True,
            framealpha=0.8,
            borderpad=0.4,
        )
        leg.get_frame().set_linewidth(0.5)

plt.tight_layout()
basename = "UMAP_NatureGrid_CORR_EUCLID_WASS_K10"
for ext in ["png", "svg", "pdf"]:
    out = os.path.join(OUTDIR, f"{basename}.{ext}")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved:", out)

plt.close()


# ================================================================
# 9. K-MEANS PER METRIC + χ² TESTS + ENRICHMENT + HEATMAPS
# ================================================================
K = 3   # clusters per metric (as before)
print(f"\n\n============= KMEANS CLUSTERING PER METRIC (K={K}) =============\n")

cluster_rows = []
chisq_rows   = []

factors_to_test = ["APOE", "sex", "risk_for_ad"]
cluster_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue/orange/green

def cramers_v(chi2, n, r, c):
    """Cramér’s V effect size."""
    return np.sqrt(chi2 / (n * (min(r - 1, c - 1))))

for mode, Z in latents.items():
    print(f"\n---- {mode}: computing {K}-means clusters ----")
    kmeans = KMeans(n_clusters=K, random_state=42)
    clusters = kmeans.fit_predict(Z)

    cluster_rows.extend([
        [mode, df_sub["MRI_Exam"].iloc[i], clusters[i]]
        for i in range(len(clusters))
    ])

    for factor in factors_to_test:
        if factor not in df_sub.columns:
            continue

        labels = df_sub[factor].astype(str).values
        valid_mask = labels != "NA"
        labs = labels[valid_mask]
        cls  = clusters[valid_mask]

        contingency = pd.crosstab(cls, labs)
        chi2, p, dof, expected = chi2_contingency(contingency)
        V = cramers_v(chi2, len(labs), contingency.shape[0], contingency.shape[1])

        enrichment = contingency.div(contingency.sum(axis=1), axis=0)

        chisq_rows.append([
            mode, factor, chi2, p, V,
            contingency.to_dict(),
            enrichment.round(3).to_dict()
        ])

        print(f"{mode}-{factor} χ² = {chi2:.3f}, p = {p:.3e}, Cramér’s V = {V:.3f}")


df_clusters = pd.DataFrame(
    cluster_rows,
    columns=["Distance", "MRI_Exam", "Cluster"]
)
df_clusters.to_csv(
    os.path.join(OUTDIR, "kmeans_cluster_assignments.csv"),
    index=False
)

df_chi = pd.DataFrame(
    chisq_rows,
    columns=["Distance", "Factor", "Chi2", "p_value", "CramersV",
             "ContingencyTable", "Enrichment"]
)
df_chi.to_csv(
    os.path.join(OUTDIR, "kmeans_chisq_results.csv"),
    index=False
)

print("\nSaved χ² test results & cluster assignments.")


# Enrichment bar plots
plt.figure(figsize=(12, 4 * len(factors_to_test)))
plot_i = 1
for factor in factors_to_test:
    plt.subplot(len(factors_to_test), 1, plot_i)
    plot_i += 1

    bar_data = df_chi[df_chi["Factor"] == factor]

    for mode in ["CORR", "EUCLID", "WASS"]:
        sub = bar_data[bar_data["Distance"] == mode]
        if len(sub) == 0:
            continue

        enrichment_dict = sub["Enrichment"].iloc[0]
        df_enrich = pd.DataFrame(enrichment_dict)

        df_enrich.T.plot(
            kind="bar", stacked=False,
            color=cluster_colors, alpha=0.8, ax=plt.gca()
        )

        plt.title(f"{factor} enrichment across clusters ({mode})")
        plt.ylabel("Proportion")
        plt.legend(title="Label", fontsize=8)

    plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR, "kmeans_cluster_bars.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()
print("Saved cluster bar plots.")


# Heatmap of contingency tables (APOE × cluster)
plt.figure(figsize=(10, 10))
for i, mode in enumerate(latents.keys()):
    plt.subplot(3, 1, i + 1)

    sub = df_chi[df_chi["Distance"] == mode]
    if len(sub) == 0:
        continue

    row = sub[sub["Factor"] == "APOE"]
    if len(row) == 0:
        continue

    contingency = pd.DataFrame(row["ContingencyTable"].iloc[0])

    sns.heatmap(contingency, annot=True, fmt="d", cmap="viridis")
    plt.title(f"{mode} – APOE × Cluster contingency")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTDIR, "kmeans_cluster_heatmap_APOE.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()
print("Saved k-means heatmap for APOE.")


# ================================================================
# 10. MULTI-METRIC CONSENSUS CLUSTERING (CORR+EUCLID+WASS)
# ================================================================
print("\n\n================ CONSENSUS CLUSTERING ================\n")

metric_list = ["CORR", "EUCLID", "WASS"]
cluster_matrix = np.zeros((n_samples, len(metric_list)), dtype=int)

for m_idx, mode in enumerate(metric_list):
    Z = latents[mode]
    kmeans = KMeans(n_clusters=K, random_state=42)
    cluster_matrix[:, m_idx] = kmeans.fit_predict(Z)

df_cluster_matrix = pd.DataFrame(
    cluster_matrix,
    columns=[f"{m}_cluster" for m in metric_list]
)
df_cluster_matrix["MRI_Exam"] = df_sub["MRI_Exam"]
df_cluster_matrix.to_csv(
    os.path.join(OUTDIR, "raw_multimetric_clusters.csv"),
    index=False
)
print("Saved raw cluster matrix → raw_multimetric_clusters.csv")

# Co-assignment matrix: proportion of metrics where i and j share cluster
coassign = np.zeros((n_samples, n_samples), float)
for i in range(n_samples):
    for j in range(n_samples):
        coassign[i, j] = np.mean(cluster_matrix[i, :] == cluster_matrix[j, :])

np.save(os.path.join(OUTDIR, "coassignment_matrix.npy"), coassign)
print("Saved coassignment matrix → coassignment_matrix.npy")

# Spectral clustering on coassignment to get consensus labels
consensus_model = SpectralClustering(
    n_clusters=K,
    affinity='precomputed',
    random_state=42
)
consensus_labels = consensus_model.fit_predict(coassign)

df_consensus = pd.DataFrame({
    "MRI_Exam": df_sub["MRI_Exam"],
    "ConsensusCluster": consensus_labels
})
df_consensus.to_csv(
    os.path.join(OUTDIR, "consensus_clusters.csv"),
    index=False
)
print("Saved consensus cluster labels → consensus_clusters.csv")

# Cluster stability = mean coassignment within each consensus cluster
stability_scores = []
for k in range(K):
    idx = np.where(consensus_labels == k)[0]
    if len(idx) < 2:
        stability_scores.append(np.nan)
        continue
    sub = coassign[np.ix_(idx, idx)]
    stab = sub[np.triu_indices(len(idx), k=1)].mean()
    stability_scores.append(stab)

df_stab = pd.DataFrame({
    "Cluster": np.arange(K),
    "Stability": stability_scores
})
df_stab.to_csv(
    os.path.join(OUTDIR, "consensus_stability.csv"),
    index=False
)
print("Saved consensus stability → consensus_stability.csv")

# Dendrogram on 1 - coassignment
dist = 1 - coassign
link = linkage(squareform(dist, checks=False), method="average")

fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
dendrogram(link, labels=df_sub["MRI_Exam"].values, leaf_rotation=90, ax=ax)
ax.set_title("Consensus Clustering Dendrogram (CORR + EUCLID + WASS)")
ax.set_ylabel("1 - coassignment")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTDIR, "consensus_dendrogram.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()
print("Saved → consensus_dendrogram.png")

# Co-assignment heatmap
plt.figure(figsize=(7, 6))
sns.heatmap(coassign, cmap="viridis", square=True)
plt.title("Co-assignment Matrix (Consensus Clustering)")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTDIR, "coassignment_heatmap.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()
print("Saved → coassignment_heatmap.png")

# Consensus enrichment vs APOE / sex / risk_for_ad
print("\n===== Consensus cluster enrichment =====\n")
consensus_rows = []

for factor in ["APOE", "sex", "risk_for_ad"]:
    if factor not in df_sub.columns:
        continue

    labels = df_sub[factor].astype(str)
    contingency = pd.crosstab(consensus_labels, labels)
    chi2, p, dof, expected = chi2_contingency(contingency)
    V = cramers_v(chi2, len(labels), contingency.shape[0], contingency.shape[1])

    consensus_rows.append([
        factor, chi2, p, V,
        contingency.to_dict()
    ])

    print(f"{factor}: χ²={chi2:.3f}, p={p:.3e}, Cramér’s V={V:.3f}")

df_consensus_enrich = pd.DataFrame(
    consensus_rows,
    columns=["Factor", "Chi2", "p_value", "CramersV", "Contingency"]
)
df_consensus_enrich.to_csv(
    os.path.join(OUTDIR, "consensus_cluster_enrichment.csv"),
    index=False
)
print("Saved → consensus_cluster_enrichment.csv")

print("\n🎉 DONE — GAUDI latent analysis + consensus clustering complete!\n")


# ================================================================
# 11. CONSENSUS FLOW ANALYSIS (Matplotlib-Only)
# ================================================================
print("\n\n================ CONSENSUS FLOW ANALYSIS (MATPLOTLIB) ================\n")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------
# Helper: compute flow counts between two clustering schemes
# ---------------------------------------------------------------
def compute_flow_matrix(A_clusters, B_clusters, K):
    """Return K×K matrix where entry (i,j) = samples in A=i AND B=j."""
    mat = np.zeros((K, K), dtype=int)
    for i in range(K):
        maskA = (A_clusters == i)
        for j in range(K):
            maskB = (B_clusters == j)
            mat[i, j] = np.sum(maskA & maskB)
    return mat


# ================================================================
# A. HEATMAP-BASED FLOW MATRICES (CORR→EUCLID→WASS→CONS)
# ================================================================
C = cluster_matrix    # from Section 10
flows = {
    "CORR→EUCLID": compute_flow_matrix(C[:,0], C[:,1], K),
    "EUCLID→WASS": compute_flow_matrix(C[:,1], C[:,2], K),
    "WASS→CONS":   compute_flow_matrix(C[:,2], consensus_labels, K),
}

plt.figure(figsize=(12, 4))
for idx, (title, mat) in enumerate(flows.items(), start=1):
    plt.subplot(1, 3, idx)
    sns.heatmap(mat, annot=True, fmt="d", cmap="viridis")
    plt.title(title)
    plt.xlabel(title.split("→")[1])
    plt.ylabel(title.split("→")[0])

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "flow_heatmaps_all.png"), dpi=300)
plt.close()
print("Saved heatmap flows → flow_heatmaps_all.png")


# ================================================================
# B. MATPLOTLIB SANKEY DIAGRAM (simple, stable, no Chrome)
# ================================================================
# We will build *three separate Sankey diagrams*, one for each flow step
# Each will show how clusters in metric A distribute into metric B

from matplotlib.sankey import Sankey

def plot_sankey(title, A_name, B_name, A_clusters, B_clusters, K, fname):
    """Create a simple sankey diagram using matplotlib."""
    mat = compute_flow_matrix(A_clusters, B_clusters, K)

    sankey = Sankey(unit=None)
    # one entry per A-cluster, cumulative flows to B clusters
    flows = []
    labels = []
    orientations = []

    # Build flow list: A_i → B_j are positive; a balancing negative at start
    for i in range(K):
        row = mat[i]
        total = row.sum()
        if total == 0:
            continue

        # Starting block
        flows.append(-total)
        labels.append(f"{A_name} {i}")
        orientations.append(0)

        # Add outgoing flows
        for j in range(K):
            if row[j] > 0:
                flows.append(row[j])
                labels.append(f"{B_name} {j}")
                orientations.append(0)

    sankey.add(
        flows=flows,
        labels=labels,
        orientations=orientations,
        trunklength=1.0,
        pathlengths=[0.75] * len(flows),
    )

    fig = plt.figure(figsize=(10, 6))
    sankey.finish()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=300)
    plt.close()


# Create three matplotlib Sankey diagrams
plot_sankey(
    "Flow: CORR → EUCLID",
    "CORR", "EUCLID",
    C[:,0], C[:,1], K,
    "sankey_corr_to_euclid.png"
)
plot_sankey(
    "Flow: EUCLID → WASS",
    "EUCLID", "WASS",
    C[:,1], C[:,2], K,
    "sankey_euclid_to_wass.png"
)
plot_sankey(
    "Flow: WASS → CONSENSUS",
    "WASS", "CONS",
    C[:,2], consensus_labels, K,
    "sankey_wass_to_consensus.png"
)

print("Saved 3 Sankey diagrams (matplotlib).")


# ================================================================
# C. UMAP Overlays by Consensus Cluster
# ================================================================
print("Creating consensus-colored UMAPs...")

for mode in ["CORR", "EUCLID", "WASS"]:
    Z = latents[mode]
    emb = compute_umap(Z)

    plt.figure(figsize=(5,4), dpi=300)
    plt.scatter(
        emb[:,0], emb[:,1],
        c=consensus_labels,
        cmap="tab10",
        s=40,
        alpha=0.9
    )
    add_kde_contour(plt.gca(), emb)
    plt.title(f"{mode} latent space colored by CONSENSUS")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()

    fname = f"UMAP_{mode}_consensus_colored.png"
    plt.savefig(os.path.join(OUTDIR, fname), dpi=300)
    plt.close()
    print("Saved:", fname)

print("\n🎉 DONE — Consensus flow analysis (Matplotlib-only) complete!\n")
