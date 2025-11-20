#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAUDI Latent Space Analysis and Consensus Clustering (Multi-K)
==============================================================

For each graph kNN (K_graph in {10, 20, 30, 50}):

1.  Imports and global paths
2.  Load metadata and align with graphs
3.  Load CORR / EUCLID / WASS latents for this K_graph
4.  Clean metadata (categorical & continuous)
5.  UMAP helper + KDE contours
6.  Latent metrics: silhouette + 5-fold CV AUC (APOE / sex / risk_for_ad)
7.  Family clustering: silhouette + permutation p + dendrogram
8.  Nature-style 3×N UMAP grid (CORR/EUCLID/WASS × labels)
9.  K-means clustering per metric + χ² tests + enrichment
10. Multi-metric consensus clustering (CORR+EUCLID+WASS)
11. Consensus flow analysis (Matplotlib-only)
12. Cluster composition & centroid distances tables
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

# -------------------------
# Paths (adapt as needed)
# -------------------------
ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation"
COLUMNS_ROOT = os.path.join(ROOT, "columns4gaudi111825", "columna-analyses111925")
MDATA_PATH = os.path.join(ROOT, "processed_graph_data", "metadata_with_PCs.xlsx")

# Graph k values to loop over
GRAPH_K_LIST = [4, 6, 8, 10, 12, 20, 25, 30, 50] 
METRICS = ["CORR", "EUCLID", "WASS"]

# Number of clusters in latent space (biological clusters, not kNN)
N_CLUSTERS = 3

# visualization columns
categorical_columns = ["APOE", "genotype", "sex", "risk_for_ad", "ethnicity", "Family"]
continuous_columns  = ["age", "BMI"]
label_columns       = categorical_columns + continuous_columns

# ================================================================
# 2. LOAD METADATA (ONCE) AND DEFINE HELPERS
# ================================================================
print("Loading metadata from:", MDATA_PATH)
df_all = pd.read_excel(MDATA_PATH)
df_all["MRI_Exam"] = df_all["MRI_Exam"].astype(str).str.zfill(5)
print("Metadata rows:", len(df_all))

# Categorical cleaning tokens
na_tokens = {"", "nan", "NaN", "None", "NA"}

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
# 3. UMAP + METRIC HELPERS
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

def cramers_v(chi2, n, r, c):
    """Cramér’s V effect size."""
    return np.sqrt(chi2 / (n * (min(r - 1, c - 1))))

def compute_flow_matrix(A_clusters, B_clusters, n_clusters):
    """Return n_clusters×n_clusters matrix where entry (i,j) = samples in A=i AND B=j."""
    mat = np.zeros((n_clusters, n_clusters), dtype=int)
    for i in range(n_clusters):
        maskA = (A_clusters == i)
        for j in range(n_clusters):
            maskB = (B_clusters == j)
            mat[i, j] = np.sum(maskA & maskB)
    return mat

from matplotlib.sankey import Sankey

def plot_sankey_matplotlib(OUTDIR, title, A_name, B_name,
                           A_clusters, B_clusters,
                           n_clusters, fname):
    """Simple sankey diagram using matplotlib."""
    mat = compute_flow_matrix(A_clusters, B_clusters, n_clusters)

    fig = plt.figure(figsize=(10, 6))
    sankey = Sankey(unit=None)

    flows = []
    labels = []
    orientations = []

    for i in range(n_clusters):
        row = mat[i]
        total = row.sum()
        if total == 0:
            continue

        # starting negative flow = total size of cluster i
        flows.append(-float(total))
        labels.append(f"{A_name} {i}")
        orientations.append(0)

        for j in range(n_clusters):
            if row[j] > 0:
                flows.append(float(row[j]))
                labels.append(f"{B_name} {j}")
                orientations.append(0)

    if len(flows) == 0:
        print(f"No flows for {title}, skipping Sankey.")
        plt.close(fig)
        return

    sankey.add(
        flows=flows,
        labels=labels,
        orientations=orientations,
        trunklength=1.0,
        pathlengths=[0.75] * len(flows),
    )

    sankey.finish()
    plt.title(title)
    plt.tight_layout()
    out_path = os.path.join(OUTDIR, fname)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved Sankey:", out_path)

# Binary factors for AUC / silhouette
binary_factors = {
    "APOE": ("E4+", "E4-"),
    "sex": ("F", "M"),
    "risk_for_ad": ("1", "0"),  # cast as string
}

# ================================================================
# MAIN LOOP OVER GRAPH K
# ================================================================
for K_graph in GRAPH_K_LIST:
    print("\n\n==================== PROCESSING K = {} ====================\n".format(K_graph))

    # ----------------------------------------------------------------
    # Paths for this K_graph
    # ----------------------------------------------------------------
    GRAPHS_DIR = os.path.join(COLUMNS_ROOT, "graphs_knn", f"k{K_graph}")
    MD_PT      = os.path.join(GRAPHS_DIR,
                              f"md_shared_knn_k{K_graph}_corr_euclid_wass.pt")

    LATENT_ROOT = os.path.join(COLUMNS_ROOT, f"latent_k{K_graph}")
    latent_paths = {
        "CORR":   os.path.join(LATENT_ROOT, "latent_epochs_Joint_CORR",
                               "latent_final_Joint_CORR.npy"),
        "EUCLID": os.path.join(LATENT_ROOT, "latent_epochs_Joint_EUCLID",
                               "latent_final_Joint_EUCLID.npy"),
        "WASS":   os.path.join(LATENT_ROOT, "latent_epochs_Joint_WASS",
                               "latent_final_Joint_WASS.npy"),
    }

    OUTDIR = os.path.join(COLUMNS_ROOT, f"results_K{K_graph}")
    os.makedirs(OUTDIR, exist_ok=True)
    print("Output dir:", OUTDIR)

    # ----------------------------------------------------------------
    # Load graphs and align metadata
    # ----------------------------------------------------------------
    print("Loading MD graphs from:", MD_PT)
    graphs = torch.load(MD_PT, map_location="cpu")
    subject_ids = [str(getattr(g, "subject_id")).zfill(5) for g in graphs]
    print("Found subject IDs in graphs:", len(subject_ids))
    print("Example IDs:", subject_ids[:5])

    df_sub = df_all[df_all["MRI_Exam"].isin(subject_ids)].copy()
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

    # ----------------------------------------------------------------
    # Clean metadata (categorical columns)
    # ----------------------------------------------------------------
    for col in categorical_columns:
        if col in df_sub.columns:
            s = df_sub[col].astype(str).str.strip()
            s = s.replace({val: "NA" for val in na_tokens})
            df_sub[col] = s

    if "Family" in df_sub.columns:
        fam_vals = set(df_sub["Family"].unique())
        non_na_fams = fam_vals - {"NA"}
        if len(non_na_fams) == 0:
            print("Dropping 'Family' — no non-NA entries found for this K.")
            categorical_cols_used = [c for c in categorical_columns if c != "Family"]
        else:
            print("Family categories (including NA):", fam_vals)
            categorical_cols_used = categorical_columns
    else:
        categorical_cols_used = [c for c in categorical_columns if c != "Family"]

    label_cols_used = categorical_cols_used + continuous_columns

    # ----------------------------------------------------------------
    # Load latents and align
    # ----------------------------------------------------------------
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
    # 6. LATENT METRICS: SILHOUETTE + 5-FOLD CV AUC
    # ================================================================
    metric_rows = []
    for mode, Z in latents.items():
        print(f"\n=== Metrics on raw latents: {mode}, K_graph={K_graph} ===")
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

            metric_rows.append([K_graph, mode, factor, sil, auc_mean, auc_std])

    df_metrics = pd.DataFrame(
        metric_rows,
        columns=["K_graph", "Distance", "Factor", "Silhouette", "AUC_mean", "AUC_std"]
    )
    metrics_csv = os.path.join(OUTDIR, f"distance_metrics_K{K_graph}.csv")
    df_metrics.to_csv(metrics_csv, index=False)
    print("\nSaved metrics to:", metrics_csv)
    print(df_metrics)

    # ================================================================
    # 7. FAMILY CLUSTERING STATS + DENDROGRAM
    # ================================================================
    family_rows = []
    if "Family" in df_sub.columns:
        fam_labels_all = df_sub["Family"].astype(str).values
        for mode, Z in latents.items():
            sil_fam, p_fam = family_stats(Z, fam_labels_all, n_perm=5000)
            family_rows.append([K_graph, mode, sil_fam, p_fam])

        df_family = pd.DataFrame(
            family_rows,
            columns=["K_graph", "Distance", "Family_silhouette", "Family_pvalue"]
        )
        fam_csv = os.path.join(OUTDIR, f"family_clustering_stats_K{K_graph}.csv")
        df_family.to_csv(fam_csv, index=False)
        print("\nFamily clustering stats saved to:", fam_csv)
        print(df_family)

        # Family dendrogram for CORR latents
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
            ax.set_title(f"Family dendrogram (CORR latent centroids, K={K_graph})")

            for ext in ["png", "pdf"]:
                out_path = os.path.join(OUTDIR, f"Family_dendrogram_CORR_K{K_graph}.{ext}")
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                print("Saved:", out_path)
            plt.close()
        else:
            print("Not enough distinct families for dendrogram.")
    else:
        print("\nNo 'Family' column found for clustering stats.")

    # ================================================================
    # 8. NATURE-STYLE UMAP GRID (3×labels)
    # ================================================================
    modes = METRICS
    nrows = len(modes)
    ncols = len(label_cols_used)

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
        print(f"Computing UMAP for {mode} (K_graph={K_graph})...")
        emb = compute_umap(Z)

        for j, col in enumerate(label_cols_used):
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
                labels_plot = labels_series[mask_valid].values
                emb_used = emb[mask_valid, :]
            else:
                labels_plot = labels_series.values
                emb_used = emb

            if len(labels_plot) == 0:
                ax.set_title(f"{mode} — {col} (no valid labels)", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            cats = pd.Categorical(labels_plot)
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
    basename = f"UMAP_NatureGrid_CORR_EUCLID_WASS_K{K_graph}"
    for ext in ["png", "svg", "pdf"]:
        out = os.path.join(OUTDIR, f"{basename}.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print("Saved:", out)

    plt.close()

    # ================================================================
    # 9. K-MEANS PER METRIC + χ² TESTS + ENRICHMENT
    # ================================================================
    print(f"\n\n============= KMEANS CLUSTERING PER METRIC (N_CLUSTERS={N_CLUSTERS}) =============\n")

    cluster_rows = []
    chisq_rows   = []
    kmeans_labels = {}
    factors_to_test = ["APOE", "sex", "risk_for_ad"]
    cluster_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue/orange/green

    for mode, Z in latents.items():
        print(f"\n---- {mode}: computing {N_CLUSTERS}-means clusters ----")
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
        clusters = kmeans.fit_predict(Z)
        kmeans_labels[mode] = clusters

        cluster_rows.extend([
            [K_graph, mode, df_sub["MRI_Exam"].iloc[i], clusters[i]]
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
            if contingency.size == 0:
                continue

            chi2, p, dof, expected = chi2_contingency(contingency)
            V = cramers_v(chi2, len(labs), contingency.shape[0], contingency.shape[1])

            enrichment = contingency.div(contingency.sum(axis=1), axis=0)

            chisq_rows.append([
                K_graph, mode, factor, chi2, p, V,
                contingency.to_dict(),
                enrichment.round(3).to_dict()
            ])

            print(f"{mode}-{factor} χ² = {chi2:.3f}, p = {p:.3e}, Cramér’s V = {V:.3f}")

    df_clusters = pd.DataFrame(
        cluster_rows,
        columns=["K_graph", "Distance", "MRI_Exam", "Cluster"]
    )
    df_clusters.to_csv(
        os.path.join(OUTDIR, f"kmeans_cluster_assignments_K{K_graph}.csv"),
        index=False
    )

    df_chi = pd.DataFrame(
        chisq_rows,
        columns=["K_graph", "Distance", "Factor", "Chi2", "p_value", "CramersV",
                 "ContingencyTable", "Enrichment"]
    )
    df_chi.to_csv(
        os.path.join(OUTDIR, f"kmeans_chisq_results_K{K_graph}.csv"),
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

        for mode in METRICS:
            sub = bar_data[bar_data["Distance"] == mode]
            if len(sub) == 0:
                continue

            enrichment_dict = sub["Enrichment"].iloc[0]
            df_enrich = pd.DataFrame(enrichment_dict)

            df_enrich.T.plot(
                kind="bar", stacked=False,
                color=cluster_colors, alpha=0.8, ax=plt.gca()
            )

            plt.title(f"{factor} enrichment across clusters ({mode}, K={K_graph})")
            plt.ylabel("Proportion")
            plt.legend(title="Label", fontsize=8)

        plt.tight_layout()

    plt.savefig(
        os.path.join(OUTDIR, f"kmeans_cluster_bars_K{K_graph}.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("Saved cluster bar plots.")

    # Heatmap of contingency tables (APOE × cluster)
    plt.figure(figsize=(10, 10))
    for i, mode in enumerate(METRICS):
        plt.subplot(3, 1, i + 1)

        sub = df_chi[(df_chi["Distance"] == mode) & (df_chi["Factor"] == "APOE")]
        if len(sub) == 0:
            plt.title(f"{mode} – APOE × Cluster contingency (no data)")
            continue

        contingency = pd.DataFrame(sub["ContingencyTable"].iloc[0])

        sns.heatmap(contingency, annot=True, fmt=".0f", cmap="viridis")
        plt.title(f"{mode} – APOE × Cluster contingency (K={K_graph})")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTDIR, f"kmeans_cluster_heatmap_APOE_K{K_graph}.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("Saved k-means heatmap for APOE.")

    # ================================================================
    # 10. MULTI-METRIC CONSENSUS CLUSTERING (CORR+EUCLID+WASS)
    # ================================================================
    print("\n\n================ CONSENSUS CLUSTERING (K_graph = {}) ================\n".format(K_graph))

    metric_list = METRICS
    cluster_matrix = np.zeros((n_samples, len(metric_list)), dtype=int)
    for m_idx, mode in enumerate(metric_list):
        cluster_matrix[:, m_idx] = kmeans_labels[mode]

    df_cluster_matrix = pd.DataFrame(
        cluster_matrix,
        columns=[f"{m}_cluster" for m in metric_list]
    )
    df_cluster_matrix["MRI_Exam"] = df_sub["MRI_Exam"]
    df_cluster_matrix.to_csv(
        os.path.join(OUTDIR, f"raw_multimetric_clusters_K{K_graph}.csv"),
        index=False
    )
    print("Saved raw cluster matrix.")

    # Co-assignment matrix: proportion of metrics where i and j share cluster
    coassign = np.zeros((n_samples, n_samples), float)
    for i in range(n_samples):
        for j in range(n_samples):
            coassign[i, j] = np.mean(cluster_matrix[i, :] == cluster_matrix[j, :])

    np.save(os.path.join(OUTDIR, f"coassignment_matrix_K{K_graph}.npy"), coassign)
    print("Saved coassignment matrix.")

    # Spectral clustering on coassignment to get consensus labels
    consensus_model = SpectralClustering(
        n_clusters=N_CLUSTERS,
        affinity='precomputed',
        random_state=42
    )
    consensus_labels = consensus_model.fit_predict(coassign)

    df_consensus = pd.DataFrame({
        "MRI_Exam": df_sub["MRI_Exam"],
        "ConsensusCluster": consensus_labels
    })
    df_consensus.to_csv(
        os.path.join(OUTDIR, f"consensus_clusters_K{K_graph}.csv"),
        index=False
    )
    print("Saved consensus cluster labels.")

    # Cluster stability = mean coassignment within each consensus cluster
    stability_scores = []
    for k_idx in range(N_CLUSTERS):
        idx = np.where(consensus_labels == k_idx)[0]
        if len(idx) < 2:
            stability_scores.append(np.nan)
            continue
        sub = coassign[np.ix_(idx, idx)]
        stab = sub[np.triu_indices(len(idx), k=1)].mean()
        stability_scores.append(stab)

    df_stab = pd.DataFrame({
        "Cluster": np.arange(N_CLUSTERS),
        "Stability": stability_scores
    })
    df_stab.to_csv(
        os.path.join(OUTDIR, f"consensus_stability_K{K_graph}.csv"),
        index=False
    )
    print("Saved consensus stability.")

    # Dendrogram on 1 - coassignment
    dist = 1 - coassign
    link = linkage(squareform(dist, checks=False), method="average")

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    dendrogram(link, labels=df_sub["MRI_Exam"].values, leaf_rotation=90, ax=ax)
    ax.set_title(f"Consensus Clustering Dendrogram (CORR+EUCLID+WASS, K={K_graph})")
    ax.set_ylabel("1 - coassignment")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTDIR, f"consensus_dendrogram_K{K_graph}.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("Saved consensus dendrogram.")

    # Co-assignment heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(coassign, cmap="viridis", square=True)
    plt.title(f"Co-assignment Matrix (Consensus Clustering, K={K_graph})")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTDIR, f"coassignment_heatmap_K{K_graph}.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("Saved coassignment heatmap.")

    # Consensus enrichment vs APOE / sex / risk_for_ad
    print("\n===== Consensus cluster enrichment (K_graph = {}) =====\n".format(K_graph))
    consensus_rows = []

    for factor in ["APOE", "sex", "risk_for_ad"]:
        if factor not in df_sub.columns:
            continue

        labels_factor = df_sub[factor].astype(str)
        contingency = pd.crosstab(consensus_labels, labels_factor)
        if contingency.size == 0:
            continue

        chi2, p, dof, expected = chi2_contingency(contingency)
        V = cramers_v(chi2, len(labels_factor), contingency.shape[0], contingency.shape[1])

        consensus_rows.append([
            K_graph, factor, chi2, p, V,
            contingency.to_dict()
        ])

        print(f"{factor}: χ²={chi2:.3f}, p={p:.3e}, Cramér’s V={V:.3f}")

    df_consensus_enrich = pd.DataFrame(
        consensus_rows,
        columns=["K_graph", "Factor", "Chi2", "p_value", "CramersV", "Contingency"]
    )
    df_consensus_enrich.to_csv(
        os.path.join(OUTDIR, f"consensus_cluster_enrichment_K{K_graph}.csv"),
        index=False
    )
    print("Saved consensus cluster enrichment.")

    # ================================================================
    # 11. CONSENSUS FLOW ANALYSIS (Matplotlib-Only)
    # ================================================================
    print("\n\n================ CONSENSUS FLOW ANALYSIS (MATPLOTLIB, K_graph = {}) ================\n".format(K_graph))

    # A. Heatmap-based flow matrices
    flows = {
        "CORR→EUCLID": compute_flow_matrix(cluster_matrix[:,0], cluster_matrix[:,1], N_CLUSTERS),
        "EUCLID→WASS": compute_flow_matrix(cluster_matrix[:,1], cluster_matrix[:,2], N_CLUSTERS),
        "WASS→CONS":   compute_flow_matrix(cluster_matrix[:,2], consensus_labels, N_CLUSTERS),
    }

    plt.figure(figsize=(12, 4))
    for idx, (title, mat) in enumerate(flows.items(), start=1):
        plt.subplot(1, 3, idx)
        sns.heatmap(mat, annot=True, fmt=".0f", cmap="viridis")
        plt.title(f"{title} (K={K_graph})")
        plt.xlabel(title.split("→")[1])
        plt.ylabel(title.split("→")[0])

    plt.tight_layout()
    heatmap_path = os.path.join(OUTDIR, f"flow_heatmaps_all_K{K_graph}.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print("Saved heatmap flows →", heatmap_path)

    # B. Matplotlib Sankey diagrams
    plot_sankey_matplotlib(
        OUTDIR,
        f"Flow: CORR → EUCLID (K={K_graph})",
        "CORR", "EUCLID",
        cluster_matrix[:,0], cluster_matrix[:,1],
        N_CLUSTERS,
        f"sankey_corr_to_euclid_K{K_graph}.png"
    )
    plot_sankey_matplotlib(
        OUTDIR,
        f"Flow: EUCLID → WASS (K={K_graph})",
        "EUCLID", "WASS",
        cluster_matrix[:,1], cluster_matrix[:,2],
        N_CLUSTERS,
        f"sankey_euclid_to_wass_K{K_graph}.png"
    )
    plot_sankey_matplotlib(
        OUTDIR,
        f"Flow: WASS → CONSENSUS (K={K_graph})",
        "WASS", "CONS",
        cluster_matrix[:,2], consensus_labels,
        N_CLUSTERS,
        f"sankey_wass_to_consensus_K{K_graph}.png"
    )

    # C. UMAP overlays colored by consensus cluster
    print("Creating consensus-colored UMAPs...")
    for mode in METRICS:
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
        plt.title(f"{mode} latent space colored by CONSENSUS (K={K_graph})")
        plt.xticks([]); plt.yticks([])
        plt.tight_layout()

        fname = f"UMAP_{mode}_consensus_colored_K{K_graph}.png"
        plt.savefig(os.path.join(OUTDIR, fname), dpi=300)
        plt.close()
        print("Saved:", fname)

    # ================================================================
    # 12. CLUSTER COMPOSITION & CENTROID DISTANCES
    # ================================================================
    print(f"\n==== Saving cluster composition & centroid distances for K={K_graph} ====\n")

    comp_rows = []
    factors_for_comp = ["APOE", "sex", "risk_for_ad", "genotype", "ethnicity"]

    for metric in METRICS:
        labels_metric = kmeans_labels[metric]
        Z = latents[metric]

        for c_idx in range(N_CLUSTERS):
            idx = np.where(labels_metric == c_idx)[0]
            row = {
                "K_graph": K_graph,
                "metric": metric,
                "cluster": c_idx,
                "size": len(idx),
                "pct": len(idx) / len(labels_metric) if len(labels_metric) > 0 else np.nan,
            }
            for factor in factors_for_comp:
                if factor in df_sub.columns:
                    vc = df_sub[factor].iloc[idx].astype(str).value_counts(normalize=True)
                    for cat, pct in vc.items():
                        row[f"{factor}_{cat}"] = round(pct, 3)
            comp_rows.append(row)

        # centroid distances for this metric
        if Z.shape[0] > 0:
            centroids = []
            for c_idx in range(N_CLUSTERS):
                idx = np.where(labels_metric == c_idx)[0]
                if len(idx) == 0:
                    # use NaNs for empty cluster centroid
                    centroids.append(np.full(Z.shape[1], np.nan))
                else:
                    centroids.append(Z[idx].mean(axis=0))
            centroids = np.vstack(centroids)
            # if some centroids are NaN, pdist will propagate NaN; that's fine
            dist_mat = squareform(pdist(centroids, metric="euclidean"))
            df_dist = pd.DataFrame(
                dist_mat,
                columns=[f"{metric}_C{c}" for c in range(N_CLUSTERS)],
                index=[f"{metric}_C{c}" for c in range(N_CLUSTERS)],
            )
            dist_path = os.path.join(OUTDIR, f"cluster_centroid_distances_{metric}_K{K_graph}.csv")
            df_dist.to_csv(dist_path)
            print("Saved centroid distances:", dist_path)

    df_comp = pd.DataFrame(comp_rows)
    comp_path = os.path.join(OUTDIR, f"cluster_composition_K{K_graph}.csv")
    df_comp.to_csv(comp_path, index=False)
    print("Saved cluster composition:", comp_path)

    print(f"\nFinished K={K_graph} successfully.\n")

print("\n🎉 DONE — GAUDI multi-K latent analysis + consensus clustering complete!\n")
