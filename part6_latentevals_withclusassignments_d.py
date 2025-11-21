#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PART 6 — LATENT EVALUATION FOR A GIVEN K
========================================
Patched version with:

    • Global Palette A for binary factors:
        0 / "E4-" → green (#2ecc71)
        1 / "E4+" → purple (#8e44ad)

    • Legends for categorical variables (APOE, sex, etc.)
    • Colorbars for continuous variables (age, BMI)
    • UMAP + PCA grid plots per metric × factor
    • kNN classification (APOE, risk_for_ad)
    • Latent–trait correlations (age, BMI, APOE, risk_for_ad)
    • Cluster trait composition (k-means & consensus)
    • APOE × Cluster contingency heatmaps (robust to empty tables)
    • Silhouette scores per distance metric
    • Logistic regression AUC comparison (APOE, sex, risk_for_ad)

Outputs are saved to:

    latent_eval_K{K}/plots/
    latent_eval_K{K}/tables/
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

# Silence pandas downcasting warning
pd.set_option("future.no_silent_downcasting", True)

# For nicer plots
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 300

# ================================================================
# CONFIGURATION
# ================================================================
GRAPH_K = 25                      # <<< change if needed
METRICS = ["CORR", "EUCLID", "WASS"]
FOCUS_METRIC = "EUCLID"           # used for latent–trait corr

ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation"
COLUMNS_ROOT = os.path.join(ROOT, "columns4gaudi111825", "columna-analyses111925")
MDATA_PATH = os.path.join(ROOT, "processed_graph_data", "metadata_with_PCs.xlsx")

EVAL_ROOT = os.path.join(COLUMNS_ROOT, f"latent_eval_K{GRAPH_K}")
PLOT_DIR = os.path.join(EVAL_ROOT, "plots")
TABLE_DIR = os.path.join(EVAL_ROOT, "tables")

for d in [EVAL_ROOT, PLOT_DIR, TABLE_DIR]:
    os.makedirs(d, exist_ok=True)

RES_DIR = os.path.join(COLUMNS_ROOT, f"results_K{GRAPH_K}")

# Traits of interest
TRAIT_COLS_CONT = ["age", "BMI"]
TRAIT_COLS_BIN = ["APOE", "sex", "risk_for_ad"]
TRAIT_COLS_EXTRA = ["ethnicity", "Family"]  # only plotted if present

# Global Palette A for binary factors
PALETTE_A = {
    0: "#2ecc71",   # green → E4-
    1: "#8e44ad"    # purple → E4+
}

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


def knn_latent_classification(Z, labels, n_neighbors=5):
    """5-fold CV kNN classification on latents for a categorical label."""
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


def scatter_binary_paletteA(ax, x, y, labels, title, label_order=None):
    """
    Scatter for binary labels using global Palette A:
        0 → green, 1 → purple
    labels: already 0/1 or mapped to that.
    """
    labels = np.asarray(labels)
    uvals = np.unique(labels[~pd.isna(labels)])
    if label_order is None:
        uvals = sorted(uvals)
    else:
        uvals = label_order

    for code in uvals:
        mask = labels == code
        ax.scatter(
            x[mask],
            y[mask],
            c=PALETTE_A.get(code, "#333333"),
            s=35,
            alpha=0.9,
            edgecolor="k",
            linewidth=0.15,
            label=str(code),
        )
    ax.set_title(title, fontsize=8)
    ax.legend(title="Group", fontsize=6, title_fontsize=7, loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])


def scatter_trait_with_legend_or_cbar(ax, x, y, df, factor):
    """
    Handles both categorical (legend) and continuous (colorbar) factors.
    Uses Palette A for binary APOE/sex/risk_for_ad.
    """
    if factor not in df.columns:
        ax.text(0.5, 0.5, f"{factor}\nmissing", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    series = df[factor]

    # Continuous factor → colorbar
    if factor in TRAIT_COLS_CONT:
        vals = pd.to_numeric(series, errors="coerce")
        sc = ax.scatter(
            x, y,
            c=vals,
            cmap="viridis",
            s=35,
            edgecolor="k",
            linewidth=0.15,
            alpha=0.9
        )
        cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.01)
        cb.set_label(factor, fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Categorical factor
    labels = series.astype(str)
    cats = pd.Categorical(labels)
    codes = cats.codes

    # If binary and in TRAIT_COLS_BIN → Palette A
    if factor in TRAIT_COLS_BIN and len(cats.categories) == 2:
        cat_to_code = {cats.categories[0]: 0, cats.categories[1]: 1}
        mapped_codes = np.array([cat_to_code[c] for c in cats])
        scatter_binary_paletteA(
            ax, x, y, mapped_codes,
            title=factor,
            label_order=[0, 1]
        )
        # Relabel legend entries to category names
        handles, _ = ax.get_legend_handles_labels()
        labels_leg = [str(cats.categories[0]), str(cats.categories[1])]
        ax.legend(
            handles,
            labels_leg,
            title=factor,
            fontsize=6,
            title_fontsize=7,
            loc="upper right",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Multi-class categorical → tab10 palette + legend
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in codes]

    ax.scatter(
        x, y,
        c=colors,
        s=35,
        edgecolor="k",
        linewidth=0.15,
        alpha=0.9,
    )

    handles = []
    for i, cat in enumerate(cats.categories):
        handles.append(
            plt.Line2D(
                [0], [0], marker="o", linestyle="",
                color=cmap(i % 10),
                label=str(cat),
                markersize=4,
            )
        )

    ax.legend(
        handles=handles,
        title=factor,
        fontsize=6,
        title_fontsize=7,
        loc="upper right",
        frameon=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])


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
subject_ids = [str(getattr(g, "subject_id")).zfill(5) for g in graphs]

df_sub = df_all[df_all["MRI_Exam"].isin(subject_ids)].copy()
df_sub["__order"] = df_sub["MRI_Exam"].apply(lambda s: subject_ids.index(s))
df_sub = df_sub.sort_values("__order").reset_index(drop=True)
df_sub.drop(columns="__order", inplace=True)
df_sub["MRI_Exam"] = df_sub["MRI_Exam"].astype(str)

print("Aligned metadata shape:", df_sub.shape)

# ================================================================
# LOAD LATENTS FOR CORR/EUCLID/WASS
# ================================================================
Z_dict = {}
for metric in METRICS:
    path = os.path.join(
        COLUMNS_ROOT, f"latent_k{GRAPH_K}",
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
# UMAP for clustering panels (Nature Methods style)
# ================================================================

UMAP_emb = {}  # store embeddings per metric

for metric, Z in Z_dict.items():
    Zs = StandardScaler().fit_transform(Z)
    emb = compute_umap(Zs, n_neighbors=15, min_dist=0.05)
    UMAP_emb[metric] = emb


def plot_kmeans_clusters_umap(metric, df_kmeans, df_sub, savepath):
    """UMAP + KMeans clusters with Option A styling."""
    
    # Points
    emb = UMAP_emb[metric]
    x = emb[:,0]
    y = emb[:,1]
    
    # Align clusters
    sub = df_kmeans[df_kmeans["Distance"] == metric].copy()
    df_m = pd.merge(df_sub, sub, on="MRI_Exam", how="inner")
    
    # reorder embeddings to df_m order
    idx = [df_sub.index[df_sub["MRI_Exam"] == sid].tolist()[0]
           for sid in df_m["MRI_Exam"]]
    xs = x[idx]
    ys = y[idx]
    clusters = df_m["Cluster"].values
    
    # Figure
    fig, ax = plt.subplots(figsize=(6,4), dpi=250)
    
    # Density (level-set contours)
    xx, yy = np.mgrid[
        xs.min()-0.5:xs.max()+0.5:200j,
        ys.min()-0.5:ys.max()+0.5:200j
    ]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([xs, ys])
    kernel = scipy.stats.gaussian_kde(values)
    Zk = np.reshape(kernel(positions).T, xx.shape)
    ax.contour(xx, yy, Zk, levels=14, linewidths=0.6,
               colors="#cccccc", alpha=0.8)
    
    # Colors
    palette = ["#3498db", "#e67e22", "#2ecc71"]   # consistent palette
    colors = [palette[c] for c in clusters]

    ax.scatter(xs, ys, c=colors, s=40,
               edgecolor="white", linewidth=0.3, alpha=0.95)

    ax.set_title(f"{metric}: KMeans clusters (UMAP)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # Legend
    handles = [plt.Line2D([0],[0],marker='o',color=palette[c],lw=0,
                          label=f"Cluster {c}", markersize=6)
               for c in sorted(df_m["Cluster"].unique())]
    ax.legend(handles=handles, fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()
    print("Saved:", savepath)



# ================================================================
# PART 1 — UMAP + PCA GRID WITH LEGENDS/COLORBARS
# ================================================================
print("\n=== PART 1: UMAP+PCA with legends/colorbars ===")

TRAITS_FOR_GRID = ["APOE", "genotype", "sex", "risk_for_ad",
                   "ethnicity", "Family", "age", "BMI"]

# UMAP grid (like your "Nature" panel)
for proj_name, proj_func in [("UMAP", compute_umap), ("PCA", None)]:
    fig, axes = plt.subplots(
        nrows=len(METRICS),
        ncols=len(TRAITS_FOR_GRID),
        figsize=(3.0 * len(TRAITS_FOR_GRID) / 2.0,
                 3.0 * len(METRICS) / 2.0),
        dpi=300,
        constrained_layout=True,
    )

    for i, metric in enumerate(METRICS):
        Z = StandardScaler().fit_transform(Z_dict[metric])
        if proj_name == "UMAP":
            emb = proj_func(Z)
        else:
            pcs = PCA(n_components=2).fit_transform(Z)
            emb = pcs

        for j, fac in enumerate(TRAITS_FOR_GRID):
            ax = axes[i, j] if len(METRICS) > 1 else axes[j]
            if proj_name == "UMAP":
                title = f"{metric} — {fac}"
            else:
                title = f"{metric} — {fac}"

            scatter_trait_with_legend_or_cbar(
                ax,
                emb[:, 0],
                emb[:, 1],
                df_sub,
                fac,
            )
            ax.set_title(title, fontsize=7)

        axes[i, 0].set_ylabel(metric, fontsize=8)

    out = os.path.join(
        PLOT_DIR,
        f"{proj_name}_NatureGrid_CORR_EUCLID_WASS_withLegends.png",
    )
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved", out)

# Additionally, save per-metric, per-factor PCA/UMAP like before
for metric, Z in Z_dict.items():
    Zs = StandardScaler().fit_transform(Z)
    emb_umap = compute_umap(Zs)
    pcs = PCA(n_components=2).fit_transform(Zs)

    for fac in ["APOE", "risk_for_ad", "age", "BMI"]:
        # UMAP
        fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
        scatter_trait_with_legend_or_cbar(ax, emb_umap[:, 0], emb_umap[:, 1], df_sub, fac)
        ax.set_title(f"K={GRAPH_K}, {metric} UMAP by {fac}")
        plt.tight_layout()
        out = os.path.join(PLOT_DIR, f"UMAP_{metric}_{fac}.png")
        plt.savefig(out)
        plt.close()

        # PCA
        fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
        scatter_trait_with_legend_or_cbar(ax, pcs[:, 0], pcs[:, 1], df_sub, fac)
        ax.set_title(f"K={GRAPH_K}, {metric} PCA by {fac}")
        plt.tight_layout()
        out = os.path.join(PLOT_DIR, f"PCA_{metric}_{fac}.png")
        plt.savefig(out)
        plt.close()

# ================================================================
# PART 2 — kNN CLASSIFICATION (APOE, risk_for_ad)
# ================================================================
print("\n=== PART 2: kNN classification ===")

rows = []
for metric, Z in Z_dict.items():
    Zs = StandardScaler().fit_transform(Z)
    for fac in ["APOE", "risk_for_ad"]:
        labels = df_sub[fac].astype(str).replace({"nan": "NA", "NaN": "NA"})
        acc, f1 = knn_latent_classification(Zs, labels)
        rows.append([GRAPH_K, metric, fac, acc, f1])

df_knn = pd.DataFrame(rows, columns=["K", "Metric", "Factor", "Accuracy", "MacroF1"])
df_knn.to_csv(os.path.join(TABLE_DIR, "knn_classification_results.csv"), index=False)
print("Saved kNN classification table.")

# ================================================================
# PART 3 — LATENT–TRAIT CORRELATIONS (FOCUS_METRIC)
# ================================================================
print("\n=== PART 3: latent–trait correlations ===")

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

    # APOE binary: E4- → 0, E4+ → 1 (if present)
    ap_series = df_sub["APOE"].astype(str)
    ap_map = {"E4-": 0, "E4+": 1}
    ap_bin = ap_series.map(ap_map)
    row["r_APOE_bin"] = safe_pearson(vec, ap_bin)

    risk_ord = df_sub["risk_for_ad"].astype(str).replace(
        {"0": 0, "1": 1, "2": 2, "3": 3}
    )
    risk_ord = pd.to_numeric(risk_ord, errors="coerce")
    row["r_risk_ord"] = safe_pearson(vec, risk_ord)

    corr_rows.append(row)

df_corr = pd.DataFrame(corr_rows)
df_corr.to_csv(
    os.path.join(TABLE_DIR, "latent_trait_correlations.csv"),
    index=False,
)

plt.figure(figsize=(6, 6), dpi=300)
heatmat = df_corr.set_index("Latent_dim")[["r_age", "r_BMI", "r_APOE_bin", "r_risk_ord"]]
sns.heatmap(heatmat.abs(), cmap="viridis", annot=False)
plt.title(f"Latent–Trait |r| heatmap (K={GRAPH_K}, {FOCUS_METRIC})")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "latent_trait_corr_heatmap.png"))
plt.close()

# ================================================================
# PART 4 — CLUSTER COMPOSITION (K-means + Consensus)
# ================================================================
print("\n=== PART 4: cluster composition (kmeans + consensus) ===")

def cluster_trait_composition(df_labels, cluster_col, name_prefix):
    df_labels["MRI_Exam"] = df_labels["MRI_Exam"].astype(str).str.zfill(5)
    df_m = pd.merge(df_sub, df_labels, on="MRI_Exam", how="inner")

    if df_m.empty:
        return pd.DataFrame()

    results = []
    for cl in sorted(df_m[cluster_col].unique()):
        sub = df_m[df_m[cluster_col] == cl]
        row = {
            "cluster": cl,
            "n": len(sub),
            "prefix": name_prefix,
        }
        for fac in ["APOE", "risk_for_ad", "sex"]:
            if fac not in df_m.columns:
                continue
            vc = sub[fac].astype(str).value_counts(normalize=True)
            for cat, p in vc.items():
                row[f"{fac}_{cat}"] = round(float(p), 3)
        results.append(row)

    return pd.DataFrame(results)


# Load cluster assignments
kmeans_csv = os.path.join(RES_DIR, f"kmeans_cluster_assignments_K{GRAPH_K}.csv")
consensus_csv = os.path.join(RES_DIR, f"consensus_clusters_K{GRAPH_K}.csv")

if os.path.exists(kmeans_csv):
    df_kmeans = pd.read_csv(kmeans_csv)
    df_kmeans["MRI_Exam"] = df_kmeans["MRI_Exam"].astype(str).str.zfill(5)
else:
    df_kmeans = pd.DataFrame()
    print("WARNING: kmeans cluster assignment file not found:", kmeans_csv)

if os.path.exists(consensus_csv):
    df_cons = pd.read_csv(consensus_csv)
    df_cons["MRI_Exam"] = df_cons["MRI_Exam"].astype(str).str.zfill(5)
else:
    df_cons = pd.DataFrame()
    print("WARNING: consensus cluster file not found:", consensus_csv)

# ---- K-means composition ----
out_comp = []
if not df_kmeans.empty:
    for metric in METRICS:
        sub = df_kmeans[df_kmeans["Distance"] == metric][["MRI_Exam", "Cluster"]]
        if sub.empty:
            continue
        sub = sub.rename(columns={"Cluster": "ClusterID"})
        comp = cluster_trait_composition(sub, "ClusterID", f"kmeans_{metric}")
        if not comp.empty:
            out_comp.append(comp)

if out_comp:
    df_ck = pd.concat(out_comp, ignore_index=True)
    df_ck.to_csv(
        os.path.join(TABLE_DIR, "cluster_trait_composition_kmeans.csv"),
        index=False,
    )
    print("Saved kmeans cluster trait composition table.")
else:
    print("No kmeans cluster compositions computed.")

# ---- Consensus composition ----
if not df_cons.empty:
    df_cons2 = df_cons.rename(columns={"ConsensusCluster": "ClusterID"})[
        ["MRI_Exam", "ClusterID"]
    ]
    df_cc = cluster_trait_composition(df_cons2, "ClusterID", "consensus")
    if not df_cc.empty:
        df_cc.to_csv(
            os.path.join(TABLE_DIR, "cluster_trait_composition_consensus.csv"),
            index=False,
        )
        print("Saved consensus cluster trait composition table.")

# ================================================================
# PART 5 — APOE × CLUSTER HEATMAPS & CHI-SQUARE
# ================================================================
print("\n=== PART 5: APOE × cluster heatmaps + chi-square ===")

def plot_apoe_cluster_heatmaps(df_kmeans, df_sub):
    if df_kmeans.empty or "APOE" not in df_sub.columns:
        print("Skipping APOE×cluster heatmaps (missing data).")
        return

    fig, axes = plt.subplots(
        nrows=len(METRICS), ncols=1,
        figsize=(4, 8), dpi=300,
        constrained_layout=True,
    )

    chi_rows = []

    for ax, metric in zip(axes, METRICS):
        sub = df_kmeans[df_kmeans["Distance"] == metric][["MRI_Exam", "Cluster"]].copy()
        sub["MRI_Exam"] = sub["MRI_Exam"].astype(str).str.zfill(5)

        df_m = pd.merge(
            df_sub[["MRI_Exam", "APOE"]],
            sub,
            on="MRI_Exam",
            how="inner",
        )

        if df_m.empty:
            ax.set_title(f"{metric}: no data after merge")
            ax.axis("off")
            continue

        ct = pd.crosstab(df_m["Cluster"], df_m["APOE"])

        if ct.empty or ct.shape[0] < 1 or ct.shape[1] < 1:
            ax.set_title(f"{metric}: empty crosstab")
            ax.axis("off")
            continue

        sns.heatmap(
            ct,
            annot=True,
            fmt="d",
            cmap="viridis",
            cbar=True,
            ax=ax,
        )
        ax.set_title(f"{metric} — APOE × Cluster (k-means)")
        ax.set_xlabel("APOE")
        ax.set_ylabel("Cluster")

        # Chi-square test
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, dof, _ = chi2_contingency(ct.values)
            chi_rows.append(
                {
                    "Metric": metric,
                    "chi2": chi2,
                    "p_value": p,
                    "dof": dof,
                }
            )

    out = os.path.join(PLOT_DIR, "kmeans_cluster_heatmap_APOE.png")
    plt.savefig(out)
    plt.close()
    print("Saved", out)

    if chi_rows:
        df_chi = pd.DataFrame(chi_rows)
        df_chi.to_csv(
            os.path.join(TABLE_DIR, "kmeans_APOE_cluster_chisq.csv"),
            index=False,
        )
        print("Saved chi-square stats for APOE × cluster.")

if not df_kmeans.empty:
    plot_apoe_cluster_heatmaps(df_kmeans, df_sub)

# ================================================================
# PART 6 — SILHOUETTE SCORES PER METRIC
# ================================================================
print("\n=== PART 6: silhouette scores per metric ===")

sil_rows = []

if not df_kmeans.empty:
    for metric in METRICS:
        sub = df_kmeans[df_kmeans["Distance"] == metric][["MRI_Exam", "Cluster"]].copy()
        if sub.empty:
            continue

        sub["MRI_Exam"] = sub["MRI_Exam"].astype(str).str.zfill(5)
        df_m = pd.merge(df_sub[["MRI_Exam"]], sub, on="MRI_Exam", how="inner")
        labels = df_m["Cluster"].values

        if len(np.unique(labels)) < 2:
            sil = np.nan
        else:
            Z = Z_dict[metric]
            Zs = StandardScaler().fit_transform(Z)
            # Align order
            # (df_m is already sorted via merge; df_sub used to create Zs order)
            sil = silhouette_score(Zs, labels)

        sil_rows.append({"Metric": metric, "Silhouette": sil})

df_sil = pd.DataFrame(sil_rows)
df_sil.to_csv(
    os.path.join(TABLE_DIR, "kmeans_silhouette_scores.csv"),
    index=False,
)

plt.figure(figsize=(4, 3), dpi=300)
sns.barplot(data=df_sil, x="Metric", y="Silhouette", color="#888888")
plt.title("Silhouette score per distance metric")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "kmeans_silhouette_scores.png"))
plt.close()

print("Saved silhouette scores (csv + png).")

# ================================================================
# PART 7 — LOGISTIC REGRESSION AUC (APOE, sex, risk_for_ad)
# ================================================================
print("\n=== PART 7: logistic regression AUC across metrics ===")

def logreg_auc_cv(Z, y, n_splits=5):
    """Standardized latents + 5-fold CV logistic AUC."""
    y = np.asarray(y)
    valid = ~pd.isna(y)
    Z = Z[valid]
    y = y[valid]
    if len(np.unique(y)) < 2 or len(y) < 10:
        return np.nan

    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probs_all = []
    y_all = []

    for tr, te in skf.split(Zs, y):
        clf = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=200,
        )
        clf.fit(Zs[tr], y[tr])
        prob = clf.predict_proba(Zs[te])[:, 1]
        probs_all.append(prob)
        y_all.append(y[te])

    probs_all = np.concatenate(probs_all)
    y_all = np.concatenate(y_all)
    try:
        auc = roc_auc_score(y_all, probs_all)
    except ValueError:
        auc = np.nan
    return auc


auc_rows = []
for factor in ["APOE", "sex", "risk_for_ad"]:
    # Map to binary if needed
    if factor not in df_sub.columns:
        continue

    col = df_sub[factor].astype(str)

    if factor == "APOE":
        mapping = {"E4-": 0, "E4+": 1}
    elif factor == "sex":
        mapping = {"F": 0, "M": 1}
    else:  # risk_for_ad: treat 0 vs 1+ (any risk)
        mapping = {"0": 0, "1": 1, "2": 1, "3": 1}

    y_bin = col.map(mapping)
    for metric, Z in Z_dict.items():
        auc = logreg_auc_cv(Z, y_bin)
        auc_rows.append(
            {
                "Factor": factor,
                "Metric": metric,
                "AUC": auc,
            }
        )

df_auc = pd.DataFrame(auc_rows)
df_auc.to_csv(os.path.join(TABLE_DIR, "logreg_auc_by_metric.csv"), index=False)

# Simple line plot: AUC vs distance metric for each factor
plt.figure(figsize=(6, 3), dpi=300)
for factor in df_auc["Factor"].unique():
    sub = df_auc[df_auc["Factor"] == factor]
    plt.plot(sub["Metric"], sub["AUC"], marker="o", label=factor)

plt.ylim(0.0, 1.0)
plt.ylabel("AUC (5-fold CV)")
plt.xlabel("Distance Metric")
plt.title("Logistic AUC across CORR / EUCLID / WASS")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "logreg_auc_comparison.png"))
plt.close()

print("Saved logistic AUC comparison (csv + png).")

# ================================================================
print("\n======================")
print(" DONE — Part 6 (patched with Palette A & legends)")
print("Outputs:", EVAL_ROOT)
print("======================")
