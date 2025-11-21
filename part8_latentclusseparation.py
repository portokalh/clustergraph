#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PART 8 — KMeans clustering in GAUDI latent space (Joint model, K=25)
====================================================================

For K = 3 and 4:

  • Run KMeans on GAUDI Joint EUCLID latents
  • Compute cluster sizes and pairwise centroid distances
  • Compute cluster-wise enrichment for:
        - APOE (E4+ vs E4-)
        - sex  (M vs F)
        - risk_for_ad (2–3 vs 0–1)
  • Use LAPLACE-SMOOTHED prevalence in each cluster:
        p_c = (y_c + 1) / (n_c + 2)
    to build subject-level scores and to plot bar heights
  • Compute ROC-AUC of "trait predicted from cluster membership"
    using smoothed prevalence as a score
  • Also compute bootstrap AUC mean and 95% CI
  • Save barplots ("kmeans_cluster_bars") for each K

You can run this both on unadjusted and covariate-adjusted latents by
changing LATENT_FILE.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
K_LIST   = [3, 4]      # K-means K values
METRIC   = "EUCLID"
LATENT_K = 25

ROOT  = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation"
CROOT = os.path.join(ROOT, "columns4gaudi111825", "columna-analyses111925")

# Latents: use Joint EUCLID K=25 (change if you want adjusted latents)
LATENT_FILE = os.path.join(
    CROOT,
    f"latent_k{LATENT_K}",
    f"latent_epochs_Joint_{METRIC}",
    f"latent_final_Joint_{METRIC}.npy"
)

# Metadata with Age / Sex / BMI / APOE / risk_for_ad
META_FILE = os.path.join(
    ROOT,
    "columns4gaudi111825",
    "utilities",
    "metadata_with_PCs112125.xlsx"
)

OUTDIR = os.path.join(CROOT, f"kmeans_latent_clusters_Joint_k{LATENT_K}")
os.makedirs(OUTDIR, exist_ok=True)


# ------------------------------------------------------------
# Helper to auto-pick metadata columns
# ------------------------------------------------------------
def pick_column(df, candidates, name_for_error):
    """Pick the first existing column from `candidates`."""
    for c in candidates:
        if c in df.columns:
            return df[c]
    raise ValueError(
        f"Could not find a column for {name_for_error}. "
        f"Tried: {candidates}. Available columns: {list(df.columns)}"
    )


# ------------------------------------------------------------
# Bootstrap AUC helper
# ------------------------------------------------------------
def bootstrap_auc(y, scores, n_boot=2000, random_state=0):
    """
    Parametric bootstrap for ROC AUC with replacement.

    Returns:
        mean_auc, ci_low, ci_high  (95% CI)
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    scores = np.asarray(scores)

    aucs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b = y[idx]
        s_b = scores[idx]
        # Skip samples with only one class
        if np.unique(y_b).size < 2:
            continue
        aucs.append(roc_auc_score(y_b, s_b))

    if len(aucs) == 0:
        return np.nan, np.nan, np.nan

    aucs = np.array(aucs)
    mean_auc = aucs.mean()
    ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
    return mean_auc, ci_low, ci_high


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
print("\n🔵 Loading GAUDI latents from:", LATENT_FILE)
Z = np.load(LATENT_FILE)
n_subj, latent_dim = Z.shape
print(f"  Latents shape: {Z.shape}")

print("🔵 Loading metadata from:", META_FILE)
meta = pd.read_excel(META_FILE)
print(f"  Metadata shape (raw): {meta.shape}")

# Align metadata rows to latents
if len(meta) > n_subj:
    print(f"  ⚠ Metadata has {len(meta)} rows but latents have {n_subj}; truncating.")
    meta = meta.iloc[:n_subj].copy()
elif len(meta) < n_subj:
    raise ValueError("Metadata has fewer rows than latents!")

meta.index = np.arange(n_subj)


# ------------------------------------------------------------
# Extract columns + print unique values
# ------------------------------------------------------------
APOE_series = pick_column(meta, ["APOE", "APOE4_in_name", "APOE_group"], "APOE")
SEX_series  = pick_column(meta, ["sex", "Sex", "SEX"], "sex")
RISK_series = pick_column(meta, ["risk_for_ad"], "risk_for_ad")
AGE_series  = pick_column(meta, ["age", "Age", "AGE"], "Age")
BMI_series  = pick_column(meta, ["BMI", "bmi"], "BMI")

print("\n🔍 UNIQUE VALUES IN METADATA")
print("  APOE:", APOE_series.unique())
print("  sex:", SEX_series.unique())
print("  risk_for_ad:", RISK_series.unique())
print("  age:", AGE_series.unique()[:10], "...")
print("  BMI:", BMI_series.unique()[:10], "...")


# ------------------------------------------------------------
# Recode to binary
# ------------------------------------------------------------
def to_binary_apoe(x):
    """
    Convert APOE label to binary:
       E4+ → 1
       E4– → 0
    Handles strings like 'E4-', 'E4+' (case-insensitive).
    """
    if isinstance(x, str):
        x = x.strip().upper()
        if x in ["E4+", "4+", "+", "POS", "POSITIVE"]:
            return 1
        if x in ["E4-", "4-", "-", "NEG", "NEGATIVE"]:
            return 0
        # Fallback: presence / absence of '+'
        return 1 if "+" in x else 0
    return int(x)


def to_binary_sex(x):
    """Male = 1, Female = 0."""
    if isinstance(x, str):
        x = x.lower()
        if x.startswith("m"):
            return 1
        if x.startswith("f"):
            return 0
    try:
        return int(x)
    except Exception:
        return 0


def to_binary_risk_group(x):
    """
    Recode risk_for_ad:
        0–1 → 0 (low risk)
        2–3 → 1 (high risk)
    """
    try:
        val = int(x)
        return 1 if val >= 2 else 0
    except Exception:
        return 0


APOE_bin = APOE_series.apply(to_binary_apoe).values.astype(int)
SEX_bin  = SEX_series.apply(to_binary_sex).values.astype(int)
RISK_bin = RISK_series.apply(to_binary_risk_group).values.astype(int)

traits_bin = {
    "APOE": APOE_bin,
    "sex":  SEX_bin,
    "risk_for_ad": RISK_bin,
}

print("\n🔢 BINARY COUNTS")
for t, arr in traits_bin.items():
    print(f"  {t}: {np.bincount(arr)}")


# ------------------------------------------------------------
# AUC summary container
# ------------------------------------------------------------
auc_records = []   # each row: K, trait, AUC, boot_mean, boot_ci_low, boot_ci_high


# ------------------------------------------------------------
# MAIN LOOP OVER K
# ------------------------------------------------------------
for K in K_LIST:
    print("\n===========================================")
    print(f"   🔵 KMeans clustering in latent space: K = {K}")
    print("===========================================\n")

    # ----------------- KMeans -----------------
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=50)
    cluster_labels = kmeans.fit_predict(Z)
    centers = kmeans.cluster_centers_

    # ---------- cluster sizes ----------
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    cluster_sizes.index = [f"Cluster_{i}" for i in range(K)]
    cluster_sizes.name = "n_subjects"
    cluster_sizes.to_csv(os.path.join(OUTDIR, f"kmeans_K{K}_cluster_sizes.csv"))
    print("  ✔ Cluster sizes saved")

    # ---------- centroid distances ----------
    D = cdist(centers, centers, metric="euclidean")
    pd.DataFrame(
        D,
        index=[f"Cluster_{i}" for i in range(K)],
        columns=[f"Cluster_{j}" for j in range(K)]
    ).to_csv(os.path.join(OUTDIR, f"kmeans_K{K}_centroid_distances.csv"))
    print("  ✔ Centroid distances saved")

    # ---------- enrichment ----------
    enrichment_records = []
    alpha = 1.0   # Laplace smoothing hyper-parameter

    for trait, y_bin in traits_bin.items():
        tmp = pd.DataFrame({"cluster": cluster_labels, "y": y_bin})

        # Raw counts per cluster
        stats = tmp.groupby("cluster")["y"].agg(["sum", "count"])
        stats.rename(columns={"count": "n"}, inplace=True)

        # Laplace-smoothed prevalence
        stats["prevalence"] = (stats["sum"] + alpha) / (stats["n"] + 2 * alpha)

        stats["trait"]   = trait
        stats["Cluster"] = ["Cluster_" + str(c) for c in stats.index]

        enrichment_records.append(stats.reset_index(drop=True))

        # Subject-level scores = smoothed prevalence of their cluster
        smoothed_prev = stats["prevalence"].values
        # Map: cluster index (0..K-1) → smoothed_prev
        scores = smoothed_prev[cluster_labels]

        # AUC and bootstrap CI
        try:
            auc = roc_auc_score(y_bin, scores)
        except Exception:
            auc = np.nan

        boot_mean, ci_low, ci_high = bootstrap_auc(y_bin, scores, n_boot=2000, random_state=42)

        auc_records.append({
            "K": K,
            "trait": trait,
            "AUC_cluster_score": auc,
            "AUC_bootstrap_mean": boot_mean,
            "AUC_bootstrap_ci_low": ci_low,
            "AUC_bootstrap_ci_high": ci_high,
        })

    enrich_df = pd.concat(enrichment_records, ignore_index=True)
    enrich_df.to_csv(os.path.join(OUTDIR, f"kmeans_K{K}_cluster_enrichment.csv"),
                     index=False)
    print("  ✔ Cluster enrichment (Laplace-smoothed) saved")

    # ---------- barplot ----------
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    order  = ["APOE", "sex", "risk_for_ad"]
    labels = ["APOE (E4+)", "Sex (Male)", "High AD Risk (2–3)"]

    for ax, trait, label in zip(axes, order, labels):
        df_t = enrich_df[enrich_df["trait"] == trait].copy()
        df_t = df_t.sort_values("Cluster")

        ax.bar(range(K), df_t["prevalence"])
        ax.set_xticks(range(K))
        ax.set_xticklabels([f"C{i}" for i in range(K)])
        ax.set_ylim(0, 1.0)
        ax.set_title(label)

        # annotate counts
        for i, (y_val, n_val) in enumerate(zip(df_t["prevalence"], df_t["n"])):
            ax.text(i, y_val + 0.02, f"n={n_val}", ha="center", fontsize=8)

    axes[-1].set_xlabel("Cluster")
    fig.suptitle(f"KMeans latent clusters (Joint GAUDI K={LATENT_K}), K={K}",
                 fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    outfig = os.path.join(OUTDIR, f"kmeans_K{K}_cluster_bars.png")
    plt.savefig(outfig, dpi=300)
    plt.close()
    print("  ✔ Barplot saved:", outfig)


# ------------------------------------------------------------
# SAVE AUC TABLE
# ------------------------------------------------------------
auc_df = pd.DataFrame(auc_records)
auc_df["AUC_cluster_score"]    = auc_df["AUC_cluster_score"].round(4)
auc_df["AUC_bootstrap_mean"]   = auc_df["AUC_bootstrap_mean"].round(4)
auc_df["AUC_bootstrap_ci_low"] = auc_df["AUC_bootstrap_ci_low"].round(4)
auc_df["AUC_bootstrap_ci_high"] = auc_df["AUC_bootstrap_ci_high"].round(4)

auc_path = os.path.join(OUTDIR, "kmeans_cluster_trait_AUCs.csv")
auc_df.to_csv(auc_path, index=False)

print("\n🎉 FINISHED PART 8 — KMeans latent clustering with Laplace smoothing.")
print("   AUC summary      →", auc_path)
print("   All figures/CSVs →", OUTDIR)
