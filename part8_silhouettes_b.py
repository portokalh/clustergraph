#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 13:03:32 2025

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PART 8b — K-selection for GAUDI latent space (Joint model, K=25)
================================================================

For K = 2..10, this script:

  • Runs KMeans on GAUDI Joint EUCLID latents
  • Computes cluster quality indices:
        - Within-cluster sum of squares (WSS / inertia)
        - Silhouette score
        - Calinski–Harabasz (CH) index
        - Davies–Bouldin (DB) index
        - GAP statistic (with Monte Carlo null)
        - Cluster stability (mean ARI from bootstrap)
  • Saves:
        - k_selection_indices.csv
        - K_selection_panel.png  (6-panel figure)
        - KMeans labels per K   (optional, as CSV)

Uses metadata file:
  metadata_with_PCs112125.xlsx
for alignment sanity checks only (no traits used here).

You can run it from the columna-analyses111925 folder with Spyder or python.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
K_MIN = 2
K_MAX = 10
K_RANGE = range(K_MIN, K_MAX + 1)

METRIC   = "EUCLID"
LATENT_K = 25

N_GAP_BOOT   = 100   # Monte Carlo reps for GAP statistic
N_STAB_BOOT  = 100   # Bootstrap reps for ARI stability

ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation"
CROOT = os.path.join(ROOT, "columns4gaudi111825", "columna-analyses111925")

# GAUDI Joint EUCLID latents
LATENT_FILE = os.path.join(
    CROOT,
    f"latent_k{LATENT_K}",
    f"latent_epochs_Joint_{METRIC}",
    f"latent_final_Joint_{METRIC}.npy"
)

# Updated metadata file (for alignment sanity)
META_FILE = os.path.join(
    ROOT,
    "columns4gaudi111825",
    "utilities",
    "metadata_with_PCs112125.xlsx"
)

OUTDIR = os.path.join(CROOT, f"k_selection_Joint_k{LATENT_K}")
os.makedirs(OUTDIR, exist_ok=True)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
print("\n🔵 Loading GAUDI latents from:", LATENT_FILE)
Z = np.load(LATENT_FILE)      # shape = (N_subj, latent_dim)
n_subj, latent_dim = Z.shape
print(f"  Latents shape: {Z.shape}")

print("🔵 Loading metadata from:", META_FILE)
meta = pd.read_excel(META_FILE)
print(f"  Metadata shape (raw): {meta.shape}")

if len(meta) > n_subj:
    print(f"  ⚠ Metadata has {len(meta)} rows but latents have {n_subj}; truncating.")
    meta = meta.iloc[:n_subj].copy()
elif len(meta) < n_subj:
    raise ValueError(
        f"Metadata has fewer rows ({len(meta)}) than latents ({n_subj}). "
        "Please check subject alignment."
    )

meta.index = np.arange(n_subj)
print("  ✔ Metadata aligned to latents (rows = subjects)\n")


# ------------------------------------------------------------
# UTILS
# ------------------------------------------------------------
def compute_wss(kmeans_obj):
    """
    Within-cluster sum of squares (distortion).
    For KMeans with Euclidean metric, this is just inertia_.
    """
    return float(kmeans_obj.inertia_)


def kmeans_fit(Z, K, random_state=42, n_init=50):
    """Helper to fit KMeans and return labels + object."""
    km = KMeans(
        n_clusters=K,
        random_state=random_state,
        n_init=n_init,
    )
    labels = km.fit_predict(Z)
    return km, labels


def gap_statistic(Z, K, B=100, random_state=42):
    """
    Compute GAP statistic for a single K.

    GAP(K) = E[log(W*_b)] - log(W)
      where W*_b is WSS for reference (null) distribution.
    """
    rng = np.random.default_rng(random_state)

    # Fit on real data
    km_real, labels_real = kmeans_fit(Z, K, random_state=random_state)
    W_real = compute_wss(km_real)

    # Uniform reference in bounding box of Z
    mins = Z.min(axis=0)
    maxs = Z.max(axis=0)

    log_W_null = []

    for b in range(B):
        Z_ref = rng.uniform(mins, maxs, size=Z.shape)
        km_ref, labels_ref = kmeans_fit(Z_ref, K, random_state=random_state + b)
        W_ref = compute_wss(km_ref)
        log_W_null.append(np.log(W_ref))

    log_W_null = np.array(log_W_null)
    gap = log_W_null.mean() - np.log(W_real)
    sdk = np.sqrt(1 + 1.0 / B) * log_W_null.std(ddof=1)  # Tibshirani et al.

    return gap, sdk, W_real, labels_real, km_real


def stability_ari(Z, labels_base, K, B=100, random_state=42):
    """
    Compute ARI-based clustering stability.

    - Fit base KMeans once to obtain labels_base (already done outside).
    - For each bootstrap:
        * sample subjects with replacement
        * fit KMeans on bootstrap Z
        * compare labels_boot vs labels_base on SAME bootstrapped indices
          using adjusted_rand_score.
    """
    rng = np.random.default_rng(random_state)
    n = Z.shape[0]
    ari_list = []

    for b in range(B):
        idx = rng.integers(0, n, size=n)  # bootstrap indices
        Z_boot = Z[idx]

        km_boot, labels_boot = kmeans_fit(Z_boot, K, random_state=random_state + b)

        # Compare clustering of bootstrap subjects under base vs bootstrap:
        # base labels restricted to sampled indices vs bootstrap labels
        ari = adjusted_rand_score(labels_base[idx], labels_boot)
        ari_list.append(ari)

    ari_arr = np.array(ari_list)
    return ari_arr.mean(), ari_arr.std(ddof=1)


# ------------------------------------------------------------
# MAIN LOOP: COMPUTE INDICES FOR EACH K
# ------------------------------------------------------------
records = []

print("🚀 Running K-selection over K =", list(K_RANGE), "\n")

for K in K_RANGE:
    print("===========================================")
    print(f"  Evaluating K = {K}")
    print("===========================================")

    # ---- GAP statistic (includes real WSS, base labels, km_real) ----
    print("  🔹 Computing GAP statistic and WSS …")
    gap_K, gap_se_K, W_real_K, labels_base_K, km_real_K = gap_statistic(
        Z, K, B=N_GAP_BOOT, random_state=42
    )
    print(f"     GAP(K={K}) = {gap_K:.4f} ± {gap_se_K:.4f}")
    print(f"     WSS(K={K}) = {W_real_K:.4f}")

    # Save labels for this K
    labels_df = pd.DataFrame({
        "subject_index": np.arange(n_subj),
        f"kmeans_K{K}_label": labels_base_K
    })
    labels_df.to_csv(
        os.path.join(OUTDIR, f"kmeans_labels_K{K}.csv"),
        index=False
    )

    # ---- Silhouette, CH, DB ----
    print("  🔹 Computing Silhouette / CH / DB …")
    try:
        sil = silhouette_score(Z, labels_base_K, metric="euclidean")
    except Exception:
        sil = np.nan

    try:
        ch = calinski_harabasz_score(Z, labels_base_K)
    except Exception:
        ch = np.nan

    try:
        db = davies_bouldin_score(Z, labels_base_K)
    except Exception:
        db = np.nan

    print(f"     Silhouette(K={K}) = {sil:.4f}")
    print(f"     CH index(K={K})   = {ch:.4f}")
    print(f"     DB index(K={K})   = {db:.4f}")

    # ---- ARI stability ----
    print("  🔹 Computing ARI stability (bootstrap) …")
    ari_mean, ari_std = stability_ari(
        Z, labels_base_K, K, B=N_STAB_BOOT, random_state=123
    )
    print(f"     ARI mean(K={K}) = {ari_mean:.4f}  (std = {ari_std:.4f})")

    # ---- record all ----
    records.append({
        "K": K,
        "WSS": W_real_K,
        "GAP": gap_K,
        "GAP_SE": gap_se_K,
        "Silhouette": sil,
        "Calinski_Harabasz": ch,
        "Davies_Bouldin": db,
        "ARI_mean": ari_mean,
        "ARI_std": ari_std,
    })

# ------------------------------------------------------------
# SAVE TABLE
# ------------------------------------------------------------
indices_df = pd.DataFrame(records)
indices_df.to_csv(
    os.path.join(OUTDIR, "k_selection_indices.csv"),
    index=False
)

print("\n✔ Saved K-selection indices →", os.path.join(OUTDIR, "k_selection_indices.csv"))
print(indices_df)


# ------------------------------------------------------------
# PLOTTING PANEL
# ------------------------------------------------------------
print("\n🖼  Plotting K-selection panel …")

sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

# 1. WSS (Elbow)
axes[0].plot(indices_df["K"], indices_df["WSS"], marker="o")
axes[0].set_title("Elbow: Within-Cluster Sum of Squares")
axes[0].set_xlabel("K")
axes[0].set_ylabel("WSS")

# 2. Silhouette
axes[1].plot(indices_df["K"], indices_df["Silhouette"], marker="o")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Score")

# 3. Calinski–Harabasz
axes[2].plot(indices_df["K"], indices_df["Calinski_Harabasz"], marker="o")
axes[2].set_title("Calinski–Harabasz Index")
axes[2].set_xlabel("K")
axes[2].set_ylabel("Score")

# 4. Davies–Bouldin
axes[3].plot(indices_df["K"], indices_df["Davies_Bouldin"], marker="o")
axes[3].set_title("Davies–Bouldin Index (lower = better)")
axes[3].set_xlabel("K")
axes[3].set_ylabel("Score")

# 5. GAP statistic
axes[4].errorbar(
    indices_df["K"],
    indices_df["GAP"],
    yerr=indices_df["GAP_SE"],
    fmt="-o",
    capsize=4
)
axes[4].set_title("GAP Statistic")
axes[4].set_xlabel("K")
axes[4].set_ylabel("GAP (higher = better)")

# 6. ARI stability
axes[5].errorbar(
    indices_df["K"],
    indices_df["ARI_mean"],
    yerr=indices_df["ARI_std"],
    fmt="-o",
    capsize=4
)
axes[5].set_title("Cluster Stability (ARI)")
axes[5].set_xlabel("K")
axes[5].set_ylabel("ARI (higher = more stable)")

plt.tight_layout()
panel_path = os.path.join(OUTDIR, "K_selection_panel.png")
plt.savefig(panel_path, dpi=300)
plt.close()

print("✔ Saved K-selection panel →", panel_path)
print("\n🎉 DONE. Review k_selection_indices.csv and K_selection_panel.png to choose K.")
