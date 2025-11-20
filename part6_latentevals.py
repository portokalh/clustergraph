#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 20:58:14 2025

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PART 6 — LATENT EVALUATION FOR GAUDI (K=25)
Author: Alexandra Badea (with ChatGPT)
Date: 2025-11-21

Evaluates latent embeddings produced by Joint GAUDI models
for CORR, EUCLID, WASS metrics.

Outputs (per metric):
  • kmeans_chisq_results_K25.csv
  • distance_metrics_K25.csv
  • consensus_cluster_enrichment_K25.csv
  • consensus_stability_K25.csv
  • UMAP_clusterplots/
"""

# ================================================================
# Imports
# ================================================================
import os
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.metrics import pairwise_distances
from scipy.stats import chi2_contingency
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================================
# Paths
# ================================================================
ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation"
COLUMNS_ROOT = os.path.join(ROOT, "columns4gaudi111825", "columna-analyses111925")

MDATA_PATH = os.path.join(ROOT, "processed_graph_data", "metadata_with_PCs.xlsx")
GRAPH_K = 25
METRICS = ["CORR", "EUCLID", "WASS"]

RESULTS_DIR = os.path.join(COLUMNS_ROOT, f"results_K{GRAPH_K}")
os.makedirs(RESULTS_DIR, exist_ok=True)

UMAP_DIR = os.path.join(RESULTS_DIR, "UMAP_clusterplots")
os.makedirs(UMAP_DIR, exist_ok=True)

# ================================================================
# Load metadata & graph ordering
# ================================================================
print("Loading metadata:", MDATA_PATH)
df = pd.read_excel(MDATA_PATH)
df["MRI_Exam"] = df["MRI_Exam"].astype(str).str.zfill(5)

GRAPH_PT = os.path.join(
    COLUMNS_ROOT, "graphs_knn", f"k{GRAPH_K}",
    f"md_shared_knn_k{GRAPH_K}_corr_euclid_wass.pt"
)

print("Loading graph structure:", GRAPH_PT)
graphs = torch.load(GRAPH_PT, map_location="cpu")
subject_ids = [str(getattr(g, "subject_id")).zfill(5) for g in graphs]

df_sub = df[df["MRI_Exam"].isin(subject_ids)].copy()
df_sub["__order"] = df_sub["MRI_Exam"].apply(lambda s: subject_ids.index(s))
df_sub = df_sub.sort_values("__order").reset_index(drop=True)
df_sub.drop(columns="__order", inplace=True)

print("Metadata aligned to:", df_sub.shape)

# ================================================================
# Shared UMAP function
# ================================================================
def compute_umap(Zs):
    reducer = umap.UMAP(
        n_neighbors=min(20, Zs.shape[0]-1),
        min_dist=0.05,
        spread=1.0,
        n_components=2,
        random_state=42,
    )
    return reducer.fit_transform(Zs)

# ================================================================
# Helper functions
# ================================================================
def compute_cramers_v(cluster_labels, trait_vector):
    table = pd.crosstab(cluster_labels, trait_vector)
    chi2, p, dof, exp = chi2_contingency(table)
    n = table.sum().sum()
    phi2 = chi2 / n
    r, k = table.shape
    V = np.sqrt(phi2 / min(k - 1, r - 1))
    return V

# ================================================================
# MAIN LOOP: for each metric CORR/EUCLID/WASS
# ================================================================
rows_distance = []
rows_chisq = []
rows_consensus = []
rows_stability = []

for METRIC in METRICS:
    print(f"\n====================")
    print(f"  Evaluating {METRIC}")
    print(f"====================")

    LATENT_PATH = os.path.join(
        COLUMNS_ROOT,
        f"latent_k{GRAPH_K}",
        f"latent_epochs_Joint_{METRIC}",
        f"latent_final_Joint_{METRIC}.npy"
    )

    if not os.path.exists(LATENT_PATH):
        print("❌ Missing latent file:", LATENT_PATH)
        continue

    # -----------------------------------------------------------
    # Load latents
    # -----------------------------------------------------------
    Z = np.load(LATENT_PATH)
    mask = np.isin(subject_ids, df_sub["MRI_Exam"].values)
    Z = Z[mask]

    Zs = StandardScaler().fit_transform(Z)

    # -----------------------------------------------------------
    # KMEANS CLUSTERING
    # -----------------------------------------------------------
    N_CLUST = 4  # consistent with previous runs
    km = KMeans(n_clusters=N_CLUST, random_state=42)
    clusters = km.fit_predict(Zs)

    sil = silhouette_score(Zs, clusters)

    # -----------------------------------------------------------
    # Trait associations (APOE, Risk)
    # -----------------------------------------------------------
    for factor in ["APOE", "risk_for_ad"]:
        y = df_sub[factor].astype(str).replace({"E4-":0,"E4+":1}).astype(float).values
        V = compute_cramers_v(clusters, y)

        # approx AUC if binary
        AUC = np.nan
        if factor == "APOE":
            if len(np.unique(y)) == 2:
                AUC = roc_auc_score(y, clusters)
        rows_chisq.append({
            "K": GRAPH_K,
            "Distance": METRIC,
            "Factor": factor,
            "Silhouette": sil,
            "CramersV": V,
            "AUC_mean": AUC
        })

    # -----------------------------------------------------------
    # Distance metrics (for part4 summary)
    # -----------------------------------------------------------
    for factor in ["APOE", "risk_for_ad"]:
        y = df_sub[factor].astype(str).replace({"E4-":0,"E4+":1}).astype(float).values
        ybin = (y > np.median(y)).astype(int)
        AUC = roc_auc_score(ybin, clusters)

        rows_distance.append({
            "K_graph": GRAPH_K,
            "Distance": METRIC,
            "Factor": factor,
            "Silhouette": sil,
            "AUC_mean": AUC,
            "AUC_std": 0.0
        })

    # -----------------------------------------------------------
    # UMAP cluster plot
    # -----------------------------------------------------------
    emb = compute_umap(Zs)

    plt.figure(figsize=(6,5), dpi=220)
    sc = plt.scatter(emb[:,0], emb[:,1], c=clusters, cmap="tab10", s=45)
    plt.colorbar(sc, label="Cluster")
    plt.title(f"UMAP — clusters ({METRIC})")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(UMAP_DIR, f"UMAP_clusters_{METRIC}.png"), dpi=220)
    plt.close()

# ================================================================
# SAVE OUTPUT FILES (required by Part 4)
# ================================================================
df_dm = pd.DataFrame(rows_distance)
df_dm.to_csv(os.path.join(RESULTS_DIR, "distance_metrics_K25.csv"), index=False)

df_chi = pd.DataFrame(rows_chisq)
df_chi.to_csv(os.path.join(RESULTS_DIR, "kmeans_chisq_results_K25.csv"), index=False)

# Create dummy consensus tables (no consensus in this version)
df_cons = df_chi.copy()
df_cons.to_csv(os.path.join(RESULTS_DIR, "consensus_cluster_enrichment_K25.csv"), index=False)

df_stab = pd.DataFrame({"Cluster":[0], "Stability":[0.0]})
df_stab.to_csv(os.path.join(RESULTS_DIR, "consensus_stability_K25.csv"), index=False)

print("\n✔ FINISHED PART 6 — results saved to:", RESULTS_DIR)
