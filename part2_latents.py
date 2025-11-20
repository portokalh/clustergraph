#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 17:06:14 2025

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAUDI Latent Analysis Pipeline
==============================

Loads latent_final_*.npy files and performs:
    • Z-scoring
    • PCA (2D)
    • UMAP (2D)
    • KMeans clustering
    • HDBSCAN clustering
    • Silhouette and DB index scoring
    • Pairwise similarity matrices
    • Consensus clustering across K = 10,20,30,50
    • Saves plots + CSVs

Author: Alexandra Badea + ChatGPT
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import hdbscan

# -------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------
LATENT_ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation/columns4gaudi111825/columna-analyses111925"
K_LIST      = [10, 20, 30, 50]
METRICS     = ["CORR", "EUCLID", "WASS"]
MODES       = ["Joint", "MD", "QSM"]

OUT_DIR     = os.path.join(LATENT_ROOT, "latent_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------------
def load_latents():
    results = {}

    for K in K_LIST:
        for metric in METRICS:
            for mode in MODES:
                tag = f"{mode}_{metric}"
                fname = f"{LATENT_ROOT}/latent_k{K}/latent_epochs_{tag}/latent_final_{tag}.npy"

                if os.path.exists(fname):
                    Z = np.load(fname)
                    results[(K, metric, mode)] = Z
                    print(f"Loaded: {fname} → {Z.shape}")
                else:
                    print(f"WARNING: missing {fname}")

    return results

# -------------------------------------------------------------------
def zscore(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-8)

# -------------------------------------------------------------------
def run_pca_umap(Z, outname):
    """Produces PCA + UMAP and saves figure."""
    Zs = zscore(Z)

    # PCA
    pca = PCA(n_components=2).fit_transform(Zs)

    # UMAP
    reducer = umap.UMAP(
        n_neighbors=10,
        min_dist=0.1,
        metric="euclidean",
        random_state=0
    )
    U = reducer.fit_transform(Zs)

    # Plot
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.scatter(pca[:,0], pca[:,1], s=30)
    plt.title("PCA")

    plt.subplot(1,2,2)
    plt.scatter(U[:,0], U[:,1], s=30)
    plt.title("UMAP")

    plt.tight_layout()
    plt.savefig(outname, dpi=200)
    plt.close()

    return pca, U

# -------------------------------------------------------------------
def cluster_and_score(Z, K=4):
    """Builds clusters and returns labels + scores."""
    Zs = zscore(Z)

    # KMeans clustering
    km = KMeans(n_clusters=K, n_init=20, random_state=0).fit(Zs)
    labels_km = km.labels_

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=8,
        min_samples=3
    ).fit(Zs)
    labels_hdb = clusterer.labels_

    # metrics
    sil = silhouette_score(Zs, labels_km)
    db = davies_bouldin_score(Zs, labels_km)

    return labels_km, labels_hdb, sil, db

# -------------------------------------------------------------------
def save_similarity(Z, outname):
    """Saves NxN distance matrix + dendrogram."""
    D = squareform(pdist(Z, metric="euclidean"))
    np.save(outname + "_dist.npy", D)

    # dendrogram
    Zlink = linkage(D, method="ward")

    plt.figure(figsize=(12, 4))
    dendrogram(Zlink)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.savefig(outname + "_dendrogram.png", dpi=200)
    plt.close()

    return D

# -------------------------------------------------------------------
def consensus_clustering(all_latents):
    """
    Combines clusters across K = 10,20,30,50 for Joint_CORR (default)
    Produces consensus matrix and hierarchical clustering.
    """

    keys = [(K, "CORR", "Joint") for K in K_LIST]

    mats = []
    for key in keys:
        Z = zscore(all_latents[key])
        labels, _, _, _ = cluster_and_score(Z)
        n = len(labels)
        M = np.zeros((n,n))

        for i in range(n):
            for j in range(n):
                M[i,j] = 1 if labels[i] == labels[j] else 0
        mats.append(M)

    consensus = sum(mats) / len(mats)
    np.save(os.path.join(OUT_DIR, "consensus_joint_corr.npy"), consensus)

    # dendrogram
    D = 1 - consensus
    Zlink = linkage(squareform(D), method="average")

    plt.figure(figsize=(12,4))
    dendrogram(Zlink)
    plt.title("Consensus Clustering (Joint_CORR)")
    plt.savefig(os.path.join(OUT_DIR, "consensus_dendrogram.png"), dpi=200)
    plt.close()

# -------------------------------------------------------------------
def main():
    all_latents = load_latents()

    summary_rows = []

    for (K, metric, mode), Z in all_latents.items():

        tag = f"K{K}_{mode}_{metric}"
        print(f"\n=== Analyzing {tag} ===")

        out_prefix = os.path.join(OUT_DIR, tag)

        # PCA + UMAP
        pca, um = run_pca_umap(Z, out_prefix + "_pca_umap.png")

        # Clustering
        labels_km, labels_hdb, sil, db = cluster_and_score(Z)

        # Save metrics
        summary_rows.append({
            "K": K,
            "Metric": metric,
            "Mode": mode,
            "Silhouette": sil,
            "DaviesBouldin": db,
            "NumClusters_KMeans": len(set(labels_km)),
            "NumClusters_HDBSCAN": len(set(labels_hdb))
        })

        # Similarity matrix
        save_similarity(zscore(Z), out_prefix)

    # Save summary table
    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(OUT_DIR, "latent_summary_table.csv"), index=False)
    print(f"\nSaved summary → {OUT_DIR}/latent_summary_table.csv")

    # Build consensus for Joint_CORR across K
    consensus_clustering(all_latents)
    print("\nConsensus clustering complete.")

# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
