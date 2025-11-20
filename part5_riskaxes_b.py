#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PART 5 — TRAIT-ALIGNED LATENT AXES ("risk axes")
Author: Alexandra Badea (with ChatGPT)
Date: 2025-11-20

Produces Age, BMI, APOE, and Risk axes from GAUDI latents, including:
    • UMAP colored by raw traits
    • UMAP colored by axis scores + axis arrow
    • Side-by-side raw vs axis UMAP panels
    • Regression plots (axis score → raw trait)
    • Barplots of latent-dimension loadings
"""

# ================================================================
# IMPORTS
# ================================================================
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# ================================================================
# PATHS (EDIT K + METRIC)
# ================================================================
ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation"
CROOT = os.path.join(ROOT, "columns4gaudi111825", "columna-analyses111925")
MDATA_PATH = os.path.join(ROOT, "processed_graph_data", "metadata_with_PCs.xlsx")

GRAPH_K = 25
METRIC  = "EUCLID"   # CORR, EUCLID, WASS

LATENT_PATH = os.path.join(
    CROOT,
    f"latent_k{GRAPH_K}",
    f"latent_epochs_Joint_{METRIC}",
    f"latent_final_Joint_{METRIC}.npy"
)

OUTDIR = os.path.join(CROOT, f"trait_axes_K{GRAPH_K}_{METRIC}_Joint")
os.makedirs(OUTDIR, exist_ok=True)

# Subfolders
AXIS_DIR = os.path.join(OUTDIR, "axes")
SCORE_DIR = os.path.join(OUTDIR, "axis_scores")
PLOTS = os.path.join(OUTDIR, "plots")
UMAP_AXIS = os.path.join(PLOTS, "umap_axis_only")
UMAP_PAIR = os.path.join(PLOTS, "umap_pairs")
SCATTER = os.path.join(PLOTS, "scatter")
LOADINGS = os.path.join(PLOTS, "loadings")

for d in [AXIS_DIR, SCORE_DIR, PLOTS, UMAP_AXIS, UMAP_PAIR, SCATTER, LOADINGS]:
    os.makedirs(d, exist_ok=True)


# ================================================================
# LOAD METADATA
# ================================================================
df = pd.read_excel(MDATA_PATH)
df["MRI_Exam"] = df["MRI_Exam"].astype(str).str.zfill(5)

# ================================================================
# LOAD GRAPH ORDER (so latents align to subjects)
# ================================================================
GRAPH_PT = os.path.join(
    CROOT, "graphs_knn", f"k{GRAPH_K}",
    f"md_shared_knn_k{GRAPH_K}_corr_euclid_wass.pt"
)

graphs = torch.load(GRAPH_PT, map_location="cpu")
subject_ids = [str(getattr(g, "subject_id")).zfill(5) for g in graphs]

df_sub = df[df["MRI_Exam"].isin(subject_ids)].copy()
df_sub["__order"] = df_sub["MRI_Exam"].apply(lambda x: subject_ids.index(x))
df_sub = df_sub.sort_values("__order").reset_index(drop=True)
df_sub.drop(columns="__order", inplace=True)

# ================================================================
# LOAD LATENTS
# ================================================================
Z = np.load(LATENT_PATH)

# reorder to match graph order
mask = np.isin(subject_ids, df_sub["MRI_Exam"].values)
Z = Z[mask]
assert Z.shape[0] == df_sub.shape[0]

# ================================================================
# TRAITS
# ================================================================
TRAITS = {
    "Age":  df_sub["age"].astype(float).values,
    "BMI":  df_sub["BMI"].astype(float).values,
    "APOE": df_sub["APOE"].replace({"E4-":0, "E4+":1}).astype(float).values,
    "Risk": df_sub["risk_for_ad"].astype(str).replace(
        {"0":0, "1":1, "2":2}
    ).astype(float).values,
}

# ================================================================
# STANDARDIZE LATENTS
# ================================================================
Zs = StandardScaler().fit_transform(Z)

# ================================================================
# FIT AXIS VECTORS
# ================================================================
axes = {}
scores = {}
traits_clean = {}

for name, y in TRAITS.items():
    y_clean = y.copy()
    if np.isnan(y_clean).sum() > 0:
        y_clean[np.isnan(y_clean)] = np.nanmedian(y_clean)
    traits_clean[name] = y_clean

    reg = Ridge(alpha=1.0)
    reg.fit(Zs, y_clean)

    v = reg.coef_ / np.linalg.norm(reg.coef_)
    axes[name] = v
    scores[name] = Zs @ v

    np.save(os.path.join(AXIS_DIR, f"{name}_axis.npy"), v)
    np.savetxt(os.path.join(SCORE_DIR, f"{name}_scores.csv"), scores[name], delimiter=",")


# ================================================================
# UMAP EMBEDDING
# ================================================================
def compute_umap(Zs):
    reducer = umap.UMAP(
        n_neighbors=min(15, Zs.shape[0] - 1),
        min_dist=0.05,
        random_state=42
    )
    return reducer.fit_transform(Zs)

emb = compute_umap(Zs)


# ================================================================
# DRAW AXIS ARROW
# ================================================================
def add_axis_arrow(ax, emb, Zs, axis_vec, label):
    lr = Ridge(alpha=1.0)
    lr.fit(Zs, emb)     # emb is n × 2

    direction = lr.coef_ @ axis_vec
    direction /= np.linalg.norm(direction)

    origin = emb.mean(axis=0)
    end = origin + 2 * direction

    ax.annotate(
        "",
        xy=end,
        xytext=origin,
        arrowprops=dict(
            facecolor="crimson",
            edgecolor="crimson",
            width=2,
            headwidth=10,
            alpha=0.85
        )
    )
    ax.text(end[0], end[1], f"{label} axis", color="crimson", fontsize=10, fontweight="bold")


# ================================================================
# 1) UMAP COLORED BY AXIS SCORE
# ================================================================
for name in TRAITS:
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=220)

    sc = ax.scatter(emb[:,0], emb[:,1], c=scores[name],
                    cmap="viridis", s=45, edgecolor="none")
    cbar = plt.colorbar(sc)
    cbar.set_label(f"{name} axis score", fontsize=9)

    add_axis_arrow(ax, emb, Zs, axes[name], name)

    ax.set_title(f"{name} axis projection (K={GRAPH_K}, {METRIC})")
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(UMAP_AXIS, f"{name}.png"))
    plt.close()


# ================================================================
# 2) RAW TRAIT vs AXIS SCORE — SIDE-BY-SIDE PANELS
# ================================================================
for name in TRAITS:
    fig, axarr = plt.subplots(1, 2, figsize=(11, 4.5), dpi=220)

    # LEFT: raw trait
    sc0 = axarr[0].scatter(emb[:,0], emb[:,1], c=traits_clean[name],
                           cmap="viridis", s=45, edgecolor="none")
    plt.colorbar(sc0, ax=axarr[0]).set_label(f"{name} (raw)")
    axarr[0].set_title(f"Raw {name}")
    axarr[0].set_xticks([]); axarr[0].set_yticks([])

    # RIGHT: axis score
    sc1 = axarr[1].scatter(emb[:,0], emb[:,1], c=scores[name],
                           cmap="viridis", s=45, edgecolor="none")
    plt.colorbar(sc1, ax=axarr[1]).set_label(f"{name} axis score")

    add_axis_arrow(axarr[1], emb, Zs, axes[name], name)

    axarr[1].set_title(f"{name} axis")
    axarr[1].set_xticks([]); axarr[1].set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(UMAP_PAIR, f"{name}_pair.png"))
    plt.close()


# ================================================================
# 3) SCATTER: AXIS SCORE vs RAW TRAIT
# ================================================================
for name in TRAITS:
    x = scores[name]
    y = traits_clean[name]

    fig, ax = plt.subplots(figsize=(5.8, 4.5), dpi=220)
    ax.scatter(x, y, s=45, alpha=0.7)

    # Regression line
    a, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    ax.plot(xx, a*xx + b, color="black", linewidth=2)

    r = np.corrcoef(x, y)[0,1]
    ax.set_title(f"{name}: axis score vs raw\nr={r:.2f}", fontsize=11)
    ax.set_xlabel(f"{name} axis score")
    ax.set_ylabel(f"{name} raw")

    plt.tight_layout()
    plt.savefig(os.path.join(SCATTER, f"{name}_scatter.png"))
    plt.close()


# ================================================================
# 4) LATENT DIMENSION LOADINGS
# ================================================================
dims = np.arange(1, Zs.shape[1]+1)

for name in TRAITS:
    v = axes[name]

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=220)
    ax.bar(dims, v)
    ax.axhline(0, color="black")

    ax.set_title(f"{name} axis loadings (K={GRAPH_K}, {METRIC})")
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Loading")

    plt.tight_layout()
    plt.savefig(os.path.join(LOADINGS, f"{name}_loadings.png"))
    plt.close()


print("\n✔ FINISHED PART 5 — All plots and axes saved in:")
print(OUTDIR)
