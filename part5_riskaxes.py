#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PART 5 — TRAIT-ALIGNED LATENT AXES ("risk axes")
Author: Alexandra Badea (with ChatGPT)
Date: 2025-11-20

Produces Age, BMI, APOE, Risk axes for each K and metric, and:
  • UMAP colored by axis scores with axis arrows
  • Side-by-side UMAPs: raw trait vs axis score
  • Scatterplots: axis score vs raw trait
  • Barplots of axis loadings (latent dimensions)
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

# ------------------------------------------------
# Paths (EDIT GRAPH_K and METRIC as needed)
# ------------------------------------------------
ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation"
COLUMNS_ROOT = os.path.join(ROOT, "columns4gaudi111825", "columna-analyses111925")
MDATA_PATH = os.path.join(ROOT, "processed_graph_data", "metadata_with_PCs.xlsx")

GRAPH_K = 25                       # << set this (e.g., 4, 6, 8, 10, 12, 20, 25, 30, 50)
METRIC  = "EUCLID"                 # << CORR, EUCLID, or WASS

# ================================================================
# Resolve latent paths
# ================================================================
LATENT_PATH = os.path.join(
    COLUMNS_ROOT,
    f"latent_k{GRAPH_K}",
    f"latent_epochs_Joint_{METRIC}",
    f"latent_final_Joint_{METRIC}.npy"
)

OUTDIR = os.path.join(
    COLUMNS_ROOT,
    f"trait_axes_K{GRAPH_K}_{METRIC}_Joint"
)
os.makedirs(OUTDIR, exist_ok=True)

# Sub-directories
AXIS_DIR = os.path.join(OUTDIR, "axes")
SCORE_DIR = os.path.join(OUTDIR, "axis_scores")
PLOT_DIR  = os.path.join(OUTDIR, "plots")

UMAP_SINGLE_DIR = os.path.join(PLOT_DIR, "umap_axis_only")
UMAP_PAIR_DIR   = os.path.join(PLOT_DIR, "umap_pairs_raw_vs_axis")
SCATTER_DIR     = os.path.join(PLOT_DIR, "trait_vs_axis_scatter")
LOADING_DIR     = os.path.join(PLOT_DIR, "axis_loadings")

for d in [AXIS_DIR, SCORE_DIR, PLOT_DIR,
          UMAP_SINGLE_DIR, UMAP_PAIR_DIR, SCATTER_DIR, LOADING_DIR]:
    os.makedirs(d, exist_ok=True)

# ================================================================
# Load metadata
# ================================================================
print("Loading metadata from:", MDATA_PATH)
df = pd.read_excel(MDATA_PATH)
df["MRI_Exam"] = df["MRI_Exam"].astype(str).str.zfill(5)

# ================================================================
# Load corresponding graphs to extract subject order
# ================================================================
GRAPH_PT = os.path.join(
    COLUMNS_ROOT,
    "graphs_knn",
    f"k{GRAPH_K}",
    f"md_shared_knn_k{GRAPH_K}_corr_euclid_wass.pt"
)

print("Loading graphs from:", GRAPH_PT)
graphs = torch.load(GRAPH_PT, map_location="cpu")
subject_ids = [str(getattr(g, "subject_id")).zfill(5) for g in graphs]

df_sub = df[df["MRI_Exam"].isin(subject_ids)].copy()
df_sub["__order"] = df_sub["MRI_Exam"].apply(lambda s: subject_ids.index(s))
df_sub = df_sub.sort_values("__order").reset_index(drop=True)
df_sub.drop(columns="__order", inplace=True)

print("Aligned metadata shape:", df_sub.shape)

# ================================================================
# Load latents and align rows
# ================================================================
print("Loading latents from:", LATENT_PATH)
Z = np.load(LATENT_PATH)
if Z.shape[0] != len(subject_ids):
    raise RuntimeError(
        f"Latent rows ({Z.shape[0]}) != graph subject rows ({len(subject_ids)})"
    )

# mask to reorder
keep_mask = np.isin(subject_ids, df_sub["MRI_Exam"].values)
Z = Z[keep_mask]
assert Z.shape[0] == df_sub.shape[0]

print("Latent shape after alignment:", Z.shape)

# ================================================================
# Define traits to model
# ================================================================
TRAITS = {
    "Age":  df_sub["age"].values.astype(float),
    "BMI":  df_sub["BMI"].values.astype(float),
    "APOE": df_sub["APOE"].replace({"E4-": 0, "E4+": 1}).astype(float).values,
    "Risk": df_sub["risk_for_ad"].astype(str).replace(
        {"0": 0, "1": 1, "2": 2}
    ).astype(float).values,
}

# ================================================================
# Helper — UMAP
# ================================================================
def compute_umap(Z):
    reducer = umap.UMAP(
        n_neighbors=min(15, Z.shape[0] - 1),
        min_dist=0.05,
        spread=1.0,
        n_components=2,
        random_state=42,
    )
    return reducer.fit_transform(Z)

# ================================================================
# Standardize latent space
# ================================================================
Zs = StandardScaler().fit_transform(Z)

# ================================================================
# Fit axes (Ridge regression Zs → trait)
# ================================================================
axes = {}
scores = {}
traits_clean = {}
regressors = {}

for name, y in TRAITS.items():
    y_clean = np.copy(y)
    nanmask = np.isnan(y_clean)
    if nanmask.sum() > 0:
        y_clean[nanmask] = np.nanmedian(y_clean)

    traits_clean[name] = y_clean

    reg = Ridge(alpha=1.0)
    reg.fit(Zs, y_clean)

    v = reg.coef_.astype(float)
    v = v / np.linalg.norm(v)

    axes[name] = v
    scores[name] = Zs @ v
    regressors[name] = reg

    np.save(os.path.join(AXIS_DIR, f"{name}_axis.npy"), v)
    np.savetxt(
        os.path.join(SCORE_DIR, f"{name}_scores.csv"),
        scores[name],
        delimiter=","
    )

print("Saved axes and axis scores to:", AXIS_DIR, "and", SCORE_DIR)

# ================================================================
# UMAP embedding (shared across traits)
# ================================================================
emb = compute_umap(Zs)

# ================================================================
# Helper — draw axis arrow in UMAP space
# ================================================================
def annotate_axis_vector(ax, emb, Zs, axis_vector, color="crimson", label="axis"):
    """
    Draws an arrow indicating the direction of the trait axis in UMAP space.
    Uses a linear regression mapping from latent Zs to UMAP coordinates.
    """
    lr = Ridge(alpha=1.0)
    lr.fit(Zs, emb)           # emb is (n_samples × 2)

    # Project axis_vector into UMAP space
    # lr.coef_: shape (2 × latent_dim)
    direction = lr.coef_ @ axis_vector
    direction = direction / np.linalg.norm(direction)

    start = emb.mean(axis=0)
    end   = start + 2.0 * direction

    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            facecolor=color,
            edgecolor=color,
            width=2,
            headwidth=10,
            alpha=0.85
        )
    )

    ax.text(
        end[0], end[1],
        f"{label} axis",
        color=color,
        fontsize=10,
        fontweight="bold"
    )

# ================================================================
# 1) UMAP colored by axis scores (+arrow) — single-panel per trait
# ================================================================
for name in TRAITS.keys():
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=250)

    sc = ax.scatter(
        emb[:, 0], emb[:, 1],
        c=scores[name],
        cmap="viridis",
        s=40,
        edgecolor="none",
        alpha=0.92
    )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{name} axis score", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    annotate_axis_vector(
        ax,
        emb,
        Zs,
        axes[name],
        color="crimson",
        label=name
    )

    ax.set_title(f"K={GRAPH_K}, {METRIC} latent space\nProjected {name} axis", fontsize=11)
    ax.set_xlabel("UMAP-1", fontsize=9)
    ax.set_ylabel("UMAP-2", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    out = os.path.join(UMAP_SINGLE_DIR, f"UMAP_axis_{name}.png")
    plt.savefig(out, dpi=250)
    plt.close()
    print("Saved:", out)

# ================================================================
# 2) Side-by-side UMAP: raw trait vs axis score
# ================================================================
for name in TRAITS.keys():
    y_clean = traits_clean[name]
    axis_scores = scores[name]

    fig, axes_pair = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=250)

    # Left: raw trait
    sc0 = axes_pair[0].scatter(
        emb[:, 0], emb[:, 1],
        c=y_clean,
        cmap="viridis",
        s=40,
        edgecolor="none",
        alpha=0.92
    )
    cbar0 = plt.colorbar(sc0, ax=axes_pair[0], fraction=0.046, pad=0.04)
    cbar0.set_label(f"{name} (raw)", fontsize=9)
    cbar0.ax.tick_params(labelsize=8)
    axes_pair[0].set_title(f"UMAP colored by raw {name}", fontsize=11)
    axes_pair[0].set_xticks([])
    axes_pair[0].set_yticks([])

    # Right: axis scores + arrow
    sc1 = axes_pair[1].scatter(
        emb[:, 0], emb[:, 1],
        c=axis_scores,
        cmap="viridis",
        s=40,
        edgecolor="none",
        alpha=0.92
    )
    cbar1 = plt.colorbar(sc1, ax=axes_pair[1], fraction=0.046, pad=0.04)
    cbar1.set_label(f"{name} axis score", fontsize=9)
    cbar1.ax.tick_params(labelsize=8)

    annotate_axis_vector(
        axes_pair[1],
        emb,
        Zs,
        axes[name],
        color="crimson",
        label=name
    )

    axes_pair[1].set_title(f"UMAP colored by {name} axis", fontsize=11)
    axes_pair[1].set_xticks([])
    axes_pair[1].set_yticks([])

    fig.suptitle(f"K={GRAPH_K}, {METRIC} — {name}: raw vs axis", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(UMAP_PAIR_DIR, f"UMAP_pair_{name}.png")
    plt.savefig(out, dpi=250)
    plt.close()
    print("Saved:", out)

# ================================================================
# 3) Scatterplots: axis score vs raw trait
# ================================================================
for name in TRAITS.keys():
    y_clean = traits_clean[name]
    axis_scores = scores[name]

    mask = ~np.isnan(y_clean)
    x = axis_scores[mask]
    y = y_clean[mask]

    if len(x) < 3:
        print(f"Not enough data for scatterplot of {name}")
        continue

    # Simple linear fit score→trait
    coef = np.polyfit(x, y, 1)
    a, b = coef
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = a * x_line + b

    # Correlation / R²
    r = np.corrcoef(x, y)[0, 1]
    r2 = r ** 2

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=250)
    ax.scatter(x, y, s=35, alpha=0.75, edgecolor="none")
    ax.plot(x_line, y_line, linewidth=2)

    ax.set_xlabel(f"{name} axis score", fontsize=10)
    ax.set_ylabel(f"{name} (raw)", fontsize=10)
    ax.set_title(
        f"{name}: raw vs axis score\nr = {r:.2f},  R² = {r2:.2f}",
        fontsize=11
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(SCATTER_DIR, f"scatter_{name}.png")
    plt.savefig(out, dpi=250)
    plt.close()
    print("Saved:", out)

# ================================================================
# 4) Barplots of axis loadings (latent dimensions)
# ================================================================
latent_dim = Zs.shape[1]
dims = np.arange(1, latent_dim + 1)

for name in TRAITS.keys():
    v = axes[name]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=250)
    ax.bar(dims, v)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xticks(dims)
    ax.set_xlabel("Latent dimension", fontsize=10)
    ax.set_ylabel("Axis loading", fontsize=10)
    ax.set_title(
        f"{name} axis loadings (K={GRAPH_K}, {METRIC})",
        fontsize=11
    )

    plt.tight_layout()
    out = os.path.join(LOADING_DIR, f"axis_loadings_{name}.png")
    plt.savefig(out, dpi=250)
    plt.close()
    print("Saved:", out)

print("\n✔ FINISHED PART 5 — all axes and plots saved to:", OUTDIR)
