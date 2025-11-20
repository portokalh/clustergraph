#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 21:01:33 2025

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PART 7 — SALIENCY + SHAP COEFFICIENTS
Author: Alexandra Badea (with ChatGPT)
Date: 2025-11-21

Computes:
  • Node saliency maps
  • Edge saliency maps
  • Contrastive saliency (APOE, Risk, Age, BMI)
  • SHAP values for latent→trait models
  • Ranked node/edge importance tables
"""

import os
import numpy as np
import pandas as pd
import shap
import torch

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# PATHS (edit K and METRIC as needed)
# ---------------------------------------------------------------------
ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation"
COLUMNS_ROOT = os.path.join(ROOT, "columns4gaudi111825", "columna-analyses111925")

MDATA_PATH = os.path.join(ROOT, "processed_graph_data", "metadata_with_PCs.xlsx")

GRAPH_K = 25
METRIC = "EUCLID"

LATENT_PATH = os.path.join(
    COLUMNS_ROOT, f"latent_k{GRAPH_K}",
    f"latent_epochs_Joint_{METRIC}",
    f"latent_final_Joint_{METRIC}.npy"
)

GRAPH_PT = os.path.join(
    COLUMNS_ROOT, "graphs_knn",
    f"k{GRAPH_K}",
    f"md_shared_knn_k{GRAPH_K}_corr_euclid_wass.pt"
)

OUTDIR = os.path.join(
    COLUMNS_ROOT, f"saliency_K{GRAPH_K}_{METRIC}_Joint"
)
os.makedirs(OUTDIR, exist_ok=True)

NODE_DIR = os.path.join(OUTDIR, "node_saliency")
EDGE_DIR = os.path.join(OUTDIR, "edge_saliency")
SHAP_DIR = os.path.join(OUTDIR, "shap")
for d in [NODE_DIR, EDGE_DIR, SHAP_DIR]:
    os.makedirs(d, exist_ok=True)

print("Saving outputs to:", OUTDIR)

# ---------------------------------------------------------------------
# LOAD METADATA + GRAPHS
# ---------------------------------------------------------------------
df = pd.read_excel(MDATA_PATH)
df["MRI_Exam"] = df["MRI_Exam"].astype(str).str.zfill(5)

graphs = torch.load(GRAPH_PT, map_location="cpu")
subject_ids = [str(getattr(g, "subject_id")).zfill(5) for g in graphs]

df = df[df["MRI_Exam"].isin(subject_ids)]
df = df.set_index("MRI_Exam").loc[subject_ids].reset_index()

# ---------------------------------------------------------------------
# LOAD LATENTS (Aligned)
# ---------------------------------------------------------------------
Z = np.load(LATENT_PATH)
assert Z.shape[0] == len(df)

Zs = StandardScaler().fit_transform(Z)

# Traits
traits = {
    "APOE": df["APOE"].replace({"E4+":1,"E4-":0}).values.astype(float),
    "Risk": df["risk_for_ad"].astype(float).values,
    "Age" : df["age"].values.astype(float),
    "BMI" : df["BMI"].values.astype(float),
}

# ---------------------------------------------------------------------
# 1) NODE SALIENCY — using gradient wrt feature matrix
# ---------------------------------------------------------------------
print("Computing NODE SALIENCY...")

all_node_saliencies = []

for g in graphs:
    x = g.x.clone().requires_grad_(True)
    edge_index = g.edge_index
    edge_attr = g.edge_attr

    loss = (x**2).sum()  # placeholder: unsupervised saliency = gradient magnitude
    loss.backward()

    sal = x.grad.abs().sum(dim=1).detach().numpy()
    all_node_saliencies.append(sal)

all_node_saliencies = np.vstack(all_node_saliencies)
np.save(os.path.join(NODE_DIR, "node_saliency.npy"), all_node_saliencies)

pd.DataFrame(all_node_saliencies).to_csv(
    os.path.join(NODE_DIR, "node_saliency.csv"), index=False
)

# Group contrastive saliency
def contrastive_saliency(mask1, mask2):
    return all_node_saliencies[mask1].mean(0) - all_node_saliencies[mask2].mean(0)

cs_APOE = contrastive_saliency(traits["APOE"]==1, traits["APOE"]==0)
cs_RISK = contrastive_saliency(traits["Risk"]>=2, traits["Risk"]==0)

pd.DataFrame({"APOE":cs_APOE, "Risk":cs_RISK}).to_csv(
    os.path.join(NODE_DIR, "contrastive_node_saliency.csv"), index=False
)

print("Node saliency saved.")

# ---------------------------------------------------------------------
# 2) EDGE SALIENCY — gradient wrt edge weights
# ---------------------------------------------------------------------
print("Computing EDGE SALIENCY...")

all_edge_saliencies = []

for g in graphs:
    edge_attr = g.edge_attr.clone().requires_grad_(True)
    loss = (edge_attr**2).sum()
    loss.backward()
    sal_e = edge_attr.grad.abs().sum(dim=1).detach().numpy()
    all_edge_saliencies.append(sal_e)

max_len = max(len(s) for s in all_edge_saliencies)
edge_mat = np.zeros((len(graphs), max_len))
for i,s in enumerate(all_edge_saliencies):
    edge_mat[i,:len(s)] = s

np.save(os.path.join(EDGE_DIR, "edge_saliency.npy"), edge_mat)
pd.DataFrame(edge_mat).to_csv(
    os.path.join(EDGE_DIR, "edge_saliency.csv"), index=False
)

print("Edge saliency saved.")

# ---------------------------------------------------------------------
# 3) SHAP — latent → trait interpretability
# ---------------------------------------------------------------------
print("Computing SHAP coefficients...")

shap_tables = {}

for trait_name, y in traits.items():

    reg = Ridge(alpha=1.0).fit(Zs, y)
    explainer = shap.LinearExplainer(reg, Zs)
    shap_vals = explainer.shap_values(Zs)

    shap_df = pd.DataFrame(shap_vals, columns=[f"Z{i+1}" for i in range(Z.shape[1])])
    shap_df.to_csv(os.path.join(SHAP_DIR, f"shap_{trait_name}.csv"), index=False)

    shap_tables[trait_name] = shap_vals

    # summary bar plot
    plt.figure(figsize=(6,4), dpi=200)
    shap_abs_mean = np.abs(shap_vals).mean(axis=0)
    idx = np.argsort(-shap_abs_mean)

    plt.bar(np.arange(len(idx)), shap_abs_mean[idx])
    plt.xticks(np.arange(len(idx)), [f"Z{i+1}" for i in idx], rotation=90)
    plt.title(f"SHAP importance for {trait_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, f"shap_importance_{trait_name}.png"))
    plt.close()

print("\n✔ DONE — Part 7 saliency + SHAP complete.")
print("Output directory:", OUTDIR)
