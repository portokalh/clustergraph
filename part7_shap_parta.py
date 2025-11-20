#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIFIED PART 7 — Interpretable GAUDI (K=25) with APOE Axis

Runs (for APOE axis):
  7-1  Global Region SHAP (via laminar features)
  7a   Depth-wise SHAP (21 MD + 21 QSM)
  7b   Laminar SHAP (Supra / Granular / Infra × MD/QSM)
  7c   Region × Laminar Heatmap
  7d   Subject-Level Latent SHAP (GAUDI latents)
"""

# ============= IMPORTS ======================================
import os
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================
AXIS_NAME = "APOE"      # "APOE", "Age", "BMI", "Risk"
K = 25
METRIC = "EUCLID"

ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation"
CROOT = os.path.join(ROOT, "columns4gaudi111825", "columna-analyses111925")

# Subject-level GAUDI latents (Joint EUCLID, K=25)
FALLBACK_LATENT = os.path.join(
    CROOT,
    "latent_k25",
    "latent_epochs_Joint_EUCLID",
    "latent_final_Joint_EUCLID.npy"
)

# APOE (or other trait) axis from Part 5
FALLBACK_AXIS = os.path.join(
    CROOT,
    "trait_axes_K25_EUCLID_Joint",
    "axes",
    f"{AXIS_NAME}_axis.npy"
)

# Rebuilt & z-scored depth profiles (from your depth-profile script)
PROFILES_DIR = os.path.join(ROOT, "columns4gaudi111825", "profiles_depth21")
MD_FILE      = os.path.join(PROFILES_DIR, "md_profiles.npy")
QSM_FILE     = os.path.join(PROFILES_DIR, "qsm_profiles.npy")

# Region labels (DK68 names, 1 per region)
REGION_FILE  = os.path.join(
    ROOT, "columns4gaudi111825", "utilities", "node_region_labels.csv"
)

# Main results directory for this axis
RESULTS_DIR = os.path.join(
    ROOT, "columns4gaudi111825", f"results_k{K}_{AXIS_NAME}"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Optional: cache latents in this results folder
Z_NODES_FILE = os.path.join(RESULTS_DIR, "Z_subjects.npy")


# ------------------------------------------------------------
# Utility loaders
# ------------------------------------------------------------
def load_subject_latents():
    """
    Load subject-level GAUDI latents.

    If a cached copy exists in RESULTS_DIR, use that.
    Otherwise, load from FALLBACK_LATENT and cache.
    """
    if os.path.exists(Z_NODES_FILE):
        print(f"✔ Using cached subject latents: {Z_NODES_FILE}")
        return np.load(Z_NODES_FILE)

    print("⚠ Cached Z_subjects.npy missing — using fallback latents:")
    print("   ", FALLBACK_LATENT)
    Z = np.load(FALLBACK_LATENT)
    np.save(Z_NODES_FILE, Z)
    print(f"✔ Saved subject latents → {Z_NODES_FILE}")
    return Z


def load_axis():
    """Load APOE (or chosen) axis vector from Part 5 outputs."""
    if not os.path.exists(FALLBACK_AXIS):
        raise FileNotFoundError(f"Axis file missing → {FALLBACK_AXIS}")
    print(f"✔ Using axis from {FALLBACK_AXIS}")
    return np.load(FALLBACK_AXIS)


def laminar(md, qsm):
    """
    Compute 6 laminar features from 21-depth MD/QSM profiles:

        Supra    : depths 0..6
        Granular : depths 7..13
        Infra    : depths 14..20

    Returns:
        (N_nodes, 6) array
    """
    supra = slice(0, 7)
    gran  = slice(7, 14)
    infra = slice(14, 21)

    return np.stack([
        md[:, supra].mean(1), md[:, gran].mean(1), md[:, infra].mean(1),
        qsm[:, supra].mean(1), qsm[:, gran].mean(1), qsm[:, infra].mean(1)
    ], axis=1)


# ------------------------------------------------------------
# LOAD INPUTS
# ------------------------------------------------------------
print("\n======================")
print(" LOADING INPUTS…")
print("======================")

# Subject-level latents and axis (from Part 5 setup)
Z_subj = load_subject_latents()   # shape = (N_subj_latent, latent_dim)
w_axis = load_axis()              # shape = (latent_dim,)

N_subj_latent, latent_dim = Z_subj.shape
print(f"✔ Subject-level latents: {Z_subj.shape}")

# Depth-wise profiles (z-scored) — stacked across subjects × regions
md_profiles  = np.load(MD_FILE)   # shape = (N_nodes, 21)
qsm_profiles = np.load(QSM_FILE)  # shape = (N_nodes, 21)

print(f"✔ MD profiles:  {md_profiles.shape}")
print(f"✔ QSM profiles: {qsm_profiles.shape}")

if not os.path.exists(REGION_FILE):
    raise FileNotFoundError(f"Region label file missing → {REGION_FILE}")

region_labels = pd.read_csv(REGION_FILE)["region"].values
N_regions = len(region_labels)
print(f"✔ Unique regions: {N_regions}")

# ------------------------------------------------------------
# Derive node/subject counts and expand region labels
# ------------------------------------------------------------
N_nodes = md_profiles.shape[0]
if N_nodes % N_regions != 0:
    raise ValueError(f"N_nodes={N_nodes} not divisible by N_regions={N_regions}")

N_subj_profiles = N_nodes // N_regions
print(f"✔ Profiles correspond to {N_subj_profiles} subjects × {N_regions} regions")

# Expand region labels so each row in md_profiles has a region label
region_labels_expanded = np.tile(region_labels, N_subj_profiles)
assert len(region_labels_expanded) == N_nodes

# Align subject-level latents with subjects used in depth profiles
if N_subj_latent < N_subj_profiles:
    raise ValueError(
        f"Subject latents ({N_subj_latent}) < profile subjects ({N_subj_profiles})"
    )

if N_subj_latent > N_subj_profiles:
    print(f"⚠ Latent subjects ({N_subj_latent}) > profile subjects ({N_subj_profiles});")
    print("  Using first N_subj_profiles latents to match profiles.")
    Z_subj_use = Z_subj[:N_subj_profiles, :]
else:
    Z_subj_use = Z_subj

# Subject-level axis scores
subj_scores = Z_subj_use @ w_axis        # shape = (N_subj_profiles,)

# Node-level target: repeat subject score across that subject's 68 regions
node_scores = np.repeat(subj_scores, N_regions)   # shape = (N_nodes,)
assert node_scores.shape[0] == N_nodes

print("✔ node_scores shape:", node_scores.shape)


# ============================================================
# PART 7b / 7-1 — LAMINAR SHAP + GLOBAL REGION SHAP
# ============================================================
print("\n===============================")
print(" PART 7b / 7-1 — Laminar SHAP ")
print("      + Global Region SHAP     ")
print("===============================")

laminar_X = laminar(md_profiles, qsm_profiles)   # (N_nodes, 6)
lam_feat = [
    "MD_supra", "MD_granular", "MD_infra",
    "QSM_supra", "QSM_granular", "QSM_infra"
]

# Lasso on laminar features → node_scores
lasso_lam = Lasso(alpha=0.001)
lasso_lam.fit(laminar_X, node_scores)

expl_lam = shap.Explainer(lasso_lam, laminar_X, feature_names=lam_feat)
sh_lam   = expl_lam(laminar_X)
abs_sh_lam = np.abs(sh_lam.values)   # (N_nodes, 6)

# ---------- Save node-level laminar SHAP ----------
lam_node_df = pd.DataFrame(abs_sh_lam, columns=lam_feat)
lam_node_df["node_score"] = node_scores
lam_node_df["region"] = region_labels_expanded
lam_node_df.to_csv(
    os.path.join(RESULTS_DIR, "laminar_SHAP_nodevalues.csv"),
    index=False
)
print("✔ Saved laminar_SHAP_nodevalues.csv")

# ---------- Global laminar importance ----------
lam_global = abs_sh_lam.mean(axis=0)
lam_global_df = pd.DataFrame({
    "feature": lam_feat,
    "global_mean_absSHAP": lam_global
})
lam_global_df.to_csv(
    os.path.join(RESULTS_DIR, "laminar_SHAP_global.csv"),
    index=False
)
print("✔ Saved laminar_SHAP_global.csv")

# ---------- Region × lamina SHAP (mean |SHAP| per region/lamina) ----------
lam_region_df = (
    lam_node_df
    .groupby("region")[lam_feat]
    .mean()
)
lam_region_df.to_csv(
    os.path.join(RESULTS_DIR, "laminar_SHAP_byregion.csv")
)
print("✔ Saved laminar_SHAP_byregion.csv")

# ---------- Global REGION SHAP (one number per region) ----------
# Sum laminar contributions per region to get a global region importance
region_global = lam_region_df.sum(axis=1)
region_global = region_global.sort_values(ascending=False)

region_global.to_csv(
    os.path.join(RESULTS_DIR, "region_global_SHAP.csv"),
    header=["global_SHAP"]
)
print("✔ Saved region_global_SHAP.csv")

# Barplot of top regions
topN = 30
top_reg = region_global.head(topN)

plt.figure(figsize=(10, 12))
plt.barh(top_reg.index[::-1], top_reg.values[::-1])
plt.title(f"Top Regions Driving {AXIS_NAME} Axis (K={K})")
plt.xlabel("Global SHAP (sum over laminae)")
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "region_global_SHAP_barplot.png"),
    dpi=300
)
plt.close()
print("✔ Saved region_global_SHAP_barplot.png")


# ============================================================
# PART 7a — DEPTH-WISE SHAP (42 features)
# ============================================================
print("\n===============================")
print(" PART 7a — Depth-wise SHAP     ")
print("===============================")

depth_feature_names = [f"MD_d{i}" for i in range(21)] + [f"QSM_d{i}" for i in range(21)]
X_depth = np.hstack([md_profiles, qsm_profiles])   # (N_nodes, 42)

lasso_depth = Lasso(alpha=0.0005)
lasso_depth.fit(X_depth, node_scores)

expl_depth = shap.Explainer(lasso_depth, X_depth, feature_names=depth_feature_names)
sh_depth   = expl_depth(X_depth)
abs_sh_depth = np.abs(sh_depth.values)

# Save node-level depth SHAP
pd.DataFrame(abs_sh_depth, columns=depth_feature_names).to_csv(
    os.path.join(RESULTS_DIR, "depth_SHAP_nodevalues.csv"),
    index=False
)
print("✔ Saved depth_SHAP_nodevalues.csv")

# Global depth-wise importance
depth_global = abs_sh_depth.mean(axis=0)
depth_global_df = pd.DataFrame({
    "feature": depth_feature_names,
    "mean_abs_SHAP": depth_global
})
depth_global_df.to_csv(
    os.path.join(RESULTS_DIR, "depth_SHAP_global.csv"),
    index=False
)
print("✔ Saved depth_SHAP_global.csv")

# Top-20 depth features barplot
idx_top = np.argsort(depth_global)[::-1][:20]
plt.figure(figsize=(10, 8))
plt.barh(
    [depth_feature_names[i] for i in idx_top][::-1],
    depth_global[idx_top][::-1]
)
plt.title(f"Top 20 Depth Features ({AXIS_NAME} axis)")
plt.xlabel("Mean |SHAP|")
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "depth_SHAP_top20.png"),
    dpi=300
)
plt.close()
print("✔ Saved depth_SHAP_top20.png")


# ============================================================
# PART 7c — REGION × LAMINA HEATMAP
# ============================================================
print("\n===============================")
print(" PART 7c — Region × Lamina Heatmap")
print("===============================")

df_heat = lam_region_df.copy()   # (68 regions × 6 lamina-features)

cmap = sns.color_palette("viridis", as_cmap=True)
g = sns.clustermap(
    df_heat,
    cmap=cmap,
    linewidths=0.5,
    figsize=(12, 14),
    cbar_kws={"label": "Mean |SHAP|"}
)

plt.title(f"Region × Lamina SHAP — {AXIS_NAME} axis", pad=80)
heat_png = os.path.join(RESULTS_DIR, "laminar_SHAP_heatmap.png")
heat_pdf = os.path.join(RESULTS_DIR, "laminar_SHAP_heatmap.pdf")
g.fig.savefig(heat_png, dpi=300, bbox_inches='tight')
g.fig.savefig(heat_pdf, dpi=300, bbox_inches='tight')
plt.close()
print("✔ Saved laminar_SHAP_heatmap.png / .pdf")


# ============================================================
# PART 7d — Subject-Level SHAP on GAUDI Latent Dimensions
# ============================================================
print("\n===============================")
print(" PART 7d — Latent SHAP (Subject-level)")
print("===============================")

LATENT_FILE = FALLBACK_LATENT
AXIS_DIR = os.path.join(
    CROOT,
    f"trait_axes_K{K}_{METRIC}_Joint",
    "axes"
)
AXIS_FILE = os.path.join(AXIS_DIR, f"{AXIS_NAME}_axis.npy")

LATENT_OUTDIR = os.path.join(CROOT, f"shap_latents_K{K}_{AXIS_NAME}")
os.makedirs(LATENT_OUTDIR, exist_ok=True)

# Load latents and axis
print("🔵 Loading GAUDI latents (subject-level)…")
Z_lat = np.load(LATENT_FILE)   # shape = (N_subj_latent, latent_dim)
Nsubj_lat, D_lat = Z_lat.shape
print(f"✔ Latent matrix shape = {Z_lat.shape} (subjects × latent_dim)")

print(f"🔵 Loading axis from {AXIS_FILE}")
axis_vec = np.load(AXIS_FILE)   # shape = (latent_dim,)

# Subject-level target
target_lat = Z_lat @ axis_vec
print("✔ Subject-level target vector shape:", target_lat.shape)

# Fit Ridge + SHAP on latents
print("🔵 Fitting Ridge model + SHAP on latents…")
model_lat = Ridge(alpha=1.0)
model_lat.fit(Z_lat, target_lat)

explainer_lat = shap.Explainer(model_lat, Z_lat)
shap_values_lat = explainer_lat(Z_lat)     # (N_subj × D_lat)
sv_lat = shap_values_lat.values

# Save per-subject latent SHAP
subj_ids = [f"subj_{i:03d}" for i in range(Nsubj_lat)]
latent_cols = [f"latent_{i+1}" for i in range(D_lat)]

df_subj_lat = pd.DataFrame(sv_lat, index=subj_ids, columns=latent_cols)
df_subj_lat.to_csv(
    os.path.join(LATENT_OUTDIR, "shap_latents_per_subject.csv")
)
print("✔ Saved shap_latents_per_subject.csv")

# Global latent importance
global_imp_lat = np.mean(np.abs(sv_lat), axis=0)
df_global_lat = pd.DataFrame({
    "latent_dim": latent_cols,
    "global_mean_absSHAP": global_imp_lat
})
df_global_lat.to_csv(
    os.path.join(LATENT_OUTDIR, "shap_latents_global.csv"),
    index=False
)
print("✔ Saved shap_latents_global.csv")

# Heatmap (subjects × latent dims)
plt.figure(figsize=(14, 8))
cmap2 = sns.color_palette("viridis", as_cmap=True)

sns.heatmap(
    df_subj_lat,
    cmap=cmap2,
    cbar_kws={"label": "|SHAP|"},
    xticklabels=True,
    yticklabels=False
)

plt.title(f"Subject-Level Latent SHAP (K={K}, {METRIC}, axis={AXIS_NAME})")
plt.tight_layout()

plt.savefig(
    os.path.join(LATENT_OUTDIR, "shap_latent_heatmap.png"),
    dpi=300
)
plt.savefig(
    os.path.join(LATENT_OUTDIR, "shap_latent_heatmap.pdf"),
    dpi=300
)
plt.close()
print("✔ Saved latent SHAP heatmap (PNG/PDF)")

print("\n🎉 FINISHED UNIFIED PART 7")
print(f"Depth/laminar/region results → {RESULTS_DIR}")
print(f"Latent SHAP results         → {LATENT_OUTDIR}")




# ============================================================
# PART 7e — Top-10 Regions × Lamina SHAP (paper figure)
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os




# ============================================================
# PART 7e — Top-10 Regions × Lamina SHAP Panel (Paper-ready)
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns

print("\n============================================")
print(" PART 7e — Top 10 Regions × Lamina Panel")
print("============================================")

lam_region_path = os.path.join(RESULTS_DIR, "laminar_SHAP_byregion.csv")
lam_region = pd.read_csv(lam_region_path, index_col=0)

# Compute global importance
lam_region["global_SHAP"] = lam_region.sum(axis=1)

# Select top 10 regions
top10 = lam_region.sort_values("global_SHAP", ascending=False).iloc[:10]

# Remove global column for heatmap
top10_lam = top10.drop(columns=["global_SHAP"])

plt.figure(figsize=(8, 6))
sns.heatmap(
    top10_lam,
    cmap="viridis",
    linewidths=0.5,
    annot=True,
    fmt=".3f",
    cbar_kws={"label": "Mean |SHAP|"},
)

plt.title(f"Top-10 Regions × Lamina SHAP (Axis = {AXIS_NAME})", fontsize=14)
plt.tight_layout()

out_png = os.path.join(RESULTS_DIR, "top10_regions_lamina_heatmap.png")
plt.savefig(out_png, dpi=300)
plt.close()

print("✔ Saved:", out_png)



# ============================================================
# PART 7f — 3D Brain Map (Region SHAP using MNI DK68 centroids)
# ============================================================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection

print("\n============================================")
print(" PART 7f — 3D Brain Map (Region SHAP using MNI DK68 centroids)")
print("============================================")

# Load region global SHAP
region_global_path = os.path.join(RESULTS_DIR, "region_global_SHAP.csv")
reg_shap = pd.read_csv(region_global_path, index_col=0)

# Load DK68 centroids (user's file)
coord_path = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation/columns4gaudi111825/utilities/dk68_centroids_mni.csv"
coords = pd.read_csv(coord_path)

# ---- Detect region name column automatically ----
possible_region_cols = ["region", "label", "ROI", "Region", "name"]
region_col = None
for c in possible_region_cols:
    if c in coords.columns:
        region_col = c
        break
if region_col is None:
    raise ValueError(f"❌ No region name column found in centroid file. Columns={coords.columns}")

coords = coords.set_index(region_col)

# ---- Detect coordinate columns automatically ----
possible_X = ["X","x","coord_x","mni_x","R","RAS_R"]
possible_Y = ["Y","y","coord_y","mni_y","A","RAS_A"]
possible_Z = ["Z","z","coord_z","mni_z","S","RAS_S"]

def detect_col(possible, cols):
    for p in possible:
        if p in cols:
            return p
    raise ValueError(f"❌ Could not detect coordinate column among {possible}")

Xcol = detect_col(possible_X, coords.columns)
Ycol = detect_col(possible_Y, coords.columns)
Zcol = detect_col(possible_Z, coords.columns)

print(f"✔ Detected coordinate columns: {Xcol}, {Ycol}, {Zcol}")

# reorder coords to match region order in SHAP file
coords = coords.loc[reg_shap.index]

# Extract coordinates
X = coords[Xcol].values
Y = coords[Ycol].values
Z = coords[Zcol].values

# Marker color = region SHAP
color = reg_shap["global_SHAP"].values
size = 300 * (color / color.max())

# ---- 3D Scatter Plot ----
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X, Y, Z,
    c=color,
    s=size,
    cmap="viridis",
    alpha=0.95,
    edgecolor="k",
    linewidth=0.4,
)

cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
cbar.set_label("Global Region SHAP", fontsize=11)

ax.set_title(f"3D Brain Map — Region SHAP ({AXIS_NAME} axis)", fontsize=14)
ax.set_xlabel("X (MNI)")
ax.set_ylabel("Y (MNI)")
ax.set_zlabel("Z (MNI)")

# Annotate top regions
topN = 10
top_regs = reg_shap.sort_values("global_SHAP", ascending=False).head(topN)

for region in top_regs.index:
    xi, yi, zi = coords.loc[region, [Xcol, Ycol, Zcol]]
    ax.text(xi, yi, zi, region, fontsize=7)

plt.tight_layout()

out3d = os.path.join(RESULTS_DIR, "brain3D_region_SHAP.png")
plt.savefig(out3d, dpi=350)
plt.close()

print("✔ Saved 3D Brain SHAP plot:", out3d)









#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PART 7g — DK68 SHAP on fsaverage5 surface (patch-like map)

Uses:
  - region_global_SHAP.csv  (Unified Part 7 output)
  - dk68_centroids_mni.csv  (DK68 region centroids in MNI)

Creates:
  - DK68_SHAP_fsavg5_4panel.png  (4 views, 200 dpi)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import datasets, surface, plotting
from scipy.spatial import cKDTree

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
SHAP_FILE = (
    "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation/"
    "columns4gaudi111825/results_k25_APOE/region_global_SHAP.csv"
)

CENTROID_FILE = (
    "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation/"
    "columns4gaudi111825/utilities/dk68_centroids_mni.csv"
)

OUT_PNG = (
    "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation/"
    "columns4gaudi111825/results_k25_APOE/DK68_SHAP_fsavg5_4panel.png"
)

# ------------------------------------------------------------
# Load SHAP + DK68 centroids
# ------------------------------------------------------------
print("🔵 Loading region SHAP and DK68 centroids...")
shap_df = pd.read_csv(SHAP_FILE, index_col=0)
shap_series = shap_df["global_SHAP"]  # index = region name

centroids = pd.read_csv(CENTROID_FILE)   # columns: region, x, y, z
centroids["shap"] = centroids["region"].map(shap_series).fillna(0.0)

print("  Regions with SHAP:", shap_series.shape[0])
print("  Centroids loaded:", centroids.shape[0])

# ------------------------------------------------------------
# Fetch fsaverage5 surfaces
# ------------------------------------------------------------
print("🔵 Fetching fsaverage5 surfaces from nilearn...")
fsavg5 = datasets.fetch_surf_fsaverage("fsaverage5")

coords_l, faces_l = surface.load_surf_mesh(fsavg5.pial_left)
coords_r, faces_r = surface.load_surf_mesh(fsavg5.pial_right)

print("  Left  hemi vertices:", coords_l.shape[0])
print("  Right hemi vertices:", coords_r.shape[0])

# ------------------------------------------------------------
# Assign SHAP to each vertex by nearest centroid (per hemisphere)
# ------------------------------------------------------------
print("🔵 Mapping DK68 regions → fsaverage5 vertices...")

# Split centroids by hemisphere
lh_cent = centroids[centroids["region"].str.startswith("lh_")].copy()
rh_cent = centroids[centroids["region"].str.startswith("rh_")].copy()

# Build KD-trees in MNI-ish coords
tree_l = cKDTree(lh_cent[["x", "y", "z"]].values)
tree_r = cKDTree(rh_cent[["x", "y", "z"]].values)

# Query nearest centroid for each surface vertex
dist_l, idx_l = tree_l.query(coords_l)
dist_r, idx_r = tree_r.query(coords_r)

lh_shap = lh_cent["shap"].values[idx_l]
rh_shap = rh_cent["shap"].values[idx_r]

# Optionally, clip extremes slightly for nicer color range
all_vals = np.concatenate([lh_shap, rh_shap])
vmin = np.percentile(all_vals, 5)
vmax = np.percentile(all_vals, 95)

print(f"  SHAP value range (5–95th pct): {vmin:.3f} .. {vmax:.3f}")

# ------------------------------------------------------------
# Multi-panel plotting
# ------------------------------------------------------------
print("🔵 Rendering fsaverage5 4-panel DK68 SHAP figure...")

fig = plt.figure(figsize=(16, 8), dpi=200)

views = [
    ("Lateral LH", fsavg5.infl_left,  lh_shap, "left",  "lateral", 221),
    ("Medial LH",  fsavg5.infl_left,  lh_shap, "left",  "medial",  222),
    ("Lateral RH", fsavg5.infl_right, rh_shap, "right", "lateral", 223),
    ("Medial RH",  fsavg5.infl_right, rh_shap, "right", "medial",  224),
]

axes_list = []

for title, surf_mesh, stat, hemi, view, subplot_code in views:
    ax = fig.add_subplot(subplot_code, projection="3d")
    axes_list.append(ax)

    plotting.plot_surf_stat_map(
        surf_mesh,
        stat,
        hemi=hemi,
        view=view,
        bg_map=fsavg5.sulc_left if hemi == "left" else fsavg5.sulc_right,
        cmap="viridis",
        colorbar=False,  # we add one shared colorbar below
        vmin=vmin,
        vmax=vmax,
        axes=ax,
        title=title,
        darkness=None,   # keep fairly light background
    )

# Shared colorbar
sm = plt.cm.ScalarMappable(
    cmap="viridis",
    norm=plt.Normalize(vmin=vmin, vmax=vmax)
)
sm.set_array([])

cbar = fig.colorbar(sm, ax=axes_list, shrink=0.6, pad=0.05)
cbar.set_label("Region SHAP (APOE axis)", fontsize=11)

plt.tight_layout()
fig.savefig(OUT_PNG, dpi=200)
plt.close(fig)

print(f"✔ Saved fsaverage5 DK68 SHAP multi-panel figure → {OUT_PNG}")





"""
PART 7g — DK68 Region SHAP on FreeSurfer fsaverage (pial + inflated)
====================================================================

Produces two publication-quality multi-panel figures:

  • DK68_SHAP_fsavg_pial_4panel.png / .pdf
      - fsaverage pial surface, DK68 patches colored by region SHAP

  • DK68_SHAP_fsavg_inflated_4panel.png / .pdf
      - fsaverage inflated surface, DK68 patches colored by region SHAP

Assumes:
  • FreeSurfer installed and FREESURFER_HOME set, OR default at /home/apps/freesurfer
  • DK68 regions in region_global_SHAP.csv use names like lh_bankssts, rh_superiortemporal, etc.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilearn import plotting
from nilearn import surface as nlsurf
from nibabel.freesurfer.io import read_annot, read_morph_data

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
# FreeSurfer root: use FREESURFER_HOME if available, else default
FS_HOME = os.environ.get("FREESURFER_HOME", "/home/apps/freesurfer")

FS_SUBJECTS = os.path.join(FS_HOME, "subjects")
FS_FSAVERAGE = os.path.join(FS_SUBJECTS, "fsaverage")

# Surfaces
LH_PIAL      = os.path.join(FS_FSAVERAGE, "surf", "lh.pial")
RH_PIAL      = os.path.join(FS_FSAVERAGE, "surf", "rh.pial")
LH_INFLATED  = os.path.join(FS_FSAVERAGE, "surf", "lh.inflated")
RH_INFLATED  = os.path.join(FS_FSAVERAGE, "surf", "rh.inflated")

# Sulc (for background shading)
LH_SULC      = os.path.join(FS_FSAVERAGE, "surf", "lh.sulc")
RH_SULC      = os.path.join(FS_FSAVERAGE, "surf", "rh.sulc")

# DK68 (Desikan-Killiany) annotation files
LH_APARC     = os.path.join(FS_FSAVERAGE, "label", "lh.aparc.annot")
RH_APARC     = os.path.join(FS_FSAVERAGE, "label", "rh.aparc.annot")

# Project root
ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation"
CROOT = os.path.join(ROOT, "columns4gaudi111825")

# Region SHAP from unified Part 7
SHAP_FILE = os.path.join(
    CROOT,
    "results_k25_APOE",
    "region_global_SHAP.csv"
)

# Output directory (utilities to keep figures handy)
OUTDIR = os.path.join(CROOT, "utilities")
os.makedirs(OUTDIR, exist_ok=True)

OUT_PIAL      = os.path.join(OUTDIR, "DK68_SHAP_fsavg_pial_4panel.png")
OUT_INFLATED  = os.path.join(OUTDIR, "DK68_SHAP_fsavg_inflated_4panel.png")

# Save figures directly in results_k25_APOE
OUTDIR = os.path.join(
    CROOT,
    "results_k25_APOE"
)
os.makedirs(OUTDIR, exist_ok=True)

OUT_PIAL      = os.path.join(OUTDIR, "DK68_SHAP_fsavg_pial_4panel.png")
OUT_INFLATED  = os.path.join(OUTDIR, "DK68_SHAP_fsavg_inflated_4panel.png")


# ------------------------------------------------------------------
# 1. Load region-level SHAP
# ------------------------------------------------------------------
print("🔵 Loading region-level SHAP from:", SHAP_FILE)
shap_df = pd.read_csv(SHAP_FILE, index_col=0)
# Expect index like "lh_bankssts", "rh_superiortemporal", etc.
shap_dict = shap_df["global_SHAP"].to_dict()

print(f"  Loaded SHAP for {len(shap_dict)} regions.")


# ------------------------------------------------------------------
# 2. Load annotations & build vertex-wise SHAP
# ------------------------------------------------------------------
def load_hemi_annot(annot_path, hemi_prefix, shap_lookup):
    """
    annot_path : path to lh.aparc.annot or rh.aparc.annot
    hemi_prefix: "lh" or "rh"
    shap_lookup: dict {region_name -> shap_value}, with keys like lh_bankssts
    """
    labels, ctab, names = read_annot(annot_path)
    # Names come as bytes; decode and normalize
    clean_names = []
    for n in names:
        if isinstance(n, bytes):
            n = n.decode("utf-8")
        clean_names.append(n)

    data = np.zeros_like(labels, dtype=float)

    for idx, name in enumerate(clean_names):
        # Freesurfer names for DK68 are like "bankssts", "superiortemporal".
        # We map them to your region naming convention: lh_bankssts, rh_bankssts, etc.
        base = name.lower().replace("-", "_").replace("__", "_")
        key = f"{hemi_prefix}_{base}"
        if key in shap_lookup:
            data[labels == idx] = shap_lookup[key]

    return data


print("🔵 Loading FreeSurfer DK68 annotations...")
lh_shap = load_hemi_annot(LH_APARC, "lh", shap_dict)
rh_shap = load_hemi_annot(RH_APARC, "rh", shap_dict)

all_vals = np.concatenate([lh_shap, rh_shap])
vmin = np.percentile(all_vals, 5)
vmax = np.percentile(all_vals, 95)

print(f"  Vertex-wise SHAP range (5–95th pct): {vmin:.4f} .. {vmax:.4f}")


# ------------------------------------------------------------------
# 3. Load sulc for background
# ------------------------------------------------------------------
print("🔵 Loading sulcal depth (background shading)...")
lh_sulc = read_morph_data(LH_SULC)
rh_sulc = read_morph_data(RH_SULC)


# ------------------------------------------------------------------
# 4. Multi-panel plotting helper
# ------------------------------------------------------------------
def plot_fsaverage_panel(
    surf_lh, surf_rh,
    data_lh, data_rh,
    sulc_lh, sulc_rh,
    title_suffix,
    out_png
):
    """
    Make a 4-panel figure: LH/RH lateral + medial for given surfaces & data.
    Saves PNG + PDF.
    """
    print(f"🔵 Rendering 4-panel fsaverage ({title_suffix})…")

    fig = plt.figure(figsize=(14, 8), dpi=300)
    # layout: 2 rows, 3 columns; last col for colorbar
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.06], wspace=0.02, hspace=0.05)

    # ---- LH lateral
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    plotting.plot_surf_stat_map(
        surf_mesh=surf_lh,
        stat_map=data_lh,
        hemi="left",
        view="lateral",
        bg_map=sulc_lh,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
        alpha=1.0,
        darkness=None,
        axes=ax1,
        title="Lateral LH"
    )

    # ---- LH medial
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    plotting.plot_surf_stat_map(
        surf_mesh=surf_lh,
        stat_map=data_lh,
        hemi="left",
        view="medial",
        bg_map=sulc_lh,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
        alpha=1.0,
        darkness=None,
        axes=ax2,
        title="Medial LH"
    )

    # ---- RH lateral
    ax3 = fig.add_subplot(gs[1, 0], projection="3d")
    plotting.plot_surf_stat_map(
        surf_mesh=surf_rh,
        stat_map=data_rh,
        hemi="right",
        view="lateral",
        bg_map=sulc_rh,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
        alpha=1.0,
        darkness=None,
        axes=ax3,
        title="Lateral RH"
    )

    # ---- RH medial
    ax4 = fig.add_subplot(gs[1, 1], projection="3d")
    plotting.plot_surf_stat_map(
        surf_mesh=surf_rh,
        stat_map=data_rh,
        hemi="right",
        view="medial",
        bg_map=sulc_rh,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
        alpha=1.0,
        darkness=None,
        axes=ax4,
        title="Medial RH"
    )

    # ---- Shared colorbar
    cax = fig.add_subplot(gs[:, 2])
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_clim(vmin, vmax)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Region SHAP (APOE axis)", fontsize=10)

    fig.suptitle(f"DK68 Region SHAP on fsaverage — {title_suffix}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.96, 0.95])

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.replace(".png", ".pdf"), dpi=300)
    plt.close(fig)

    print(f"✔ Saved: {out_png}")


# ------------------------------------------------------------------
# 5. PIAL surface figure
# ------------------------------------------------------------------
print("🔵 Plotting pial surface figure…")
plot_fsaverage_panel(
    surf_lh=LH_PIAL,
    surf_rh=RH_PIAL,
    data_lh=lh_shap,
    data_rh=rh_shap,
    sulc_lh=lh_sulc,
    sulc_rh=rh_sulc,
    title_suffix="Pial surface",
    out_png=OUT_PIAL
)

# ------------------------------------------------------------------
# 6. INFLATED surface figure
# ------------------------------------------------------------------
print("🔵 Plotting inflated surface figure…")
plot_fsaverage_panel(
    surf_lh=LH_INFLATED,
    surf_rh=RH_INFLATED,
    data_lh=lh_shap,
    data_rh=rh_shap,
    sulc_lh=lh_sulc,
    sulc_rh=rh_sulc,
    title_suffix="Inflated surface",
    out_png=OUT_INFLATED
)

print("\n🎉 Finished PART 7g — fsaverage pial + inflated DK68 SHAP figures.")
print("   Pial:     ", OUT_PIAL)
print("   Inflated: ", OUT_INFLATED)

