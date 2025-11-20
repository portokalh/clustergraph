#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 15:22:26 2025

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIFIED PART 7 (Covariate-Adjusted) — Interpretable GAUDI (K=25) with APOE Axis

Adjusts the APOE axis for Age, Sex, and BMI (from metadata_with_PCs.xlsx)
and then repeats all interpretability analyses:

  7-1  Global Region SHAP (via laminar features; covariate-adjusted)
  7a   Depth-wise SHAP (21 MD + 21 QSM)
  7b   Laminar SHAP (Supra / Granular / Infra × MD/QSM)
  7c   Region × Laminar Heatmap
  7d   Subject-Level Latent SHAP (GAUDI latents, adjusted axis)
  7e   Top-10 Regions × Lamina SHAP Panel
  7f   3D Brain Map (DK68 centroids in MNI)
  7g   DK68 SHAP on fsaverage5 (nilearn) + fsaverage pial/inflated (FreeSurfer)
"""

# ============= IMPORTS ======================================
import os
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import Ridge, Lasso, LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# For 3D centroid scatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# For fsaverage5 surfaces
from nilearn import datasets, surface, plotting
from scipy.spatial import cKDTree

# For FreeSurfer fsaverage pial/inflated
from nibabel.freesurfer.io import read_annot, read_morph_data

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

# Metadata + node dirs to reconstruct subject order for covariates
MDATA = os.path.join(ROOT, "processed_graph_data", "metadata_with_PCs.xlsx")
MD_NODES_DIR  = os.path.join(ROOT, "processed_graph_data_110325", "md_nodes")
QSM_NODES_DIR = os.path.join(ROOT, "processed_graph_data_110325", "qsm_nodes")

# Main results directory for covariate-adjusted axis
RESULTS_DIR = os.path.join(
    ROOT, "columns4gaudi111825", f"results_k{K}_{AXIS_NAME}_adjCov"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Optional: cache latents in this results folder
Z_SUBJ_CACHE = os.path.join(RESULTS_DIR, "Z_subjects.npy")


# ============================================================
# Utility loaders
# ============================================================
def load_subject_latents():
    """
    Load subject-level GAUDI latents.

    If a cached copy exists in RESULTS_DIR, use that.
    Otherwise, load from FALLBACK_LATENT and cache.
    """
    if os.path.exists(Z_SUBJ_CACHE):
        print(f"✔ Using cached subject latents: {Z_SUBJ_CACHE}")
        return np.load(Z_SUBJ_CACHE)

    print("⚠ Cached Z_subjects.npy missing — using fallback latents:")
    print("   ", FALLBACK_LATENT)
    Z = np.load(FALLBACK_LATENT)
    np.save(Z_SUBJ_CACHE, Z)
    print(f"✔ Saved subject latents → {Z_SUBJ_CACHE}")
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


def detect_column(candidates, cols, what):
    """Small helper to detect a column name from a list of candidates."""
    for c in candidates:
        if c in cols:
            print(f"  ✔ Using '{c}' as {what}")
            return c
    raise ValueError(f"❌ Could not find {what} in columns: {cols}")


def build_covariates_for_profiles():
    """
    Reconstruct the exact subject order used for md_profiles/qsm_profiles
    by looping metadata MRI_Exam and checking for existing node CSVs.

    Returns:
        cov_df: DataFrame with columns [Age, Sex, BMI] in profile subject order
    """
    print("\n===============================")
    print(" BUILDING COVARIATES (Age/Sex/BMI)")
    print(" IN THE SAME ORDER AS DEPTH PROFILES")
    print("===============================")

    meta = pd.read_excel(MDATA)
    meta["MRI_Exam5"] = meta["MRI_Exam"].astype(str).str.zfill(5)

    cols = list(meta.columns)

    age_col = detect_column(["Age", "age", "Age_at_MRI", "Age_at_scan"], cols, "Age")
    sex_col = detect_column(["Sex", "SEX", "sex"], cols, "Sex")
    bmi_col = detect_column(["BMI", "bmi", "BMI_bl", "BMI_scan"], cols, "BMI")

    included_ids = []
    ages = []
    sexes = []
    bmis = []

    for _, row in meta.iterrows():
        sid = row["MRI_Exam5"]
        md_file  = os.path.join(MD_NODES_DIR,  f"{sid}_md_nodes.csv")
        qsm_file = os.path.join(QSM_NODES_DIR, f"{sid}_qsm_nodes.csv")

        if not (os.path.exists(md_file) and os.path.exists(qsm_file)):
            continue  # subject missing nodes → was also skipped in profiles builder

        included_ids.append(sid)
        ages.append(row[age_col])
        sexes.append(row[sex_col])
        bmis.append(row[bmi_col])

    cov_df = pd.DataFrame({
        "MRI_Exam5": included_ids,
        "Age": np.asarray(ages, dtype=float),
        "Sex_raw": sexes,
        "BMI": np.asarray(bmis, dtype=float),
    })

    # Encode Sex numerically if needed
    if cov_df["Sex_raw"].dtype == object:
        unique_sex = sorted(cov_df["Sex_raw"].dropna().unique())
        print("  Sex unique values:", unique_sex)
        if set(unique_sex) <= {"M", "F", "m", "f"}:
            cov_df["Sex"] = cov_df["Sex_raw"].str.upper().map({"M": 0.0, "F": 1.0})
        else:
            # fallback: treat Sex_raw as a binary factor 0/1 via factorization
            codes, _ = pd.factorize(cov_df["Sex_raw"])
            cov_df["Sex"] = codes.astype(float)
    else:
        cov_df["Sex"] = cov_df["Sex_raw"].astype(float)

    cov_df.drop(columns=["Sex_raw"], inplace=True)

    print(f"✔ Covariates assembled for {cov_df.shape[0]} subjects.")
    return cov_df


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

# Derive node/subject counts and expand region labels
N_nodes = md_profiles.shape[0]
if N_nodes % N_regions != 0:
    raise ValueError(f"N_nodes={N_nodes} not divisible by N_regions={N_regions}")

N_subj_profiles = N_nodes // N_regions
print(f"✔ Profiles correspond to {N_subj_profiles} subjects × {N_regions} regions")

# Expand region labels so each row in md_profiles has a region label
region_labels_expanded = np.tile(region_labels, N_subj_profiles)
assert len(region_labels_expanded) == N_nodes


# ------------------------------------------------------------
# Build covariates in the same order as profiles & build adjusted axis
# ------------------------------------------------------------
cov_df = build_covariates_for_profiles()

if cov_df.shape[0] != N_subj_profiles:
    print(f"⚠ WARNING: covariate subjects ({cov_df.shape[0]}) "
          f"!= profile subjects ({N_subj_profiles}). "
          "Check metadata / node dirs consistency.")

# Use only first N_subj_profiles rows of latents (to match profiles & covariates)
if N_subj_latent < N_subj_profiles:
    raise ValueError(
        f"Latents ({N_subj_latent}) < profiles ({N_subj_profiles}); "
        "cannot align for covariate adjustment."
    )

if cov_df.shape[0] < N_subj_profiles:
    raise ValueError(
        f"Covariates ({cov_df.shape[0]}) < profiles ({N_subj_profiles}); "
        "cannot align cleanly."
    )

Z_subj_use = Z_subj[:N_subj_profiles, :]
print("✔ Using first N_subj_profiles latents:", Z_subj_use.shape)

# Subject-level raw APOE axis scores
subj_scores_raw = Z_subj_use @ w_axis        # shape = (N_subj_profiles,)

# Standardize covariates (Age, Sex, BMI)
age = cov_df["Age"].values.astype(float)
sex = cov_df["Sex"].values.astype(float)
bmi = cov_df["BMI"].values.astype(float)

def zscore(v):
    m = np.nanmean(v)
    s = np.nanstd(v)
    if s == 0:
        s = 1.0
    return (v - m) / s

X_cov = np.column_stack([
    zscore(age),
    zscore(sex),
    zscore(bmi),
])

# Fit linear regression: subj_scores_raw ~ Age + Sex + BMI
print("\n🔵 Regressing APOE axis on Age, Sex, BMI...")
reg = LinearRegression()
reg.fit(X_cov, subj_scores_raw)
y_hat = reg.predict(X_cov)
subj_scores_adj = subj_scores_raw - y_hat

# Save covariates + axis/residuals for record
cov_out = cov_df.copy()
cov_out["APOE_axis_raw"] = subj_scores_raw
cov_out["APOE_axis_adjCov"] = subj_scores_adj
cov_out.to_csv(
    os.path.join(RESULTS_DIR, "covariates_and_APOE_axis_adj.csv"),
    index=False
)
print("✔ Saved covariates_and_APOE_axis_adj.csv")

# Node-level target: repeat adjusted subject score across that subject's 68 regions
node_scores = np.repeat(subj_scores_adj, N_regions)   # shape = (N_nodes,)
assert node_scores.shape[0] == N_nodes
print("✔ node_scores (covariate-adjusted) shape:", node_scores.shape)


# ============================================================
# PART 7b / 7-1 — LAMINAR SHAP + GLOBAL REGION SHAP (adjCov)
# ============================================================
print("\n===============================")
print(" PART 7b / 7-1 — Laminar SHAP (adjCov)")
print("      + Global Region SHAP     ")
print("===============================")

laminar_X = laminar(md_profiles, qsm_profiles)   # (N_nodes, 6)
lam_feat = [
    "MD_supra", "MD_granular", "MD_infra",
    "QSM_supra", "QSM_granular", "QSM_infra"
]

# Lasso on laminar features → adjusted node_scores
lasso_lam = Lasso(alpha=0.001)
lasso_lam.fit(laminar_X, node_scores)

expl_lam = shap.Explainer(lasso_lam, laminar_X, feature_names=lam_feat)
sh_lam   = expl_lam(laminar_X)
abs_sh_lam = np.abs(sh_lam.values)   # (N_nodes, 6)

# ---------- Save node-level laminar SHAP ----------
lam_node_df = pd.DataFrame(abs_sh_lam, columns=lam_feat)
lam_node_df["node_score_adjCov"] = node_scores
lam_node_df["region"] = region_labels_expanded
lam_node_df.to_csv(
    os.path.join(RESULTS_DIR, "laminar_SHAP_nodevalues_adjCov.csv"),
    index=False
)
print("✔ Saved laminar_SHAP_nodevalues_adjCov.csv")

# ---------- Global laminar importance ----------
lam_global = abs_sh_lam.mean(axis=0)
lam_global_df = pd.DataFrame({
    "feature": lam_feat,
    "global_mean_absSHAP": lam_global
})
lam_global_df.to_csv(
    os.path.join(RESULTS_DIR, "laminar_SHAP_global_adjCov.csv"),
    index=False
)
print("✔ Saved laminar_SHAP_global_adjCov.csv")

# ---------- Region × lamina SHAP (mean |SHAP| per region/lamina) ----------
lam_region_df = (
    lam_node_df
    .groupby("region")[lam_feat]
    .mean()
)
lam_region_df.to_csv(
    os.path.join(RESULTS_DIR, "laminar_SHAP_byregion_adjCov.csv")
)
print("✔ Saved laminar_SHAP_byregion_adjCov.csv")

# ---------- Global REGION SHAP (one number per region) ----------
region_global = lam_region_df.sum(axis=1)
region_global = region_global.sort_values(ascending=False)

region_global.to_csv(
    os.path.join(RESULTS_DIR, "region_global_SHAP_adjCov.csv"),
    header=["global_SHAP"]
)
print("✔ Saved region_global_SHAP_adjCov.csv")

# Barplot of top regions
topN = 30
top_reg = region_global.head(topN)

plt.figure(figsize=(10, 12))
plt.barh(top_reg.index[::-1], top_reg.values[::-1])
plt.title(f"Top Regions Driving {AXIS_NAME} Axis (K={K}, adj Age/Sex/BMI)")
plt.xlabel("Global SHAP (sum over laminae, adjCov)")
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "region_global_SHAP_barplot_adjCov.png"),
    dpi=300
)
plt.close()
print("✔ Saved region_global_SHAP_barplot_adjCov.png")


# ============================================================
# PART 7a — DEPTH-WISE SHAP (42 features, adjCov)
# ============================================================
print("\n===============================")
print(" PART 7a — Depth-wise SHAP (adjCov)")
print("===============================")

depth_feature_names = [f"MD_d{i}" for i in range(21)] + [f"QSM_d{i}" for i in range(21)]
X_depth = np.hstack([md_profiles, qsm_profiles])   # (N_nodes, 42)

lasso_depth = Lasso(alpha=0.0005)
lasso_depth.fit(X_depth, node_scores)

expl_depth = shap.Explainer(lasso_depth, X_depth, feature_names=depth_feature_names)
sh_depth   = expl_depth(X_depth)
abs_sh_depth = np.abs(sh_depth.values)

pd.DataFrame(abs_sh_depth, columns=depth_feature_names).to_csv(
    os.path.join(RESULTS_DIR, "depth_SHAP_nodevalues_adjCov.csv"),
    index=False
)
print("✔ Saved depth_SHAP_nodevalues_adjCov.csv")

depth_global = abs_sh_depth.mean(axis=0)
depth_global_df = pd.DataFrame({
    "feature": depth_feature_names,
    "mean_abs_SHAP": depth_global
})
depth_global_df.to_csv(
    os.path.join(RESULTS_DIR, "depth_SHAP_global_adjCov.csv"),
    index=False
)
print("✔ Saved depth_SHAP_global_adjCov.csv")

idx_top = np.argsort(depth_global)[::-1][:20]
plt.figure(figsize=(10, 8))
plt.barh(
    [depth_feature_names[i] for i in idx_top][::-1],
    depth_global[idx_top][::-1]
)
plt.title(f"Top 20 Depth Features ({AXIS_NAME} axis, adjCov)")
plt.xlabel("Mean |SHAP|")
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "depth_SHAP_top20_adjCov.png"),
    dpi=300
)
plt.close()
print("✔ Saved depth_SHAP_top20_adjCov.png")


# ============================================================
# PART 7c — REGION × LAMINA HEATMAP (adjCov)
# ============================================================
print("\n===============================")
print(" PART 7c — Region × Lamina Heatmap (adjCov)")
print("===============================")

df_heat = lam_region_df.copy()   # (68 regions × 6 lamina-features)

cmap = sns.color_palette("viridis", as_cmap=True)
g = sns.clustermap(
    df_heat,
    cmap=cmap,
    linewidths=0.5,
    figsize=(12, 14),
    cbar_kws={"label": "Mean |SHAP| (adjCov)"}
)

plt.title(f"Region × Lamina SHAP — {AXIS_NAME} axis (adj Age/Sex/BMI)", pad=80)
heat_png = os.path.join(RESULTS_DIR, "laminar_SHAP_heatmap_adjCov.png")
heat_pdf = os.path.join(RESULTS_DIR, "laminar_SHAP_heatmap_adjCov.pdf")
g.fig.savefig(heat_png, dpi=300, bbox_inches='tight')
g.fig.savefig(heat_pdf, dpi=300, bbox_inches='tight')
plt.close()
print("✔ Saved laminar_SHAP_heatmap_adjCov.png / .pdf")


# ============================================================
# PART 7d — Latent SHAP (Subject-level, adjCov axis)
# ============================================================
print("\n===============================")
print(" PART 7d — Latent SHAP (Subject-level, adjCov)")
print("===============================")

LATENT_FILE = FALLBACK_LATENT
AXIS_DIR = os.path.join(
    CROOT,
    f"trait_axes_K{K}_{METRIC}_Joint",
    "axes"
)
AXIS_FILE = os.path.join(AXIS_DIR, f"{AXIS_NAME}_axis.npy")

LATENT_OUTDIR = os.path.join(CROOT, f"shap_latents_K{K}_{AXIS_NAME}_adjCov")
os.makedirs(LATENT_OUTDIR, exist_ok=True)

print("🔵 Loading GAUDI latents (subject-level)…")
Z_lat = np.load(LATENT_FILE)   # (N_subj_latent, latent_dim)
Nsubj_lat, D_lat = Z_lat.shape
print(f"✔ Latent matrix shape = {Z_lat.shape} (subjects × latent_dim)")

print(f"🔵 Loading axis from {AXIS_FILE}")
axis_vec = np.load(AXIS_FILE)   # (latent_dim,)

# We only have covariates for the first N_subj_profiles subjects
Z_lat_use = Z_lat[:N_subj_profiles, :]
target_lat_raw = Z_lat_use @ axis_vec

# Use same X_cov (Age/Sex/BMI) as before
print("🔵 Regressing latent axis on Age, Sex, BMI (same subset)…")
reg_lat = LinearRegression()
reg_lat.fit(X_cov, target_lat_raw)
target_lat_adj = target_lat_raw - reg_lat.predict(X_cov)

# Fit Ridge + SHAP on latents with adjusted target
print("🔵 Fitting Ridge model + SHAP on latents (adjCov)…")
model_lat = Ridge(alpha=1.0)
model_lat.fit(Z_lat_use, target_lat_adj)

explainer_lat = shap.Explainer(model_lat, Z_lat_use)
shap_values_lat = explainer_lat(Z_lat_use)     # (N_subj_profiles × D_lat)
sv_lat = shap_values_lat.values

subj_ids = [f"subj_{i:03d}" for i in range(N_subj_profiles)]
latent_cols = [f"latent_{i+1}" for i in range(D_lat)]

df_subj_lat = pd.DataFrame(sv_lat, index=subj_ids, columns=latent_cols)
df_subj_lat.to_csv(
    os.path.join(LATENT_OUTDIR, "shap_latents_per_subject_adjCov.csv")
)
print("✔ Saved shap_latents_per_subject_adjCov.csv")

global_imp_lat = np.mean(np.abs(sv_lat), axis=0)
df_global_lat = pd.DataFrame({
    "latent_dim": latent_cols,
    "global_mean_absSHAP": global_imp_lat
})
df_global_lat.to_csv(
    os.path.join(LATENT_OUTDIR, "shap_latents_global_adjCov.csv"),
    index=False
)
print("✔ Saved shap_latents_global_adjCov.csv")

plt.figure(figsize=(14, 8))
cmap2 = sns.color_palette("viridis", as_cmap=True)
sns.heatmap(
    df_subj_lat,
    cmap=cmap2,
    cbar_kws={"label": "|SHAP| (adjCov)"},
    xticklabels=True,
    yticklabels=False
)
plt.title(f"Subject-Level Latent SHAP (K={K}, {METRIC}, axis={AXIS_NAME}, adjCov)")
plt.tight_layout()
plt.savefig(
    os.path.join(LATENT_OUTDIR, "shap_latent_heatmap_adjCov.png"),
    dpi=300
)
plt.savefig(
    os.path.join(LATENT_OUTDIR, "shap_latent_heatmap_adjCov.pdf"),
    dpi=300
)
plt.close()
print("✔ Saved latent SHAP heatmap (adjCov, PNG/PDF)")

print("\n🎉 FINISHED UNIFIED PART 7 (adjCov core)")
print(f"Depth/laminar/region results → {RESULTS_DIR}")
print(f"Latent SHAP results         → {LATENT_OUTDIR}")


# ============================================================
# PART 7e — Top-10 Regions × Lamina SHAP Panel (adjCov)
# ============================================================
print("\n============================================")
print(" PART 7e — Top 10 Regions × Lamina Panel (adjCov)")
print("============================================")

lam_region_path = os.path.join(RESULTS_DIR, "laminar_SHAP_byregion_adjCov.csv")
lam_region = pd.read_csv(lam_region_path, index_col=0)

lam_region["global_SHAP"] = lam_region.sum(axis=1)
top10 = lam_region.sort_values("global_SHAP", ascending=False).iloc[:10]
top10_lam = top10.drop(columns=["global_SHAP"])

plt.figure(figsize=(8, 6))
sns.heatmap(
    top10_lam,
    cmap="viridis",
    linewidths=0.5,
    annot=True,
    fmt=".3f",
    cbar_kws={"label": "Mean |SHAP| (adjCov)"},
)
plt.title(f"Top-10 Regions × Lamina SHAP (Axis = {AXIS_NAME}, adjCov)", fontsize=14)
plt.tight_layout()

out_png = os.path.join(RESULTS_DIR, "top10_regions_lamina_heatmap_adjCov.png")
plt.savefig(out_png, dpi=300)
plt.close()
print("✔ Saved:", out_png)


# ============================================================
# PART 7f — 3D Brain Map (Region SHAP using DK68 centroids, adjCov)
# ============================================================
print("\n============================================")
print(" PART 7f — 3D Brain Map (DK68 centroids, adjCov)")
print("============================================")

region_global_path = os.path.join(RESULTS_DIR, "region_global_SHAP_adjCov.csv")
reg_shap = pd.read_csv(region_global_path, index_col=0)

coord_path = os.path.join(
    ROOT, "columns4gaudi111825", "utilities", "dk68_centroids_mni.csv"
)
coords = pd.read_csv(coord_path)

possible_region_cols = ["region", "label", "ROI", "Region", "name"]
region_col = None
for c in possible_region_cols:
    if c in coords.columns:
        region_col = c
        break
if region_col is None:
    raise ValueError(f"❌ No region name column found in centroid file. Columns={coords.columns}")

coords = coords.set_index(region_col)

possible_X = ["X","x","coord_x","mni_x","R","RAS_R"]
possible_Y = ["Y","y","coord_y","mni_y","A","RAS_A"]
possible_Z = ["Z","z","coord_z","mni_z","S","RAS_S"]

def detect_coord(possible, cols):
    for p in possible:
        if p in cols:
            return p
    raise ValueError(f"❌ Could not detect coordinate column among {possible}")

Xcol = detect_coord(possible_X, coords.columns)
Ycol = detect_coord(possible_Y, coords.columns)
Zcol = detect_coord(possible_Z, coords.columns)

print(f"✔ Detected coordinate columns: {Xcol}, {Ycol}, {Zcol}")

coords = coords.loc[reg_shap.index]

X = coords[Xcol].values
Y = coords[Ycol].values
Z = coords[Zcol].values

color = reg_shap["global_SHAP"].values
size = 300 * (color / color.max())

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
cbar.set_label("Global Region SHAP (adjCov)", fontsize=11)

ax.set_title(f"3D Brain Map — Region SHAP ({AXIS_NAME}, adjCov)", fontsize=14)
ax.set_xlabel("X (MNI)")
ax.set_ylabel("Y (MNI)")
ax.set_zlabel("Z (MNI)")

topN = 10
top_regs = reg_shap.sort_values("global_SHAP", ascending=False).head(topN)
for region in top_regs.index:
    xi, yi, zi = coords.loc[region, [Xcol, Ycol, Zcol]]
    ax.text(xi, yi, zi, region, fontsize=7)

plt.tight_layout()
out3d = os.path.join(RESULTS_DIR, "brain3D_region_SHAP_adjCov.png")
plt.savefig(out3d, dpi=350)
plt.close()
print("✔ Saved 3D Brain SHAP plot (adjCov):", out3d)


# ============================================================
# PART 7g — DK68 SHAP on fsaverage5 + FreeSurfer fsaverage (adjCov)
# ============================================================

print("\n============================================")
print(" PART 7g — DK68 SHAP on fsaverage5 (nilearn) (adjCov)")
print("============================================")

SHAP_FILE = region_global_path
CENTROID_FILE = coord_path
OUT_PNG_FS5 = os.path.join(RESULTS_DIR, "DK68_SHAP_fsavg5_4panel_adjCov.png")

print("🔵 Loading region SHAP and DK68 centroids...")
shap_df = pd.read_csv(SHAP_FILE, index_col=0)
shap_series = shap_df["global_SHAP"]

centroids = pd.read_csv(CENTROID_FILE)
centroids["shap"] = centroids["region"].map(shap_series).fillna(0.0)

fsavg5 = datasets.fetch_surf_fsaverage("fsaverage5")
coords_l, faces_l = surface.load_surf_mesh(fsavg5.pial_left)
coords_r, faces_r = surface.load_surf_mesh(fsavg5.pial_right)

lh_cent = centroids[centroids["region"].str.startswith("lh_")].copy()
rh_cent = centroids[centroids["region"].str.startswith("rh_")].copy()

tree_l = cKDTree(lh_cent[["x", "y", "z"]].values)
tree_r = cKDTree(rh_cent[["x", "y", "z"]].values)

dist_l, idx_l = tree_l.query(coords_l)
dist_r, idx_r = tree_r.query(coords_r)

lh_shap_fs = lh_cent["shap"].values[idx_l]
rh_shap_fs = rh_cent["shap"].values[idx_r]

all_vals_fs = np.concatenate([lh_shap_fs, rh_shap_fs])
vmin_fs = np.percentile(all_vals_fs, 5)
vmax_fs = np.percentile(all_vals_fs, 95)

fig = plt.figure(figsize=(16, 8), dpi=200)

views = [
    ("Lateral LH", fsavg5.infl_left,  lh_shap_fs, "left",  "lateral", 221),
    ("Medial LH",  fsavg5.infl_left,  lh_shap_fs, "left",  "medial",  222),
    ("Lateral RH", fsavg5.infl_right, rh_shap_fs, "right", "lateral", 223),
    ("Medial RH",  fsavg5.infl_right, rh_shap_fs, "right", "medial",  224),
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
        colorbar=False,
        vmin=vmin_fs,
        vmax=vmax_fs,
        axes=ax,
        title=title,
        darkness=None,
    )

sm = plt.cm.ScalarMappable(
    cmap="viridis",
    norm=plt.Normalize(vmin=vmin_fs, vmax=vmax_fs)
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes_list, shrink=0.6, pad=0.05)
cbar.set_label("Region SHAP (APOE axis, adjCov)", fontsize=11)

plt.tight_layout()
fig.savefig(OUT_PNG_FS5, dpi=200)
plt.close(fig)
print(f"✔ Saved fsaverage5 DK68 SHAP multi-panel figure (adjCov) → {OUT_PNG_FS5}")


print("\n============================================")
print(" PART 7g (FreeSurfer fsaverage pial + inflated, adjCov)")
print("============================================")

# FreeSurfer root: use FREESURFER_HOME if available, else default
FS_HOME = os.environ.get("FREESURFER_HOME", "/home/apps/freesurfer")
FS_SUBJECTS = os.path.join(FS_HOME, "subjects")
FS_FSAVERAGE = os.path.join(FS_SUBJECTS, "fsaverage")

LH_PIAL      = os.path.join(FS_FSAVERAGE, "surf", "lh.pial")
RH_PIAL      = os.path.join(FS_FSAVERAGE, "surf", "rh.pial")
LH_INFLATED  = os.path.join(FS_FSAVERAGE, "surf", "lh.inflated")
RH_INFLATED  = os.path.join(FS_FSAVERAGE, "surf", "rh.inflated")
LH_SULC      = os.path.join(FS_FSAVERAGE, "surf", "lh.sulc")
RH_SULC      = os.path.join(FS_FSAVERAGE, "surf", "rh.sulc")
LH_APARC     = os.path.join(FS_FSAVERAGE, "label", "lh.aparc.annot")
RH_APARC     = os.path.join(FS_FSAVERAGE, "label", "rh.aparc.annot")

OUT_PIAL      = os.path.join(RESULTS_DIR, "DK68_SHAP_fsavg_pial_4panel_adjCov.png")
OUT_INFLATED  = os.path.join(RESULTS_DIR, "DK68_SHAP_fsavg_inflated_4panel_adjCov.png")

print("🔵 Loading region-level SHAP (adjCov) from:", SHAP_FILE)
shap_df_fs = pd.read_csv(SHAP_FILE, index_col=0)
shap_dict_fs = shap_df_fs["global_SHAP"].to_dict()

def load_hemi_annot(annot_path, hemi_prefix, shap_lookup):
    labels, ctab, names = read_annot(annot_path)
    clean_names = []
    for n in names:
        if isinstance(n, bytes):
            n = n.decode("utf-8")
        clean_names.append(n)

    data = np.zeros_like(labels, dtype=float)
    for idx, name in enumerate(clean_names):
        base = name.lower().replace("-", "_").replace("__", "_")
        key = f"{hemi_prefix}_{base}"
        if key in shap_lookup:
            data[labels == idx] = shap_lookup[key]
    return data

print("🔵 Loading FreeSurfer DK68 annotations (adjCov)…")
lh_shap_fs_fs = load_hemi_annot(LH_APARC, "lh", shap_dict_fs)
rh_shap_fs_fs = load_hemi_annot(RH_APARC, "rh", shap_dict_fs)

all_vals_fs2 = np.concatenate([lh_shap_fs_fs, rh_shap_fs_fs])
vmin2 = np.percentile(all_vals_fs2, 5)
vmax2 = np.percentile(all_vals_fs2, 95)

print("🔵 Loading sulcal depth (background shading)…")
lh_sulc_fs = read_morph_data(LH_SULC)
rh_sulc_fs = read_morph_data(RH_SULC)

def plot_fsaverage_panel(
    surf_lh, surf_rh,
    data_lh, data_rh,
    sulc_lh, sulc_rh,
    title_suffix,
    out_png
):
    print(f"🔵 Rendering 4-panel fsaverage ({title_suffix}, adjCov)…")

    fig = plt.figure(figsize=(14, 8), dpi=300)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.06],
                          wspace=0.02, hspace=0.05)

    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    plotting.plot_surf_stat_map(
        surf_mesh=surf_lh,
        stat_map=data_lh,
        hemi="left",
        view="lateral",
        bg_map=sulc_lh,
        cmap="viridis",
        vmin=vmin2,
        vmax=vmax2,
        colorbar=False,
        alpha=1.0,
        darkness=None,
        axes=ax1,
        title="Lateral LH"
    )

    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    plotting.plot_surf_stat_map(
        surf_mesh=surf_lh,
        stat_map=data_lh,
        hemi="left",
        view="medial",
        bg_map=sulc_lh,
        cmap="viridis",
        vmin=vmin2,
        vmax=vmax2,
        colorbar=False,
        alpha=1.0,
        darkness=None,
        axes=ax2,
        title="Medial LH"
    )

    ax3 = fig.add_subplot(gs[1, 0], projection="3d")
    plotting.plot_surf_stat_map(
        surf_mesh=surf_rh,
        stat_map=data_rh,
        hemi="right",
        view="lateral",
        bg_map=sulc_rh,
        cmap="viridis",
        vmin=vmin2,
        vmax=vmax2,
        colorbar=False,
        alpha=1.0,
        darkness=None,
        axes=ax3,
        title="Lateral RH"
    )

    ax4 = fig.add_subplot(gs[1, 1], projection="3d")
    plotting.plot_surf_stat_map(
        surf_mesh=surf_rh,
        stat_map=data_rh,
        hemi="right",
        view="medial",
        bg_map=sulc_rh,
        cmap="viridis",
        vmin=vmin2,
        vmax=vmax2,
        colorbar=False,
        alpha=1.0,
        darkness=None,
        axes=ax4,
        title="Medial RH"
    )

    cax = fig.add_subplot(gs[:, 2])
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_clim(vmin2, vmax2)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Region SHAP (APOE axis, adjCov)", fontsize=10)

    fig.suptitle(f"DK68 Region SHAP on fsaverage — {title_suffix} (adjCov)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.96, 0.95])

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.replace(".png", ".pdf"), dpi=300)
    plt.close(fig)

    print(f"✔ Saved: {out_png}")

print("🔵 Plotting pial surface figure (adjCov)…")
plot_fsaverage_panel(
    surf_lh=LH_PIAL,
    surf_rh=RH_PIAL,
    data_lh=lh_shap_fs_fs,
    data_rh=rh_shap_fs_fs,
    sulc_lh=lh_sulc_fs,
    sulc_rh=rh_sulc_fs,
    title_suffix="Pial surface",
    out_png=OUT_PIAL
)

print("🔵 Plotting inflated surface figure (adjCov)…")
plot_fsaverage_panel(
    surf_lh=LH_INFLATED,
    surf_rh=RH_INFLATED,
    data_lh=lh_shap_fs_fs,
    data_rh=rh_shap_fs_fs,
    sulc_lh=lh_sulc_fs,
    sulc_rh=rh_sulc_fs,
    title_suffix="Inflated surface",
    out_png=OUT_INFLATED
)

print("\n🎉 Finished FULL COVARIATE-ADJUSTED PART 7 (APOE)")
print("   Core SHAP outputs:      ", RESULTS_DIR)
print("   Latent SHAP outputs:    ", LATENT_OUTDIR)
print("   fsaverage5 spheres:     ", OUT_PNG_FS5)
print("   fsaverage pial/inflated:", OUT_PIAL, ",", OUT_INFLATED)
