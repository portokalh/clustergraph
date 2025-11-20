#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 11:39:16 2025

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build clean 21-depth MD + QSM profiles
(drop MD extra column, z-score across nodes)

Outputs:
   profiles_depth21/md_profiles.npy
   profiles_depth21/qsm_profiles.npy
"""

import os
import numpy as np
import pandas as pd

ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation"
CROOT = os.path.join(ROOT, "columns4gaudi111825", "columna-analyses111925")

# Node CSVs
MD_NODES_DIR  = os.path.join(ROOT, "processed_graph_data_110325", "md_nodes")
QSM_NODES_DIR = os.path.join(ROOT, "processed_graph_data_110325", "qsm_nodes")

# Metadata (alignment by MRI_Exam)
MDATA = os.path.join(ROOT, "processed_graph_data", "metadata_with_PCs.xlsx")

# Output folder
OUTDIR = os.path.join(ROOT, "columns4gaudi111825", "profiles_depth21")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------
# Load subject ordering
# ---------------------------------------------------------
meta = pd.read_excel(MDATA)
meta["MRI_Exam"] = meta["MRI_Exam"].astype(str).str.zfill(5)
subject_ids = meta["MRI_Exam"].tolist()

print(f"Metadata has {len(subject_ids)} subjects.")


# ---------------------------------------------------------
# Load depth profiles per subject
# ---------------------------------------------------------
def load_md_profile(csv_path):
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]

    arr = df[feat_cols].values

    # Drop last column if 22 features (extra thickness column)
    if arr.shape[1] == 22:
        arr = arr[:, :21]

    if arr.shape[1] != 21:
        raise ValueError(f"MD feature count != 21 after trimming: {csv_path}")

    return arr


def load_qsm_profile(csv_path):
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]

    arr = df[feat_cols].values

    # QSM should have exactly 21 features
    if arr.shape[1] != 21:
        raise ValueError(f"QSM feature count != 21: {csv_path}")

    return arr


md_all = []
qsm_all = []

missing = []

for sid in subject_ids:
    md_file  = os.path.join(MD_NODES_DIR,  f"{sid}_md_nodes.csv")
    qsm_file = os.path.join(QSM_NODES_DIR, f"{sid}_qsm_nodes.csv")

    if not os.path.exists(md_file) or not os.path.exists(qsm_file):
        missing.append(sid)
        continue

    md_prof  = load_md_profile(md_file)
    qsm_prof = load_qsm_profile(qsm_file)

    md_all.append(md_prof)
    qsm_all.append(qsm_prof)

# Stack across subjects
md_all  = np.vstack(md_all)
qsm_all = np.vstack(qsm_all)

print("Raw shapes before z-scoring:")
print("MD  profiles:", md_all.shape)
print("QSM profiles:", qsm_all.shape)


# ---------------------------------------------------------
# Z-score per depth index (columnwise)
# ---------------------------------------------------------
def zscore(arr):
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std == 0] = 1
    return (arr - mean) / std

md_z  = zscore(md_all)
qsm_z = zscore(qsm_all)

# ---------------------------------------------------------
# Save final profiles
# ---------------------------------------------------------
np.save(os.path.join(OUTDIR, "md_profiles.npy"),  md_z)
np.save(os.path.join(OUTDIR, "qsm_profiles.npy"), qsm_z)

print("\n✔ Z-scored profiles saved:")
print("  ", os.path.join(OUTDIR, "md_profiles.npy"))
print("  ", os.path.join(OUTDIR, "qsm_profiles.npy"))

if missing:
    print("\n⚠ Missing subjects:")
    print(missing)
