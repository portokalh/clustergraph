#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:33:20 2025

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build SHARED kNN Graphs for MD & QSM (multi-K, with QA)
=======================================================

Supports MULTIPLE k values:
    K_LIST = [10, 20, 30, 50]

For each K, outputs are stored in:
    graphs_knn/k{K}/
        md_shared_knn_k{K}_corr_euclid_wass.pt
        qsm_shared_knn_k{K}_corr_euclid_wass.pt

Also generates a QA CSV:
    graph_QA_summary.csv

with per-(K, modality, subject) statistics:
    nodes, edges, avg_deg, min_deg, max_deg,
    mean/std(corr), mean/std(euclid), mean/std(wass)
"""

import os
import glob
import pandas as pd
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

# ==========================================================
# CONFIGURATION
# ==========================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
GAUDI_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
COLUMNS_ROOT = SCRIPT_DIR
BASE_NODES_DIR = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation/processed_graph_data_110325"

MD_NODES_DIR  = os.path.join(BASE_NODES_DIR, "md_nodes")
QSM_NODES_DIR = os.path.join(BASE_NODES_DIR, "qsm_nodes")

# --- MULTIPLE K VALUES ---
K_LIST = [4, 6, 8, 10, 12, 20, 25, 30, 50]

QA_CSV = os.path.join(COLUMNS_ROOT, "graph_QA_summary.csv")

print("SCRIPT_DIR   =", SCRIPT_DIR)
print("GAUDI_ROOT   =", GAUDI_ROOT)
print("COLUMNS_ROOT =", COLUMNS_ROOT)
print("MD_NODES_DIR =", MD_NODES_DIR)
print("QSM_NODES_DIR =", QSM_NODES_DIR)

# ==========================================================
# Utilities
# ==========================================================

def extract_subject_id(filename: str) -> str:
    """First 5 chars of filename = MRI_Exam ID, e.g. '00775'."""
    return os.path.basename(filename)[:5]


def print_graph_properties(g: Data, label: str):
    deg = torch.bincount(g.edge_index[0], minlength=g.num_nodes)
    print(f"\n----- GRAPH PROPERTIES: {label} -----")
    print(f"Subject ID: {getattr(g, 'subject_id', 'NONE')}")
    print(f"Nodes:      {g.num_nodes}")
    print(f"Edges:      {g.num_edges}")
    print(f"Node feats: {tuple(g.x.shape)}")
    if g.edge_attr is not None:
        print(f"Edge attrs: {tuple(g.edge_attr.shape)}")
        print(f"  Edge attr means: {g.edge_attr.mean(dim=0).numpy()}")
        print(f"  Edge attr stds : {g.edge_attr.std(dim=0).numpy()}")
    print(f"Avg degree: {deg.float().mean():.2f} (min={deg.min()}, max={deg.max()})")
    print("-----------------------------------------")


def load_nodes(nodes_file: str) -> torch.Tensor:
    df = pd.read_csv(nodes_file)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    if "node_idx" not in df.columns:
        raise ValueError(f"'node_idx' missing in {nodes_file}")
    df = df.sort_values("node_idx").reset_index(drop=True)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if not feat_cols:
        raise ValueError(f"No feat_* columns in {nodes_file}")
    return torch.tensor(df[feat_cols].values, dtype=torch.float32)


def discover_subjects(md_dir: str, qsm_dir: str):
    md_files  = glob.glob(os.path.join(md_dir, "*_md_nodes.csv"))
    qsm_files = glob.glob(os.path.join(qsm_dir, "*_qsm_nodes.csv"))

    md_ids  = {os.path.basename(f).replace("_md_nodes.csv", "") for f in md_files}
    qsm_ids = {os.path.basename(f).replace("_qsm_nodes.csv", "") for f in qsm_files}

    common = sorted(md_ids & qsm_ids)
    print(f"Found {len(common)} subjects with BOTH MD and QSM")
    return common


def build_shared_knn_edges(x_md, x_qsm, k):
    """
    Build SHARED kNN edges by concatenating MD & QSM node features
    and running kNN in the joint space.
    """
    if x_md.shape[1] == x_qsm.shape[1] + 1:
        # Known case: MD has one extra column
        x_md = x_md[:, :-1]
    elif x_md.shape[1] != x_qsm.shape[1]:
        raise ValueError(f"MD/QSM feature mismatch: {x_md.shape} vs {x_qsm.shape}")

    X = torch.cat([x_md, x_qsm], dim=1).cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, indices = nbrs.kneighbors(X)

    src, dst = [], []
    N = X.shape[0]
    for i in range(N):
        for j in indices[i, 1:]:
            src.append(i); dst.append(j)
            src.append(j); dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index, x_md  # x_md is the "fixed" MD feature matrix


def compute_edge_attr(x, edge_index):
    """
    Compute multi-metric edge attributes:
        0: correlation-like similarity
        1: Euclidean distance
        2: Wasserstein-1-like distance on sorted features

    Then z-score edge_attr PER GRAPH, so GAUDI sees ~N(0,1) scales.
    """
    src, dst = edge_index[0], edge_index[1]
    x_src, x_dst = x[src], x[dst]

    # Euclidean
    euclid = torch.norm(x_src - x_dst, dim=1)

    # Correlation-like (per edge, across feature vectors)
    xc = x_src - x_src.mean(dim=1, keepdim=True)
    yc = x_dst - x_dst.mean(dim=1, keepdim=True)
    corr = (xc * yc).sum(dim=1) / (
        torch.sqrt((xc**2).sum(dim=1) * (yc**2).sum(dim=1)) + 1e-6
    )

    # Wasserstein-1 with sorted features
    xs, _ = torch.sort(x_src, dim=1)
    ys, _ = torch.sort(x_dst, dim=1)
    wass = torch.abs(xs - ys).mean(dim=1)

    edge_attr = torch.stack([corr, euclid, wass], dim=1)

    # Per-graph z-scoring of edge_attr
    mu = edge_attr.mean(0, keepdim=True)
    sd = edge_attr.std(0, keepdim=True)
    sd[sd == 0] = 1.0
    edge_attr = (edge_attr - mu) / sd
    return edge_attr


# ==========================================================
# MAIN MULTI-K LOOP (with QA)
# ==========================================================
def main():
    subjects = discover_subjects(MD_NODES_DIR, QSM_NODES_DIR)

    qa_rows = []

    for K in K_LIST:
        print("\n" + "=" * 70)
        print(f"🔵 Building graphs for K = {K}")
        print("=" * 70)

        out_dir = os.path.join(COLUMNS_ROOT, "graphs_knn", f"k{K}")
        os.makedirs(out_dir, exist_ok=True)

        md_out = []
        qsm_out = []

        for sid in subjects:
            md_file  = os.path.join(MD_NODES_DIR,  sid + "_md_nodes.csv")
            qsm_file = os.path.join(QSM_NODES_DIR, sid + "_qsm_nodes.csv")

            print(f"\n====== SUBJECT {sid} (K={K}) ======")
            x_md  = load_nodes(md_file)
            x_qsm = load_nodes(qsm_file)
            subject_id = extract_subject_id(md_file)

            edge_index, x_md_fixed = build_shared_knn_edges(x_md, x_qsm, k=K)

            edge_attr_md  = compute_edge_attr(x_md_fixed, edge_index)
            edge_attr_qsm = compute_edge_attr(x_qsm,      edge_index)

            g_md = Data(x=x_md_fixed, edge_index=edge_index, edge_attr=edge_attr_md)
            g_md.subject_id = subject_id

            g_qsm = Data(x=x_qsm, edge_index=edge_index, edge_attr=edge_attr_qsm)
            g_qsm.subject_id = subject_id

            print_graph_properties(g_md,  f"MD {sid}, K={K}")
            print_graph_properties(g_qsm, f"QSM {sid}, K={K}")

            # --- QA rows for MD ---
            deg_md = torch.bincount(g_md.edge_index[0], minlength=g_md.num_nodes)
            qa_rows.append({
                "K": K,
                "Modality": "MD",
                "Subject": subject_id,
                "Nodes": g_md.num_nodes,
                "Edges": g_md.num_edges,
                "AvgDegree": float(deg_md.float().mean()),
                "MinDegree": int(deg_md.min()),
                "MaxDegree": int(deg_md.max()),
                "CorrMean":  float(g_md.edge_attr[:,0].mean()),
                "CorrStd":   float(g_md.edge_attr[:,0].std()),
                "EuclidMean":float(g_md.edge_attr[:,1].mean()),
                "EuclidStd": float(g_md.edge_attr[:,1].std()),
                "WassMean":  float(g_md.edge_attr[:,2].mean()),
                "WassStd":   float(g_md.edge_attr[:,2].std()),
            })

            # --- QA rows for QSM ---
            deg_qsm = torch.bincount(g_qsm.edge_index[0], minlength=g_qsm.num_nodes)
            qa_rows.append({
                "K": K,
                "Modality": "QSM",
                "Subject": subject_id,
                "Nodes": g_qsm.num_nodes,
                "Edges": g_qsm.num_edges,
                "AvgDegree": float(deg_qsm.float().mean()),
                "MinDegree": int(deg_qsm.min()),
                "MaxDegree": int(deg_qsm.max()),
                "CorrMean":  float(g_qsm.edge_attr[:,0].mean()),
                "CorrStd":   float(g_qsm.edge_attr[:,0].std()),
                "EuclidMean":float(g_qsm.edge_attr[:,1].mean()),
                "EuclidStd": float(g_qsm.edge_attr[:,1].std()),
                "WassMean":  float(g_qsm.edge_attr[:,2].mean()),
                "WassStd":   float(g_qsm.edge_attr[:,2].std()),
            })

            md_out.append(g_md)
            qsm_out.append(g_qsm)

        # SAVE graphs
        md_path  = os.path.join(out_dir, f"md_shared_knn_k{K}_corr_euclid_wass.pt")
        qsm_path = os.path.join(out_dir, f"qsm_shared_knn_k{K}_corr_euclid_wass.pt")

        torch.save(md_out, md_path)
        torch.save(qsm_out, qsm_path)

        print(f"\n💾 Saved MD  graphs → {md_path}")
        print(f"💾 Saved QSM graphs → {qsm_path}")

    # SAVE QA summary (overwrite each run so it's clean)
    qa_df = pd.DataFrame(qa_rows)
    qa_df.to_csv(QA_CSV, index=False)
    print(f"\n📊 Graph QA summary saved → {QA_CSV}")
    print("\n🎯 DONE — all K in K_LIST processed.")


if __name__ == "__main__":
    main()
