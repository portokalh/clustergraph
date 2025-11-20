#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-K GAUDI Aggregation
=========================

Loads:
    distance_metrics_K*.csv
    kmeans_chisq_results_K*.csv
    consensus_cluster_enrichment_K*.csv
    consensus_stability_K*.csv

From:
    /.../GAUDI-implementation/columns4gaudi111825/columna-analyses111925/results_K*

Saves:
    multiK_summary/master_results_raw.csv
    multiK_summary/master_results.csv
    multiK_summary/modelscore_heatmap.png
    multiK_summary/auc_vs_K_APOE.png
    multiK_summary/auc_vs_K_risk.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# ROOT PATHS (YOUR EXACT DIRECTORY STRUCTURE)
# ============================================================
ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation"

ANALYSIS_ROOT = os.path.join(
    ROOT,
    "columns4gaudi111825",
    "columna-analyses111925",
)

MULTIK_DIR = os.path.join(ANALYSIS_ROOT, "multiK_summary")
os.makedirs(MULTIK_DIR, exist_ok=True)

K_LIST = [4, 6, 8, 10, 12, 20, 25, 30, 50]
MODALITIES = ["CORR", "EUCLID", "WASS"]

FACTOR_APOE = "APOE"
FACTOR_RISK = "risk_for_ad"


# ============================================================
# PATH HELPERS FOR EACH K
# ============================================================
def results_dir(K):
    return os.path.join(ANALYSIS_ROOT, f"results_K{K}")


def path_distance(K):
    return os.path.join(results_dir(K), f"distance_metrics_K{K}.csv")


def path_chisq(K):
    return os.path.join(results_dir(K), f"kmeans_chisq_results_K{K}.csv")


def path_consensus_enrich(K):
    return os.path.join(results_dir(K), f"consensus_cluster_enrichment_K{K}.csv")


def path_consensus_stab(K):
    return os.path.join(results_dir(K), f"consensus_stability_K{K}.csv")


# ============================================================
# COLLECT ALL ROWS
# ============================================================
rows = []

for K in K_LIST:
    print(f"\n================== K={K} ==================")

    dm_path = path_distance(K)
    chi_path = path_chisq(K)
    cons_path = path_consensus_enrich(K)
    stab_path = path_consensus_stab(K)

    print("distance file:", dm_path)
    print("chi-sq file:  ", chi_path)
    print("enrich file:  ", cons_path)
    print("stability:    ", stab_path)

    df_dm = pd.read_csv(dm_path) if os.path.exists(dm_path) else None
    df_chi = pd.read_csv(chi_path) if os.path.exists(chi_path) else None
    df_cons = pd.read_csv(cons_path) if os.path.exists(cons_path) else None
    df_stab = pd.read_csv(stab_path) if os.path.exists(stab_path) else None

    # ---------- Per-modality rows ----------
    if df_dm is not None:
        for modality in MODALITIES:
            for factor in [FACTOR_APOE, FACTOR_RISK]:
                sub = df_dm[(df_dm["Distance"] == modality) &
                            (df_dm["Factor"] == factor)]

                AUC = sub["AUC_mean"].iloc[0] if not sub.empty else np.nan
                Sil = sub["Silhouette"].iloc[0] if not sub.empty else np.nan

                V = np.nan
                if df_chi is not None:
                    sub_chi = df_chi[(df_chi["Distance"] == modality) &
                                     (df_chi["Factor"] == factor)]
                    if not sub_chi.empty:
                        V = sub_chi["CramersV"].iloc[0]

                rows.append({
                    "K": K,
                    "Modality": modality,
                    "Factor": factor,
                    "AUC": AUC,
                    "Silhouette": Sil,
                    "CramersV": V,
                    "Stability": np.nan,
                    "Source": "kmeans"
                })

    # ---------- Consensus rows ----------
    if df_cons is not None:
        for factor in df_cons["Factor"].unique():
            sub_c = df_cons[df_cons["Factor"] == factor]
            if sub_c.empty:
                continue
            V = sub_c["CramersV"].iloc[0]
            stab_val = df_stab["Stability"].mean() if df_stab is not None else np.nan

            rows.append({
                "K": K,
                "Modality": "CONS",
                "Factor": factor,
                "AUC": np.nan,
                "Silhouette": np.nan,
                "CramersV": V,
                "Stability": stab_val,
                "Source": "consensus"
            })


# ============================================================
# SAVE RAW MASTER TABLE
# ============================================================
df_all = pd.DataFrame(rows)
raw_path = os.path.join(MULTIK_DIR, "master_results_raw.csv")
df_all.to_csv(raw_path, index=False)
print("\nSaved RAW master table:", raw_path)


# ============================================================
# BUILD FINAL SUMMARY TABLE
# ============================================================
def add_model_score(df):
    df = df.copy()

    metric_cols = [
        "AUC_APOE", "AUC_risk",
        "Silhouette_APOE",
        "Cramer_APOE",
        "Stability",
    ]

    for col in metric_cols:
        vals = df[col].astype(float)
        mask = vals.notna()
        if mask.sum() < 3:
            df[col + "_z"] = np.nan
        else:
            df[col + "_z"] = (vals - vals[mask].mean()) / vals[mask].std()

    df["ModelScore"] = df[[c for c in df.columns if c.endswith("_z")]].mean(axis=1)
    return df


summary_rows = []
for (K, modality), sub in df_all.groupby(["K", "Modality"]):
    apo = sub[sub["Factor"] == FACTOR_APOE]
    risk = sub[sub["Factor"] == FACTOR_RISK]

    AUC_APOE = apo["AUC"].iloc[0] if not apo.empty else np.nan
    Sil_APOE = apo["Silhouette"].iloc[0] if not apo.empty else np.nan
    V_APOE   = apo["CramersV"].iloc[0] if not apo.empty else np.nan
    AUC_risk = risk["AUC"].iloc[0] if not risk.empty else np.nan
    stab     = sub["Stability"].dropna().iloc[0] if sub["Stability"].notna().any() else np.nan

    summary_rows.append({
        "K": K,
        "Modality": modality,
        "AUC_APOE": AUC_APOE,
        "AUC_risk": AUC_risk,
        "Silhouette_APOE": Sil_APOE,
        "Cramer_APOE": V_APOE,
        "Stability": stab,
    })

df_summary = pd.DataFrame(summary_rows)
df_summary = add_model_score(df_summary)

final_path = os.path.join(MULTIK_DIR, "master_results.csv")
df_summary.to_csv(final_path, index=False)
print("Saved FINAL master table:", final_path)


# ============================================================
# FIGURES
# ============================================================

# ---- ModelScore heatmap ----
pivot = df_summary.pivot(index="K", columns="Modality", values="ModelScore")

plt.figure(figsize=(7, 5), dpi=300)
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
plt.title("ModelScore by K and Modality")
plt.tight_layout()
plt.savefig(os.path.join(MULTIK_DIR, "modelscore_heatmap.png"))
plt.close()

# ---- AUC vs K (APOE) ----
plt.figure(figsize=(7, 5), dpi=300)
for mod in df_summary["Modality"].unique():
    sub = df_summary[df_summary["Modality"] == mod]
    plt.plot(sub["K"], sub["AUC_APOE"], marker="o", label=mod)
plt.xlabel("K")
plt.ylabel("AUC (APOE)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(MULTIK_DIR, "auc_vs_K_APOE.png"))
plt.close()

# ---- AUC vs K (risk for AD) ----
plt.figure(figsize=(7, 5), dpi=300)
for mod in df_summary["Modality"].unique():
    sub = df_summary[df_summary["Modality"] == mod]
    plt.plot(sub["K"], sub["AUC_risk"], marker="o", label=mod)
plt.xlabel("K")
plt.ylabel("AUC (risk_for_ad)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(MULTIK_DIR, "auc_vs_K_risk.png"))
plt.close()

print("\n🎉 DONE — Multi-K GAUDI aggregation + plots generated!\n")
