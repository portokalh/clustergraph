#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
part4_aggregatefrompart3.py

Aggregate multi-K GAUDI latent stats across:
    K ∈ {10, 20, 30, 50}
    modalities ∈ {CORR, EUCLID, WASS, CONS (consensus)}

Inputs per K (in results_KXX/):
    - distance_metrics_KXX.csv  or distance_metrics.csv
    - kmeans_chisq_results.csv or kmeans_chisq_results_KXX.csv
    - consensus_cluster_enrichment*.csv
    - consensus_stability*.csv

Outputs:
    - multiK_summary/master_results.csv
    - multiK_summary/modelscore_heatmap.png
    - multiK_summary/auc_vs_K_APOE.png
    - multiK_summary/auc_vs_K_risk.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation"
ANALYSIS_ROOT = os.path.join(
    ROOT,
    "columns4gaudi111825",
    "columna-analyses111925"
)
MULTIK_DIR = os.path.join(ANALYSIS_ROOT, "multiK_summary")
os.makedirs(MULTIK_DIR, exist_ok=True)

K_LIST = [4, 6, 8, 10, 12, 20, 25, 30, 50] 
MODALITIES = ["CORR", "EUCLID", "WASS"]
FACTOR_APOE = "APOE"
FACTOR_RISK = "risk_for_ad"


# ------------------------------------------------------------
# Helper: try several filename patterns
# ------------------------------------------------------------
def find_file(base_dir, base_name, K=None):
    """
    Look for a CSV under base_dir with several possible patterns:
        base_name.csv
        base_name_K{K}.csv
        base_name_K{K_graph}.csv
    Returns full path or None.
    """
    candidates = [os.path.join(base_dir, f"{base_name}.csv")]
    if K is not None:
        candidates.append(os.path.join(base_dir, f"{base_name}_K{K}.csv"))
        candidates.append(os.path.join(base_dir, f"{base_name}_K{int(K)}.csv"))

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ------------------------------------------------------------
# Collect all rows
# ------------------------------------------------------------
rows = []

for K in K_LIST:
    res_dir = os.path.join(ANALYSIS_ROOT, f"results_K{K}")
    print(f"\n=========== PROCESSING K={K} in {res_dir} ===========")

    if not os.path.isdir(res_dir):
        print(f"⚠ No directory for K={K}, skipping.")
        continue

    # -------- distance_metrics: per-modality AUC/silhouette --------
    dm_path = find_file(res_dir, "distance_metrics", K)
    if dm_path is None:
        print(f"⚠ Missing distance metrics for K={K}")
        df_dm = None
    else:
        print(f"✓ Using distance metrics: {dm_path}")
        df_dm = pd.read_csv(dm_path)

    # -------- kmeans_chisq_results: per-modality Cramer's V --------
    chi_path = find_file(res_dir, "kmeans_chisq_results", K)
    if chi_path is None:
        print(f"⚠ Missing χ² results for K={K}")
        df_chi = None
    else:
        print(f"✓ Using k-means χ²: {chi_path}")
        df_chi = pd.read_csv(chi_path)

    # -------- consensus_cluster_enrichment: consensus Cramer's V ----
    cons_enrich_path = find_file(res_dir, "consensus_cluster_enrichment", K)
    if cons_enrich_path is None:
        print(f"⚠ Missing consensus enrichment for K={K}")
        df_cons = None
    else:
        print(f"✓ Using consensus enrichment: {cons_enrich_path}")
        df_cons = pd.read_csv(cons_enrich_path)

    # -------- consensus_stability: cluster stability ---------------
    cons_stab_path = find_file(res_dir, "consensus_stability", K)
    if cons_stab_path is None:
        print(f"⚠ Missing consensus stability for K={K}")
        df_stab = None
    else:
        print(f"✓ Using consensus stability: {cons_stab_path}")
        df_stab = pd.read_csv(cons_stab_path)

    # ============================================================
    # 1) Per-modality rows: CORR, EUCLID, WASS
    # ============================================================
    if df_dm is not None:
        # normalize column names across scripts
        # expected columns: K_graph, Distance, Factor, Silhouette, AUC_mean, AUC_std
        # but some versions may lack K_graph
        if "K_graph" not in df_dm.columns:
            df_dm["K_graph"] = K

        for modality in MODALITIES:
            for factor in [FACTOR_APOE, FACTOR_RISK]:
                # Subset for this modality + factor
                sub = df_dm[(df_dm["Distance"] == modality) &
                            (df_dm["Factor"] == factor)]
                if sub.empty:
                    AUC = np.nan
                    Sil = np.nan
                else:
                    AUC = sub["AUC_mean"].iloc[0]
                    Sil = sub["Silhouette"].iloc[0]

                # Cramer's V from kmeans_chisq_results
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
                    "Stability": np.nan,  # not defined per modality
                    "Source": "kmeans"
                })

    # ============================================================
    # 2) Consensus rows: treat as Modality = "CONS"
    # ============================================================
    if df_cons is not None:
        # We only get CramersV per factor
        for factor in df_cons["Factor"].unique():
            sub_c = df_cons[df_cons["Factor"] == factor]
            if sub_c.empty:
                continue
            V = sub_c["CramersV"].iloc[0]
            # Stability: average over clusters (one value per K)
            stab_val = np.nan
            if df_stab is not None and "Stability" in df_stab.columns:
                stab_val = df_stab["Stability"].mean()

            rows.append({
                "K": K,
                "Modality": "CONS",    # consensus combination of CORR/EUCLID/WASS
                "Factor": factor,
                "AUC": np.nan,         # AUC not defined at consensus level here
                "Silhouette": np.nan,  # silhouette not defined here
                "CramersV": V,
                "Stability": stab_val,
                "Source": "consensus"
            })

# ------------------------------------------------------------
# Build master table
# ------------------------------------------------------------
df_all = pd.DataFrame(rows)
master_path = os.path.join(MULTIK_DIR, "master_results_raw.csv")
df_all.to_csv(master_path, index=False)
print("\nSaved RAW master table →", master_path)
print(df_all.head())

if df_all.empty:
    print("\n⚠ master_results is empty – no plots generated.")
    raise SystemExit

# ------------------------------------------------------------
# Pivot into a publication-friendly master table
#  We’ll focus on APOE and risk_for_ad and compute a ModelScore
#  per (K, Modality) using all available metrics.
# ------------------------------------------------------------
def add_model_score(df):
    """
    For each row, compute a composite ModelScore based on:
      AUC (for APOE & risk)
      Silhouette (APOE)
      CramersV (APOE)
      Stability (CONS)
    We z-score each numeric metric across all rows and then
    take the mean of the available z-scores for that row.
    """
    df = df.copy()
    metric_cols = ["AUC_APOE", "AUC_risk", "Silhouette_APOE", "Cramer_APOE", "Stability"]

    # z-score each metric
    for col in metric_cols:
        vals = df[col].values.astype("float")
        mask = ~np.isnan(vals)
        if mask.sum() < 3:
            df[col + "_z"] = np.nan
            continue
        m = np.nanmean(vals[mask])
        s = np.nanstd(vals[mask])
        if s == 0:
            df[col + "_z"] = np.nan
        else:
            df[col + "_z"] = (vals - m) / s

    # row-wise mean of available z-scores
    zcols = [c for c in df.columns if c.endswith("_z")]
    df["ModelScore"] = df[zcols].mean(axis=1, skipna=True)
    return df

# Build per (K, Modality) summary rows
summary_rows = []
for (K, modality), sub in df_all.groupby(["K", "Modality"]):
    # pull APOE row
    apo = sub[sub["Factor"] == FACTOR_APOE]
    # pull risk row
    rsk = sub[sub["Factor"] == FACTOR_RISK]

    AUC_APOE = apo["AUC"].iloc[0] if not apo.empty else np.nan
    Sil_APOE = apo["Silhouette"].iloc[0] if not apo.empty else np.nan
    V_APOE   = apo["CramersV"].iloc[0] if not apo.empty else np.nan

    AUC_risk = rsk["AUC"].iloc[0] if not rsk.empty else np.nan

    # stability: only from consensus, but we assign it for CONS modality
    stab_vals = sub["Stability"].dropna()
    stab = stab_vals.iloc[0] if not stab_vals.empty else np.nan

    summary_rows.append({
        "K": K,
        "Modality": modality,
        "AUC_APOE": AUC_APOE,
        "AUC_risk": AUC_risk,
        "Silhouette_APOE": Sil_APOE,
        "Cramer_APOE": V_APOE,
        "Stability": stab
    })

df_summary = pd.DataFrame(summary_rows)
df_summary = add_model_score(df_summary)

master_final_path = os.path.join(MULTIK_DIR, "master_results.csv")
df_summary.to_csv(master_final_path, index=False)
print("\nSaved FINAL master table →", master_final_path)
print(df_summary)


# ------------------------------------------------------------
# FIGURES
# ------------------------------------------------------------

# 1) Heatmap of ModelScore (rows: K, cols: Modality)
pivot_score = df_summary.pivot(index="K", columns="Modality", values="ModelScore")
plt.figure(figsize=(6, 4), dpi=300)
sns.heatmap(pivot_score, annot=True, fmt=".2f", cmap="viridis")
plt.title("Composite ModelScore by K and Modality")
plt.ylabel("K (graph)")
plt.tight_layout()
heatmap_path = os.path.join(MULTIK_DIR, "modelscore_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()
print("Saved →", heatmap_path)

# 2) AUC vs K for APOE (per modality)
plt.figure(figsize=(6, 4), dpi=300)
for modality in df_summary["Modality"].unique():
    sub = df_summary[(df_summary["Modality"] == modality)]
    if sub["AUC_APOE"].notna().sum() == 0:
        continue
    plt.plot(sub["K"], sub["AUC_APOE"], marker="o", label=modality)
plt.xlabel("K (graph)")
plt.ylabel("AUC_APOE")
plt.title("APOE AUC vs K for each modality")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
auc_apoe_path = os.path.join(MULTIK_DIR, "auc_vs_K_APOE.png")
plt.savefig(auc_apoe_path, dpi=300)
plt.close()
print("Saved →", auc_apoe_path)

# 3) AUC vs K for risk_for_ad (per modality)
plt.figure(figsize=(6, 4), dpi=300)
for modality in df_summary["Modality"].unique():
    sub = df_summary[(df_summary["Modality"] == modality)]
    if sub["AUC_risk"].notna().sum() == 0:
        continue
    plt.plot(sub["K"], sub["AUC_risk"], marker="o", label=modality)
plt.xlabel("K (graph)")
plt.ylabel("AUC_risk_for_ad")
plt.title("Risk-for-AD AUC vs K for each modality")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
auc_risk_path = os.path.join(MULTIK_DIR, "auc_vs_K_risk.png")
plt.savefig(auc_risk_path, dpi=300)
plt.close()
print("Saved →", auc_risk_path)

print("\n✅ DONE: multi-K aggregation + figures saved.\n")
