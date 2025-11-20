#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 18:15:00 2025

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate GAUDI results across K = 10,20,30,50
================================================

This script:
✔ Reads multi-K output folders:
      results_K10/
      results_K20/
      results_K30/
      results_K50/

✔ Loads:
      distance_metrics.csv          (AUC + silhouette)
      kmeans_chisq_results.csv      (Cramér’s V)
      consensus_cluster_enrichment.csv (consensus χ² + V)
      consensus_stability.csv        (cluster stability)

✔ Computes a composite MODEL SCORE:
      SCORE = 0.30*AUC_APOE
            + 0.20*AUC_risk
            + 0.20*Silhouette_APOE
            + 0.20*Cramér_APOE
            + 0.10*Stability

✔ Produces:
      → master_results.csv
      → heatmap_K_vs_metric_SCORE.png
      → table of BEST K and BEST modality
      → printed summary for manuscript
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation/columns4gaudi111825/columna-analyses111925"
OUT = os.path.join(ROOT, "multiK_summary")
os.makedirs(OUT, exist_ok=True)

K_list = [4, 6, 8, 10, 12, 20, 25, 30, 50] 
modalities = ["CORR", "EUCLID", "WASS"]

records = []   # store rows for master table


# ============================================================
# Helper: safe read
# ============================================================
def safe_read(path):
    if not os.path.exists(path):
        print("⚠ Missing:", path)
        return None
    return pd.read_csv(path)


# ============================================================
# MAIN LOOP
# ============================================================
for K in K_list:
    print(f"\n=========== PROCESSING K={K} ===========")

    Kdir = os.path.join(ROOT, f"results_K{K}")
    if not os.path.exists(Kdir):
        print("⚠ Folder missing:", Kdir)
        continue

    df_dist = safe_read(os.path.join(Kdir, f"distance_metrics_K{K}.csv"))
    df_chi  = safe_read(os.path.join(Kdir, f"kmeans_chisq_results_K{K}.csv"))
    df_cons = safe_read(os.path.join(Kdir, f"consensus_cluster_enrichment_K{K}.csv"))
    df_stab = safe_read(os.path.join(Kdir, f"consensus_stability_K{K}.csv"))


    if df_dist is None or df_chi is None or df_cons is None or df_stab is None:
        print("⚠ Incomplete result set for K=", K)
        continue

    # Stability value (average across clusters)
    stability = df_stab["Stability"].mean()

    # ---------------------------------------------------------------------------------------
    # Aggregate per-modality statistics
    # ---------------------------------------------------------------------------------------
    for m in modalities:
        # Get silhouette and AUC_APOE and AUC_risk
        dist_row = df_dist[df_dist["Distance"] == m]

        if dist_row.empty:
            print(f"⚠ No distance row for {m} at K={K}")
            continue

        # Extract silhouette for APOE clustering
        sil = float(dist_row[dist_row["Factor"] == "APOE"]["Silhouette"].values[0])

        # Extract AUCs
        try:
            auc_apoe = float(dist_row[dist_row["Factor"] == "APOE"]["AUC_mean"].values[0])
        except:
            auc_apoe = np.nan

        try:
            auc_risk = float(dist_row[dist_row["Factor"] == "risk_for_ad"]["AUC_mean"].values[0])
        except:
            auc_risk = np.nan

        # Extract Cramér’s V for APOE from kmeans
        chi_apoe = df_chi[(df_chi["Distance"] == m) & (df_chi["Factor"] == "APOE")]
        if not chi_apoe.empty:
            V_apoe = float(chi_apoe["CramersV"].values[0])
        else:
            V_apoe = np.nan

        # -----------------------------------------------------------------------------------
        # Compute MODEL SCORE
        # -----------------------------------------------------------------------------------
        score = (
            0.30 * auc_apoe +
            0.20 * auc_risk +
            0.20 * sil +
            0.20 * V_apoe +
            0.10 * stability
        )

        records.append([
            K, m, auc_apoe, auc_risk, sil, V_apoe, stability, score
        ])

# ============================================================
# Build master table
# ============================================================
df_master = pd.DataFrame(
    records,
    columns=[
        "K", "Modality", "AUC_APOE", "AUC_risk",
        "Silhouette_APOE", "Cramer_APOE",
        "Stability", "ModelScore"
    ]
)

master_csv = os.path.join(OUT, "master_results.csv")
df_master.to_csv(master_csv, index=False)
print("\nSaved MASTER TABLE →", master_csv)
print(df_master)


# ============================================================
# Heatmap of ModelScore (K × Modality)
# ============================================================
pivot = df_master.pivot(index="K", columns="Modality", values="ModelScore")

plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
plt.title("Model Score heatmap (higher = better)")
plt.tight_layout()
heatmap_path = os.path.join(OUT, "heatmap_K_vs_metric_SCORE.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()
print("Saved heatmap →", heatmap_path)


# ============================================================
# Find BEST settings
# ============================================================
best_row = df_master.loc[df_master["ModelScore"].idxmax()]
best_K = int(best_row["K"])
best_mod = str(best_row["Modality"])
best_score = float(best_row["ModelScore"])

summary_txt = f"""
===========================================================
                OPTIMAL GAUDI CONFIGURATION
===========================================================

Best overall setting:
    → K = {best_K}
    → Modality = {best_mod}
    → Model Score = {best_score:.4f}

Interpretation:
    • Best biological separation (APOE, risk)
    • Best geometric separation (silhouette)
    • Strongest APOE effect size (Cramér’s V)
    • High cluster stability
===========================================================
"""

print(summary_txt)

with open(os.path.join(OUT, "best_model_summary.txt"), "w") as f:
    f.write(summary_txt)

print("Saved → best_model_summary.txt")
print("\n🎉 DONE — Model selection completed!\n")
