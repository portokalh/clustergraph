#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 21:16:11 2025

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PART 7 — LATENT SALIENCY & SHAP

Author: Alexandra Badea (with ChatGPT)
Date: 2025-11-20

Goal
----
For a given K (e.g., K=25) and GAUDI latents (Joint MD+QSM):

  • Load CORR / EUCLID / WASS latents and align them to metadata.
  • For APOE and risk_for_ad:
      - Train simple classifiers on standardized latents
        (Logistic Regression for APOE, Random Forest for risk).
      - Compute latent-dimension effect sizes (Cohen's d, mean diff).
  • If `shap` is available:
      - Compute SHAP values in latent space.
      - Save mean |SHAP| per latent dimension.
      - Generate SHAP summary plots for APOE and risk_for_ad.

Outputs
-------
ROOT/columns4gaudi111825/columna-analyses111925/saliency_K{K}/
    latent_effects/
        latent_effects_{metric}_APOE.csv
        latent_effects_{metric}_risk_for_ad.csv
    shap_tables/      (if shap installed)
        shap_importance_{metric}_APOE.csv
        shap_importance_{metric}_risk_for_ad.csv
    shap_plots/       (if shap installed)
        shap_summary_bar_{metric}_APOE.png
        shap_summary_beeswarm_{metric}_APOE.png
        shap_summary_bar_{metric}_risk_for_ad.png
        shap_summary_beeswarm_{metric}_risk_for_ad.png
"""

# ================================================================
# IMPORTS
# ================================================================
import os
import numpy as np
import pandas as pd

import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import seaborn as sns

# Try to import shap (optional)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠ `shap` is not installed. SHAP analyses will be skipped.")


# ================================================================
# PATHS / PARAMETERS
# ================================================================
ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation"
COLUMNS_ROOT = os.path.join(ROOT, "columns4gaudi111825", "columna-analyses111925")
MDATA_PATH = os.path.join(ROOT, "processed_graph_data", "metadata_with_PCs.xlsx")

GRAPH_K = 25
METRICS = ["CORR", "EUCLID", "WASS"]   # latent distance metrics

SALIENCY_ROOT = os.path.join(COLUMNS_ROOT, f"saliency_K{GRAPH_K}")
LATENT_EFFECT_DIR = os.path.join(SALIENCY_ROOT, "latent_effects")
SHAP_TABLE_DIR   = os.path.join(SALIENCY_ROOT, "shap_tables")
SHAP_PLOT_DIR    = os.path.join(SALIENCY_ROOT, "shap_plots")

for d in [SALIENCY_ROOT, LATENT_EFFECT_DIR, SHAP_TABLE_DIR, SHAP_PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# Traits / targets
TARGETS = ["APOE", "risk_for_ad"]


# ================================================================
# HELPERS
# ================================================================
def cohen_d_two_groups(x, g):
    """
    Compute Cohen's d for x across two groups defined by g (0/1 or two distinct labels).
    Returns np.nan if fewer than 2 groups or too few samples.
    """
    x = np.asarray(x)
    g = np.asarray(g)
    uniq = np.unique(g)
    if uniq.size != 2:
        return np.nan

    g1, g2 = uniq
    x1 = x[g == g1]
    x2 = x[g == g2]
    if x1.size < 5 or x2.size < 5:
        return np.nan

    m1, m2 = x1.mean(), x2.mean()
    s1, s2 = x1.std(ddof=1), x2.std(ddof=1)
    # pooled SD
    sp = np.sqrt(((x1.size - 1) * s1**2 + (x2.size - 1) * s2**2) / (x1.size + x2.size - 2))
    if sp == 0:
        return np.nan
    return float((m1 - m2) / sp)


def mean_difference_two_groups(x, g):
    """
    Mean difference (group1 - group2) for 2-group target.
    """
    x = np.asarray(x)
    g = np.asarray(g)
    uniq = np.unique(g)
    if uniq.size != 2:
        return np.nan
    g1, g2 = uniq
    x1 = x[g == g1]
    x2 = x[g == g2]
    if x1.size < 1 or x2.size < 1:
        return np.nan
    return float(x1.mean() - x2.mean())


def ttest_two_groups(x, g):
    """
    t-statistic (Welch's) across 2 groups.
    """
    x = np.asarray(x)
    g = np.asarray(g)
    uniq = np.unique(g)
    if uniq.size != 2:
        return np.nan
    g1, g2 = uniq
    x1 = x[g == g1]
    x2 = x[g == g2]
    if x1.size < 3 or x2.size < 3:
        return np.nan
    t, _ = ttest_ind(x1, x2, equal_var=False)
    return float(t)


def prepare_targets(df_sub):
    """
    Create numeric target arrays for APOE and risk_for_ad.
    Returns dict: {"APOE": (y_apoe, mask_apoe), "risk_for_ad": (y_risk, mask_risk)}
    where mask_* indicates non-NA samples.
    """
    out = {}

    # APOE: binarize E4- vs E4+
    if "APOE" in df_sub.columns:
        apoe_str = df_sub["APOE"].astype(str).values
        # keep only E4- / E4+
        mask_apoe = np.isin(apoe_str, ["E4-", "E4+"])
        y_apoe = np.where(apoe_str[mask_apoe] == "E4+", 1, 0)
        out["APOE"] = (y_apoe, mask_apoe)
    else:
        out["APOE"] = (None, None)

    # risk_for_ad: treat as ordinal 0/1/2/3 (or fewer)
    if "risk_for_ad" in df_sub.columns:
        r_str = df_sub["risk_for_ad"].astype(str).values
        mask_risk = ~np.isin(r_str, ["", "nan", "NaN", "None"])
        # map to integers
        risk_map = {"0": 0, "1": 1, "2": 2, "3": 3}
        y_risk = np.array([risk_map.get(v, np.nan) for v in r_str[mask_risk]], dtype=float)
        # drop NaNs
        not_nan = ~np.isnan(y_risk)
        final_mask = mask_risk.copy()
        idx_mask = np.where(mask_risk)[0]
        final_mask[idx_mask[~not_nan]] = False
        y_risk = y_risk[not_nan].astype(int)
        out["risk_for_ad"] = (y_risk, final_mask)
    else:
        out["risk_for_ad"] = (None, None)

    return out


# ================================================================
# 1) LOAD METADATA + GRAPHS (subject order)
# ================================================================
print("=== PART 7: LATENT SALIENCY & SHAP ===")
print("Loading metadata from:", MDATA_PATH)
df_all = pd.read_excel(MDATA_PATH)
df_all["MRI_Exam"] = df_all["MRI_Exam"].astype(str).str.zfill(5)

GRAPHS_PT = os.path.join(
    COLUMNS_ROOT,
    "graphs_knn",
    f"k{GRAPH_K}",
    f"md_shared_knn_k{GRAPH_K}_corr_euclid_wass.pt"
)
print("Loading graphs from:", GRAPHS_PT)
graphs = torch.load(GRAPHS_PT, map_location="cpu")
subject_ids = [str(getattr(g, "subject_id")).zfill(5) for g in graphs]
print("Found", len(subject_ids), "subjects in graphs.")

# Align metadata to graphs
df_sub = df_all[df_all["MRI_Exam"].isin(subject_ids)].copy()
df_sub["__order"] = df_sub["MRI_Exam"].apply(lambda s: subject_ids.index(s))
df_sub = df_sub.sort_values("__order").reset_index(drop=True)
df_sub.drop(columns="__order", inplace=True)
print("Aligned metadata shape:", df_sub.shape)

# Prepare target arrays (APOE, risk_for_ad)
targets_dict = prepare_targets(df_sub)


# ================================================================
# 2) LOOP OVER METRICS: LOAD LATENTS, STANDARDIZE
# ================================================================
for metric in METRICS:
    print(f"\n================ METRIC: {metric} (K={GRAPH_K}) ================\n")

    latent_path = os.path.join(
        COLUMNS_ROOT,
        f"latent_k{GRAPH_K}",
        f"latent_epochs_Joint_{metric}",
        f"latent_final_Joint_{metric}.npy"
    )
    print("Loading latents from:", latent_path)
    Z = np.load(latent_path)

    if Z.shape[0] != len(subject_ids):
        raise RuntimeError(
            f"{metric}: latent rows ({Z.shape[0]}) != graph subject rows ({len(subject_ids)})"
        )

    keep_mask = np.isin(subject_ids, df_sub["MRI_Exam"].values)
    Z = Z[keep_mask]
    assert Z.shape[0] == df_sub.shape[0]

    print("Latent shape:", Z.shape)
    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z)
    latent_dim = Zs.shape[1]
    dims = np.arange(1, latent_dim + 1)

    # ------------------------------------------------------------
    # 2A) LATENT EFFECTS: APOE (binary)
    # ------------------------------------------------------------
    y_apoe, mask_apoe = targets_dict["APOE"]
    if y_apoe is not None and mask_apoe is not None and mask_apoe.sum() >= 10:
        print(f"\n-- Latent saliency for APOE ({metric}) --")
        Zs_apoe = Zs[mask_apoe]

        # Simple logistic regression classifier on latents
        clf_apoe = LogisticRegression(max_iter=5000, penalty="l2")
        clf_apoe.fit(Zs_apoe, y_apoe)

        # Effect sizes per dimension
        effects_rows = []
        for d in range(latent_dim):
            z_d = Zs_apoe[:, d]
            d_cohen = cohen_d_two_groups(z_d, y_apoe)
            mean_diff = mean_difference_two_groups(z_d, y_apoe)
            tstat = ttest_two_groups(z_d, y_apoe)
            effects_rows.append(
                {
                    "K_graph": GRAPH_K,
                    "Metric": metric,
                    "Latent_dim": d + 1,
                    "Cohen_d_APOE": d_cohen,
                    "MeanDiff_APOE": mean_diff,
                    "t_APOE": tstat,
                    "LogReg_coef_APOE": float(clf_apoe.coef_[0, d]),
                }
            )

        df_effects = pd.DataFrame(effects_rows)
        eff_out = os.path.join(LATENT_EFFECT_DIR, f"latent_effects_{metric}_APOE.csv")
        df_effects.to_csv(eff_out, index=False)
        print("Saved latent effects for APOE to:", eff_out)
        print(df_effects.sort_values("Cohen_d_APOE", key=lambda x: np.abs(x), ascending=False).head(10))

        # SHAP for APOE (if available)
        if HAS_SHAP:
            print("Computing SHAP values for APOE (LogisticRegression)...")
            explainer = shap.LinearExplainer(clf_apoe, Zs_apoe, feature_dependence="correlation")
            # Use a subset of samples for SHAP to keep runtime reasonable
            rng = np.random.default_rng(42)
            n_apoe = Zs_apoe.shape[0]
            max_samples = min(200, n_apoe)
            idx = rng.choice(n_apoe, size=max_samples, replace=False)
            Zs_apoe_sample = Zs_apoe[idx]

            shap_vals = explainer.shap_values(Zs_apoe_sample)  # shape (n_samples, latent_dim)
            shap_vals = np.array(shap_vals)

            # Mean |SHAP| across samples per dimension
            mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
            df_shap_imp = pd.DataFrame(
                {
                    "K_graph": GRAPH_K,
                    "Metric": metric,
                    "Latent_dim": dims,
                    "mean_abs_SHAP_APOE": mean_abs_shap,
                }
            )
            shap_imp_out = os.path.join(SHAP_TABLE_DIR, f"shap_importance_{metric}_APOE.csv")
            df_shap_imp.to_csv(shap_imp_out, index=False)
            print("Saved SHAP importance for APOE to:", shap_imp_out)

            # SHAP summary bar plot
            plt.figure(figsize=(7, 4), dpi=250)
            order = np.argsort(-mean_abs_shap)[:15]  # top 15 dims
            plt.bar(
                [f"z{d}" for d in dims[order]],
                mean_abs_shap[order]
            )
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("mean |SHAP|")
            plt.title(f"APOE SHAP importance (top 15 latent dims)\nK={GRAPH_K}, {metric}")
            plt.tight_layout()
            out_bar = os.path.join(SHAP_PLOT_DIR, f"shap_summary_bar_{metric}_APOE.png")
            plt.savefig(out_bar, dpi=250)
            plt.close()
            print("Saved:", out_bar)

            # SHAP beeswarm-style plot (using shap's built-in function)
            try:
                shap.summary_plot(
                    shap_vals,
                    Zs_apoe_sample,
                    feature_names=[f"z{d}" for d in dims],
                    show=False
                )
                out_bee = os.path.join(SHAP_PLOT_DIR, f"shap_summary_beeswarm_{metric}_APOE.png")
                plt.tight_layout()
                plt.savefig(out_bee, dpi=250, bbox_inches="tight")
                plt.close()
                print("Saved:", out_bee)
            except Exception as e:
                print("SHAP beeswarm plot failed for APOE:", e)
        else:
            print("Skipped SHAP for APOE (no shap installed).")

    else:
        print(f"\nNo valid APOE labels for {metric}, skipping APOE saliency.")

    # ------------------------------------------------------------
    # 2B) LATENT EFFECTS: risk_for_ad (ordinal / multi-class)
    # ------------------------------------------------------------
    y_risk, mask_risk = targets_dict["risk_for_ad"]
    if y_risk is not None and mask_risk is not None and mask_risk.sum() >= 10:
        print(f"\n-- Latent saliency for risk_for_ad ({metric}) --")

        Zs_risk = Zs[mask_risk]
        # RandomForest for multi-class risk
        rf_risk = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        rf_risk.fit(Zs_risk, y_risk)

        # For effect size, we can contrast "0" vs ">=1" as high vs low risk
        y_bin = (y_risk >= 1).astype(int)

        effects_rows = []
        for d in range(latent_dim):
            z_d = Zs_risk[:, d]
            d_cohen = cohen_d_two_groups(z_d, y_bin)
            mean_diff = mean_difference_two_groups(z_d, y_bin)
            tstat = ttest_two_groups(z_d, y_bin)
            # RF feature importance is model-level, but we also record it separately
            effects_rows.append(
                {
                    "K_graph": GRAPH_K,
                    "Metric": metric,
                    "Latent_dim": d + 1,
                    "Cohen_d_risk_bin01": d_cohen,
                    "MeanDiff_risk_bin01": mean_diff,
                    "t_risk_bin01": tstat,
                }
            )

        df_effects_risk = pd.DataFrame(effects_rows)
        eff_risk_out = os.path.join(LATENT_EFFECT_DIR, f"latent_effects_{metric}_risk_for_ad.csv")
        df_effects_risk.to_csv(eff_risk_out, index=False)
        print("Saved latent effects for risk_for_ad to:", eff_risk_out)
        print(
            df_effects_risk.sort_values(
                "Cohen_d_risk_bin01",
                key=lambda x: np.abs(x),
                ascending=False
            ).head(10)
        )

        # SHAP for risk (TreeExplainer)
        if HAS_SHAP:
            print("Computing SHAP values for risk_for_ad (RandomForest)...")
            explainer_risk = shap.TreeExplainer(rf_risk)
            # sample subset of Zs_risk
            rng = np.random.default_rng(123)
            n_risk = Zs_risk.shape[0]
            max_samples = min(200, n_risk)
            idx_risk = rng.choice(n_risk, size=max_samples, replace=False)
            Zs_risk_sample = Zs_risk[idx_risk]

            shap_vals_risk = explainer_risk.shap_values(Zs_risk_sample)
            # TreeExplainer returns shap_vals as list [class0, class1, ...]
            # We'll aggregate mean |SHAP| across classes
            shap_vals_risk = np.array(shap_vals_risk)  # shape (n_classes, n_samples, latent_dim)
            mean_abs_shap_risk = np.mean(np.mean(np.abs(shap_vals_risk), axis=1), axis=0)

            df_shap_imp_risk = pd.DataFrame(
                {
                    "K_graph": GRAPH_K,
                    "Metric": metric,
                    "Latent_dim": dims,
                    "mean_abs_SHAP_risk": mean_abs_shap_risk,
                }
            )
            shap_imp_risk_out = os.path.join(SHAP_TABLE_DIR, f"shap_importance_{metric}_risk_for_ad.csv")
            df_shap_imp_risk.to_csv(shap_imp_risk_out, index=False)
            print("Saved SHAP importance for risk_for_ad to:", shap_imp_risk_out)

            # Bar plot (top 15)
            plt.figure(figsize=(7, 4), dpi=250)
            order = np.argsort(-mean_abs_shap_risk)[:15]
            plt.bar(
                [f"z{d}" for d in dims[order]],
                mean_abs_shap_risk[order]
            )
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("mean |SHAP|")
            plt.title(f"risk_for_ad SHAP importance (top 15 latent dims)\nK={GRAPH_K}, {metric}")
            plt.tight_layout()
            out_bar_risk = os.path.join(SHAP_PLOT_DIR, f"shap_summary_bar_{metric}_risk_for_ad.png")
            plt.savefig(out_bar_risk, dpi=250)
            plt.close()
            print("Saved:", out_bar_risk)

            # Beeswarm-like summary: shap.summary_plot expects one class or 2D array;
            # we'll pick the shap values for the highest risk class (max label).
            try:
                max_class = int(np.max(y_risk))
                class_idx = min(max_class, shap_vals_risk.shape[0] - 1)
                shap.summary_plot(
                    shap_vals_risk[class_idx],
                    Zs_risk_sample,
                    feature_names=[f"z{d}" for d in dims],
                    show=False
                )
                out_bee_risk = os.path.join(SHAP_PLOT_DIR, f"shap_summary_beeswarm_{metric}_risk_for_ad.png")
                plt.tight_layout()
                plt.savefig(out_bee_risk, dpi=250, bbox_inches="tight")
                plt.close()
                print("Saved:", out_bee_risk)
            except Exception as e:
                print("SHAP beeswarm plot failed for risk_for_ad:", e)
        else:
            print("Skipped SHAP for risk_for_ad (no shap installed).")
    else:
        print(f"\nNo valid risk_for_ad labels for {metric}, skipping risk saliency.")

print("\n✅ DONE — PART 7 latent saliency & SHAP complete.")
print("Outputs in:", SALIENCY_ROOT)
