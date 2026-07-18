#!/usr/bin/env python3
"""
SEED-V Q1 Complete Post-Screenshot Analysis Suite
=================================================
This script regenerates and saves all analyses/figures we created after the
CDT/EER-vs-drift screenshot, without overwriting previous runs.

Run from terminal:
cd /home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
source p4_seedv_env/bin/activate
python -u /path/to/seedv_q1_complete_post_screenshot_suite.py 2>&1 | tee outputs/run_04_q1_complete_post_screenshot/logs/run04_output.txt

Outputs:
outputs/run_04_q1_complete_post_screenshot/
  figures/
  tables/
  logs/
  checkpoints/
"""

import os
import json
import math
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import roc_curve, auc

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ============================================================
# 0. PATHS AND OUTPUT FOLDERS
# ============================================================
ROOT = Path("/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC")
PREV_OUT = ROOT / "outputs/run_02_q1_validation"
OUT = ROOT / "outputs/run_04_q1_complete_post_screenshot"
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
LOG_DIR = OUT / "logs"
CKPT_DIR = OUT / "checkpoints"
for d in [OUT, FIG_DIR, TAB_DIR, LOG_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SEED-V Q1 COMPLETE POST-SCREENSHOT ANALYSIS SUITE")
print("=" * 80)
print("Root:", ROOT)
print("Previous run folder:", PREV_OUT)
print("New output folder:", OUT)
print("Started:", datetime.now().isoformat())

# ============================================================
# Utility functions
# ============================================================
def safe_read_csv(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"Could not read {p}: {e}")
        return None

def find_first_existing(candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None

def standardize_columns(df):
    if df is None:
        return None
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    lower_map = {c.lower(): c for c in out.columns}
    rename = {}
    for want in ["variant", "seed", "eer", "auc", "drift_index", "enroll_emotion", "test_emotion", "subject"]:
        for c in out.columns:
            if c.lower() == want.lower():
                rename[c] = want
    out = out.rename(columns=rename)
    return out

def eer_auc_from_scores(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    mask = np.isfinite(scores) & np.isfinite(y_true)
    y_true, scores = y_true[mask], scores[mask]
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan, None
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float(fpr[idx]), float(auc(fpr, tpr)), float(thr[idx]), (fpr, tpr, fnr, thr)

def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / max(1, (na+nb-2)))
    if pooled == 0:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled)

def bootstrap_ci(values, n_boot=2000, seed=42, func=np.mean):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(func(sample))
    return float(func(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def savefig(name):
    p = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(p, dpi=300, bbox_inches="tight")
    print("Saved figure:", p)
    return p

# ============================================================
# 1. LOAD EXISTING RUN02 RESULT TABLES
# ============================================================
TAB_PREV = PREV_OUT / "tables"
print("\nSearching previous tables in:", TAB_PREV)

main_df = safe_read_csv(find_first_existing([
    TAB_PREV / "main_multiseed_results.csv",
    TAB_PREV / "main_multiseed_results_live.csv",
]))
main_df = standardize_columns(main_df)

cross_df = safe_read_csv(find_first_existing([
    TAB_PREV / "cross_emotion_results_all_seeds.csv",
    TAB_PREV / "cross_emotion_results.csv",
]))
cross_df = standardize_columns(cross_df)

drift_df = safe_read_csv(find_first_existing([
    TAB_PREV / "identity_drift_all_seeds.csv",
    TAB_PREV / "controlled_identity_drift_full.csv",
    TAB_PREV / "controlled_identity_drift.csv",
]))
drift_df = standardize_columns(drift_df)

score_df = safe_read_csv(find_first_existing([
    TAB_PREV / "score_level_genuine_impostor_scores.csv",
    TAB_PREV / "score_level_scores.csv",
    TAB_PREV / "genuine_impostor_scores.csv",
]))
score_df = standardize_columns(score_df)

for name, df in [("main_df", main_df), ("cross_df", cross_df), ("drift_df", drift_df), ("score_df", score_df)]:
    if df is None:
        print(f"{name}: MISSING")
    else:
        print(f"{name}: loaded shape={df.shape}, columns={list(df.columns)[:12]}")

# ============================================================
# 2. SELECT BEST VARIANT FROM EXISTING RESULTS
# ============================================================
best_variant = None
if main_df is not None and "variant" in main_df.columns and "eer" in [c.lower() for c in main_df.columns]:
    if "eer" not in main_df.columns:
        for c in main_df.columns:
            if c.lower() == "eer": main_df = main_df.rename(columns={c:"eer"})
    best_variant = main_df.groupby("variant")["eer"].mean().sort_values().index[0]
elif cross_df is not None and "variant" in cross_df.columns and "EER" in cross_df.columns:
    best_variant = cross_df.groupby("variant")["EER"].mean().sort_values().index[0]
elif cross_df is not None and "variant" in cross_df.columns and "eer" in cross_df.columns:
    best_variant = cross_df.groupby("variant")["eer"].mean().sort_values().index[0]
else:
    best_variant = "arcface_supcon_cnn"
print("Best/proposed variant:", best_variant)

# Normalize cross columns
if cross_df is not None:
    if "EER" in cross_df.columns and "eer" not in cross_df.columns:
        cross_df = cross_df.rename(columns={"EER":"eer"})
    if "AUC" in cross_df.columns and "auc" not in cross_df.columns:
        cross_df = cross_df.rename(columns={"AUC":"auc"})
    if "drift_index" not in cross_df.columns:
        for c in cross_df.columns:
            if "drift" in c.lower():
                cross_df = cross_df.rename(columns={c:"drift_index"})
                break

# If cross has multiple variants, keep proposed for drift figures
cross_prop = cross_df.copy() if cross_df is not None else None
if cross_prop is not None and "variant" in cross_prop.columns:
    cross_prop = cross_prop[cross_prop["variant"].astype(str) == str(best_variant)].copy()
    if len(cross_prop) == 0:
        cross_prop = cross_df.copy()

# ============================================================
# 3. Q1 FIGURES FROM EXISTING TABLES
# ============================================================
manifest = []

# 3.1 Multi-seed EER boxplot / bar comparison
if main_df is not None and "variant" in main_df.columns:
    # normalize eer/auc column names
    for c in list(main_df.columns):
        if c.lower() == "eer" and c != "eer": main_df = main_df.rename(columns={c:"eer"})
        if c.lower() == "auc" and c != "auc": main_df = main_df.rename(columns={c:"auc"})
    if "eer" in main_df.columns:
        labels = list(main_df.groupby("variant")["eer"].mean().sort_values(ascending=False).index)
        data = [main_df.loc[main_df["variant"] == v, "eer"].dropna().values for v in labels]
        plt.figure(figsize=(9,5))
        plt.boxplot(data, tick_labels=labels, showmeans=True)
        plt.ylabel("Equal Error Rate (EER)")
        plt.title("Multi-Seed EER Distribution across Baseline and Deep Models")
        plt.xticks(rotation=25, ha="right")
        plt.grid(axis="y", alpha=0.3)
        p = savefig("fig01_multiseed_eer_boxplot.png")
        manifest.append({"figure_file":p.name,"purpose":"Shows robustness across seeds and compares proposed model against baselines."})

# 3.2 Cross-emotion EER heatmap
if cross_prop is not None and {"enroll_emotion", "test_emotion", "eer"}.issubset(cross_prop.columns):
    mat = cross_prop.pivot_table(index="enroll_emotion", columns="test_emotion", values="eer", aggfunc="mean")
    mat.to_csv(TAB_DIR / "cross_emotion_eer_matrix.csv")
    plt.figure(figsize=(7,5.5))
    im = plt.imshow(mat.values, aspect="auto")
    plt.colorbar(im, label="Mean EER")
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=35, ha="right")
    plt.yticks(range(len(mat.index)), mat.index)
    plt.xlabel("Probe / test emotion")
    plt.ylabel("Enrollment emotion")
    plt.title("Cross-Emotion Verification Error Matrix")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i,j]
            if np.isfinite(val):
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    p = savefig("fig02_cross_emotion_eer_heatmap.png")
    manifest.append({"figure_file":p.name,"purpose":"Identifies which cognitive-state transitions are hardest for verification."})

# 3.3 Cross-emotion identity drift heatmap
if cross_prop is not None and {"enroll_emotion", "test_emotion", "drift_index"}.issubset(cross_prop.columns):
    mat = cross_prop.pivot_table(index="enroll_emotion", columns="test_emotion", values="drift_index", aggfunc="mean")
    mat.to_csv(TAB_DIR / "cross_emotion_identity_drift_matrix.csv")
    plt.figure(figsize=(7,5.5))
    im = plt.imshow(mat.values, aspect="auto")
    plt.colorbar(im, label="Mean identity drift index")
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=35, ha="right")
    plt.yticks(range(len(mat.index)), mat.index)
    plt.xlabel("Probe / test emotion")
    plt.ylabel("Enrollment emotion")
    plt.title("Cross-Emotion Identity Drift Matrix")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i,j]
            if np.isfinite(val):
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    p = savefig("fig03_cross_emotion_identity_drift_heatmap.png")
    manifest.append({"figure_file":p.name,"purpose":"Shows representation-level cognitive drift independent of final aggregate EER."})

# 3.4 EER vs drift with CDT band
cdt_mean = cdt_low = cdt_high = np.nan
if cross_prop is not None and {"drift_index", "eer"}.issubset(cross_prop.columns):
    x = cross_prop["drift_index"].astype(float).values
    y = cross_prop["eer"].astype(float).values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) >= 5:
        cdt_mean, cdt_low, cdt_high = bootstrap_ci(x, n_boot=2000, seed=123, func=np.mean)
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        # quadratic trend
        xs = np.linspace(np.min(x), np.max(x), 200)
        coef = np.polyfit(x, y, deg=2)
        ys = np.polyval(coef, xs)
        plt.figure(figsize=(8,5))
        plt.scatter(x, y, alpha=0.7, label="Cross-emotion conditions")
        plt.plot(xs, ys, label="Quadratic trend")
        plt.axvline(cdt_mean, linestyle="--", label=f"CDT mean = {cdt_mean:.3f}")
        plt.axvspan(cdt_low, cdt_high, alpha=0.2, label=f"CDT 95% band [{cdt_low:.3f}, {cdt_high:.3f}]")
        plt.xlabel("Identity drift index")
        plt.ylabel("Equal Error Rate (EER)")
        plt.title("Drift-Aware EEG Biometric Degradation under Cognitive Variability")
        plt.legend()
        plt.grid(alpha=0.3)
        p = savefig("fig04_eer_vs_drift_with_cdt_ci.png")
        manifest.append({"figure_file":p.name,"purpose":"Main Q1 figure linking identity drift to verification degradation with CDT confidence band."})
        pd.DataFrame([{
            "variant": best_variant, "cdt_mean": cdt_mean, "cdt_95_low": cdt_low, "cdt_95_high": cdt_high,
            "pearson_r": pearson_r, "pearson_p": pearson_p,
            "spearman_r": spearman_r, "spearman_p": spearman_p,
            "mean_eer": float(np.mean(y)), "std_eer": float(np.std(y, ddof=1)), "n_conditions": int(len(y))
        }]).to_csv(TAB_DIR / "cdt_and_drift_eer_statistics.csv", index=False)

# 3.5 Piecewise / phase transition model
if cross_prop is not None and {"drift_index", "eer"}.issubset(cross_prop.columns):
    df_tmp = cross_prop[["drift_index", "eer"]].dropna().sort_values("drift_index")
    x = df_tmp["drift_index"].values.astype(float)
    y = df_tmp["eer"].values.astype(float)
    if len(x) >= 8:
        coef_lin = np.polyfit(x, y, 1)
        pred_lin = np.polyval(coef_lin, x)
        mse_lin = float(np.mean((y - pred_lin)**2))
        best = (np.inf, None, None, None, None)
        # avoid tiny segments
        for k in range(3, len(x)-3):
            c1 = np.polyfit(x[:k], y[:k], 1)
            c2 = np.polyfit(x[k:], y[k:], 1)
            p1 = np.polyval(c1, x[:k])
            p2 = np.polyval(c2, x[k:])
            mse = float((np.sum((y[:k]-p1)**2) + np.sum((y[k:]-p2)**2)) / len(y))
            if mse < best[0]:
                best = (mse, k, c1, c2, float(x[k]))
        mse_pw, k, c1, c2, transition = best
        improvement_ratio = mse_lin / mse_pw if mse_pw > 0 else np.inf
        # permutation p-value
        rng = np.random.default_rng(123)
        B = 1000
        better = 0
        for _ in range(B):
            yp = rng.permutation(y)
            best_perm = np.inf
            for kk in range(3, len(x)-3):
                cc1 = np.polyfit(x[:kk], yp[:kk], 1)
                cc2 = np.polyfit(x[kk:], yp[kk:], 1)
                pp1 = np.polyval(cc1, x[:kk])
                pp2 = np.polyval(cc2, x[kk:])
                mse = float((np.sum((yp[:kk]-pp1)**2) + np.sum((yp[kk:]-pp2)**2)) / len(yp))
                best_perm = min(best_perm, mse)
            if best_perm <= mse_pw:
                better += 1
        perm_p = (better + 1) / (B + 1)
        xs1 = np.linspace(np.min(x), transition, 100)
        xs2 = np.linspace(transition, np.max(x), 100)
        plt.figure(figsize=(8,5))
        plt.scatter(x, y, alpha=0.7, label="Observed")
        plt.plot(xs1, np.polyval(c1, xs1), label="Phase 1")
        plt.plot(xs2, np.polyval(c2, xs2), linestyle="--", label="Phase 2")
        if np.isfinite(cdt_mean):
            plt.axvline(cdt_mean, linestyle="--", label=f"CDT ≈ {cdt_mean:.3f}")
        plt.axvline(transition, linestyle=":", label=f"Transition ≈ {transition:.3f}")
        plt.xlabel("Identity drift index")
        plt.ylabel("Equal Error Rate (EER)")
        plt.title("Two-Phase Degradation Model of EEG Biometric Drift")
        plt.legend()
        plt.grid(alpha=0.3)
        p = savefig("fig05_phase_transition_piecewise_model.png")
        manifest.append({"figure_file":p.name,"purpose":"Mathematically supports phase-transition behavior under cognitive drift."})
        pd.DataFrame([{
            "linear_mse": mse_lin, "piecewise_mse": mse_pw, "improvement_ratio": improvement_ratio,
            "best_split_index": int(k), "phase_transition_drift": transition,
            "cdt_mean": cdt_mean, "permutation_p_value": perm_p, "n_permutations": B
        }]).to_csv(TAB_DIR / "phase_transition_change_point_test.csv", index=False)

# 3.6 Drift severity group ablation
if cross_prop is not None and {"drift_index", "eer"}.issubset(cross_prop.columns):
    df = cross_prop[["drift_index", "eer"]].dropna().copy()
    if len(df) >= 9:
        df["drift_group"] = pd.qcut(df["drift_index"], 3, labels=["Low drift", "Medium drift", "High drift"])
        group = df.groupby("drift_group", observed=False)["eer"].agg(["count", "mean", "std"]).reset_index()
        group.to_csv(TAB_DIR / "drift_group_eer_ablation.csv", index=False)
        plt.figure(figsize=(7,5))
        xloc = np.arange(len(group))
        yerr = group["std"].fillna(0).values
        plt.bar(xloc, group["mean"].values, yerr=yerr, capsize=5)
        for i, row in group.iterrows():
            plt.text(i, row["mean"] + (row["std"] if np.isfinite(row["std"]) else 0) + 0.001, f"n={int(row['count'])}", ha="center", fontsize=9)
        plt.xticks(xloc, group["drift_group"], rotation=20, ha="right")
        plt.ylabel("Mean EER")
        plt.xlabel("Drift severity group")
        plt.title("Verification Error across Drift Severity Groups")
        plt.grid(axis="y", alpha=0.3)
        p = savefig("fig06_eer_across_drift_severity_groups.png")
        manifest.append({"figure_file":p.name,"purpose":"Ablation showing when cognitive drift degrades verification."})

# 3.7 Worst-case transitions
if cross_prop is not None and {"enroll_emotion", "test_emotion", "eer"}.issubset(cross_prop.columns):
    worst = cross_prop.copy()
    worst["transition"] = worst["enroll_emotion"].astype(str) + "→" + worst["test_emotion"].astype(str)
    worst = worst.groupby("transition")["eer"].mean().sort_values(ascending=False).head(10).reset_index()
    worst.to_csv(TAB_DIR / "worst_case_cross_emotion_transitions.csv", index=False)
    plt.figure(figsize=(9,4.5))
    plt.bar(range(len(worst)), worst["eer"])
    plt.xticks(range(len(worst)), worst["transition"], rotation=35, ha="right")
    plt.ylabel("Mean EER")
    plt.xlabel("Emotion transition")
    plt.title("Worst-Case Cross-Emotion Verification Conditions")
    plt.grid(axis="y", alpha=0.3)
    p = savefig("fig07_worst_case_cross_emotion_transitions.png")
    manifest.append({"figure_file":p.name,"purpose":"Highlights specific cognitive-state pairs causing highest biometric error."})

# 3.8 Subject-level stability boxplot
if drift_df is not None:
    ddf = drift_df.copy()
    if "drift_index" not in ddf.columns:
        for c in ddf.columns:
            if "drift" in c.lower():
                ddf = ddf.rename(columns={c:"drift_index"})
                break
    if "subject" in ddf.columns and "drift_index" in ddf.columns:
        subj = ddf.groupby("subject")["drift_index"].mean().dropna().values
        if len(subj) > 0:
            pd.DataFrame({"subject_level_mean_drift": subj}).to_csv(TAB_DIR / "subject_level_identity_drift_summary.csv", index=False)
            plt.figure(figsize=(6,5))
            plt.boxplot([subj], showmeans=True, tick_labels=["Subjects"])
            plt.ylabel("Subject-level mean identity drift")
            plt.title("Subject-wise Stability of EEG Identity Representations")
            plt.grid(axis="y", alpha=0.3)
            p = savefig("fig08_subjectwise_drift_boxplot.png")
            manifest.append({"figure_file":p.name,"purpose":"Shows inter-subject variability in representation stability."})

# 3.9 EER distribution over cross-emotion conditions
if cross_prop is not None and "eer" in cross_prop.columns:
    vals = cross_prop["eer"].dropna().astype(float).values
    if len(vals) > 0:
        plt.figure(figsize=(7,4.5))
        plt.hist(vals, bins=20, alpha=0.8)
        plt.axvline(np.mean(vals), linestyle="--", label=f"Mean = {np.mean(vals):.3f}")
        plt.axvline(np.median(vals), linestyle="-", label=f"Median = {np.median(vals):.3f}")
        plt.xlabel("Equal Error Rate (EER)")
        plt.ylabel("Number of cross-emotion conditions")
        plt.title("Distribution of Verification Error across Cognitive-State Transitions")
        plt.legend()
        plt.grid(alpha=0.3)
        p = savefig("fig09_eer_distribution_cross_emotion.png")
        manifest.append({"figure_file":p.name,"purpose":"Shows overall spread of cognitive-state verification difficulty."})

# ============================================================
# 4. SCORE-LEVEL ROC / DET / SCORE DISTRIBUTION IF AVAILABLE
# ============================================================
if score_df is not None:
    sdf = score_df.copy()
    # detect score and label columns
    score_col = None
    label_col = None
    for c in sdf.columns:
        cl = c.lower()
        if cl in ["score", "scores", "similarity", "similarity_score"]:
            score_col = c
        if cl in ["y_true", "label", "target", "is_genuine", "genuine"]:
            label_col = c
    # Sometimes table has type + score
    if score_col is None:
        for c in sdf.columns:
            if "score" in c.lower() or "similar" in c.lower():
                score_col = c
                break
    if label_col is None and "type" in sdf.columns:
        label_col = "type"
    if score_col is not None and label_col is not None:
        labels_raw = sdf[label_col]
        if labels_raw.dtype == object:
            y_true = labels_raw.astype(str).str.lower().str.contains("genuine|true|1").astype(int).values
        else:
            y_true = labels_raw.astype(int).values
        scores = sdf[score_col].astype(float).values
        eer, roc_auc, thr, curves = eer_auc_from_scores(y_true, scores)
        if curves is not None:
            fpr, tpr, fnr, thresholds = curves
            pd.DataFrame({"y_true": y_true, "score": scores}).to_csv(TAB_DIR / "score_level_genuine_impostor_scores_used.csv", index=False)
            pd.DataFrame([{"AUC":roc_auc,"EER":eer,"EER_threshold":thr,"total_scores":len(scores),"genuine_scores":int(y_true.sum()),"impostor_scores":int((1-y_true).sum())}]).to_csv(TAB_DIR / "score_level_roc_det_summary.csv", index=False)
            plt.figure(figsize=(7,5))
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}, EER={eer:.3f}")
            plt.plot([0,1],[0,1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve for EEG Biometric Verification")
            plt.legend()
            plt.grid(alpha=0.3)
            p = savefig("fig10_score_level_roc_curve.png")
            manifest.append({"figure_file":p.name,"purpose":"Score-level ROC validates verification behavior using genuine/impostor decisions."})
            plt.figure(figsize=(7,5))
            plt.plot(fpr, fnr, label=f"EER={eer:.3f}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("False Negative Rate")
            plt.title("DET Curve for EEG Biometric Verification")
            plt.legend()
            plt.grid(alpha=0.3)
            p = savefig("fig11_score_level_det_curve.png")
            manifest.append({"figure_file":p.name,"purpose":"DET curve provides biometric-system operating trade-off."})
            genuine = scores[y_true == 1]
            impostor = scores[y_true == 0]
            plt.figure(figsize=(8,5))
            plt.hist(impostor, bins=80, density=True, alpha=0.45, label="Impostor")
            plt.hist(genuine, bins=80, density=True, alpha=0.45, label="Genuine")
            plt.axvline(thr, linestyle="--", label=f"EER threshold = {thr:.3f}")
            plt.xlabel("Similarity score")
            plt.ylabel("Density")
            plt.title("Score Distribution: Genuine vs Impostor")
            plt.legend()
            plt.grid(alpha=0.3)
            p = savefig("fig12_score_distribution_genuine_vs_impostor.png")
            manifest.append({"figure_file":p.name,"purpose":"Shows genuine/impostor separability and threshold placement."})
            # threshold error tradeoff
            order = np.argsort(thresholds)
            plt.figure(figsize=(8,5))
            plt.plot(thresholds[order], fpr[order], label="False Positive Rate")
            plt.plot(thresholds[order], fnr[order], label="False Negative Rate")
            plt.axvline(thr, linestyle="--", label=f"EER threshold = {thr:.3f}")
            plt.xlabel("Decision threshold")
            plt.ylabel("Error rate")
            plt.title("Threshold vs Error Trade-off")
            plt.legend()
            plt.grid(alpha=0.3)
            p = savefig("fig13_threshold_error_tradeoff.png")
            manifest.append({"figure_file":p.name,"purpose":"Shows how threshold choice controls FPR/FNR trade-off."})
    else:
        print("Score table found, but could not detect score/label columns. Skipping ROC/DET.")
else:
    print("No score-level table found. ROC/DET needs genuine/impostor score-level CSV.")

# ============================================================
# 5. RUN03 IMPROVEMENT EXPERIMENT: MAHALANOBIS + ADAPTIVE THRESHOLD
# This section is self-contained and uses sessionwise NPZ files if present.
# It creates a new run folder and does not touch previous results.
# ============================================================
RUN03 = ROOT / "outputs/run_04_q1_complete_post_screenshot/run03_improvement_reproduction"
RUN03_FIG = RUN03 / "figures"
RUN03_TAB = RUN03 / "tables"
for d in [RUN03, RUN03_FIG, RUN03_TAB]:
    d.mkdir(parents=True, exist_ok=True)

SESSION_DIR = ROOT / "data/processed/sessionwise"
session_files = sorted(SESSION_DIR.glob("*.npz"))
session_files = [f for f in session_files if ("SESSION" in f.name.upper()) or f.name.lower().startswith("session")]
print("\nSessionwise files detected for Run03 improvement:", [f.name for f in session_files])

def get_key(npz, candidates):
    for c in candidates:
        if c in npz.files:
            return c
    return None

def extract_simple_features(X):
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Expected X: windows x channels x time. Use mean/std/bandlike simple time stats.
    mean = X.mean(axis=2)
    std = X.std(axis=2)
    mx = X.max(axis=2)
    mn = X.min(axis=2)
    feat = np.concatenate([mean, std, mx, mn], axis=1)
    # normalize rows
    feat = feat - feat.mean(axis=0, keepdims=True)
    feat = feat / (feat.std(axis=0, keepdims=True) + 1e-8)
    return feat.astype(np.float32)

if len(session_files) >= 3:
    loaded = []
    for f in session_files[:3]:
        z = np.load(f, allow_pickle=True)
        xkey = get_key(z, ["X", "x", "data", "eeg"])
        ykey = get_key(z, ["y", "subject", "subjects", "subject_id", "labels"])
        if xkey is None or ykey is None:
            print("Skipping Run03 improvement: could not identify X/y keys in", f.name, z.files)
            loaded = []
            break
        X = z[xkey]
        y = z[ykey]
        if y.ndim > 1:
            y = y.reshape(-1)
        loaded.append((f.name, X, y))
    if len(loaded) >= 3:
        print("Running Mahalanobis/adaptive-threshold reproduction on session1 enroll, session2+3 test")
        X_enroll = extract_simple_features(loaded[0][1])
        y_enroll = loaded[0][2]
        X_test = np.vstack([extract_simple_features(loaded[1][1]), extract_simple_features(loaded[2][1])])
        y_test = np.concatenate([loaded[1][2], loaded[2][2]])
        # keep common subjects
        common = np.intersect1d(np.unique(y_enroll), np.unique(y_test))
        mask_en = np.isin(y_enroll, common)
        mask_te = np.isin(y_test, common)
        X_enroll, y_enroll = X_enroll[mask_en], y_enroll[mask_en]
        X_test, y_test = X_test[mask_te], y_test[mask_te]
        subjects = np.unique(y_enroll)
        # prototypes and inverse covariance
        protos, invcovs = {}, {}
        for s in subjects:
            xs = X_enroll[y_enroll == s]
            protos[s] = xs.mean(axis=0)
            if len(xs) > 3:
                cov = np.cov(xs.T)
            else:
                cov = np.eye(xs.shape[1])
            cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
            cov = cov + np.eye(cov.shape[0]) * 1e-2
            invcovs[s] = np.linalg.pinv(cov)
        # balanced impostor sampling to avoid huge memory
        rng = np.random.default_rng(123)
        g_scores = []
        i_scores = []
        rows = []
        for idx, (xrow, yt) in enumerate(zip(X_test, y_test)):
            d_g = xrow - protos[yt]
            sg = -float(np.sqrt(np.maximum(d_g @ invcovs[yt] @ d_g.T, 0)))
            g_scores.append(sg)
            impostors = subjects[subjects != yt]
            # sample up to 15 impostors per test point
            if len(impostors) > 15:
                impostors = rng.choice(impostors, 15, replace=False)
            for s in impostors:
                d_i = xrow - protos[s]
                si = -float(np.sqrt(np.maximum(d_i @ invcovs[s] @ d_i.T, 0)))
                i_scores.append(si)
            if idx % 5000 == 0:
                print(" scored", idx, "/", len(X_test))
        g_scores = np.asarray(g_scores)
        i_scores = np.asarray(i_scores)
        y_true = np.concatenate([np.ones(len(g_scores)), np.zeros(len(i_scores))])
        scores = np.concatenate([g_scores, i_scores])
        eer, roc_auc, thr, curves = eer_auc_from_scores(y_true, scores)
        print("Run03 Mahalanobis AUC:", roc_auc, "EER:", eer, "threshold:", thr)
        pd.DataFrame({"y_true": y_true.astype(int), "score": scores}).to_csv(RUN03_TAB / "run03_mahalanobis_score_level_scores.csv", index=False)
        pd.DataFrame([{"method":"mahalanobis_adaptive_threshold", "AUC":roc_auc, "EER":eer, "threshold":thr, "genuine":len(g_scores), "impostor":len(i_scores)}]).to_csv(RUN03_TAB / "run03_mahalanobis_summary.csv", index=False)
        if curves is not None:
            fpr, tpr, fnr, thresholds = curves
            plt.figure(figsize=(7,5))
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}, EER={eer:.3f}")
            plt.plot([0,1],[0,1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Run03 Mahalanobis ROC Curve")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout(); plt.savefig(RUN03_FIG / "run03_mahalanobis_roc.png", dpi=300, bbox_inches="tight")
            plt.figure(figsize=(8,5))
            plt.hist(i_scores, bins=80, density=True, alpha=0.45, label="Impostor")
            plt.hist(g_scores, bins=80, density=True, alpha=0.45, label="Genuine")
            plt.axvline(thr, linestyle="--", label=f"Adaptive threshold={thr:.3f}")
            plt.xlabel("Mahalanobis score")
            plt.ylabel("Density")
            plt.title("Run03 Genuine vs Impostor Score Distribution")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout(); plt.savefig(RUN03_FIG / "run03_mahalanobis_score_distribution.png", dpi=300, bbox_inches="tight")
else:
    print("Run03 improvement skipped: sessionwise NPZ files not found.")

# ============================================================
# 6. DRIFT-INVARIANT LOSS CODE SNIPPET SAVED FOR MANUSCRIPT / NEXT TRAINING
# ============================================================
drift_loss_code = r'''
# Drift-invariant loss block for future model training
# Use this inside the training step after computing identity embeddings z_id
# and emotion labels e.

def pairwise_cosine_matrix(z):
    z = torch.nn.functional.normalize(z, dim=1)
    return z @ z.T

def drift_invariant_loss(z_id, y_subject, y_emotion, margin=0.10):
    """
    Penalizes same-subject embeddings from different emotions when they drift apart.
    z_id: identity embedding [B, D]
    y_subject: subject label [B]
    y_emotion: emotion/cognitive-state label [B]
    """
    sim = pairwise_cosine_matrix(z_id)
    same_subject = y_subject[:, None].eq(y_subject[None, :])
    diff_emotion = ~y_emotion[:, None].eq(y_emotion[None, :])
    mask = same_subject & diff_emotion
    if mask.sum() == 0:
        return z_id.new_tensor(0.0)
    # Want high cross-emotion similarity for the same subject.
    return torch.relu((1.0 - margin) - sim[mask]).mean()

# Example total loss:
# loss = arcface_loss + 0.5 * supcon_loss + 0.2 * drift_invariant_loss(z_id, subject, emotion)
'''
(Path(TAB_DIR) / "drift_invariant_loss_code_snippet.py").write_text(drift_loss_code)

# ============================================================
# 7. FINAL SUMMARY AND MANIFEST
# ============================================================
novelty_sentence = (
    "Existing EEG biometric studies primarily report aggregate verification performance, "
    "whereas this work explicitly models, quantifies, and statistically validates "
    "cognitive-state-induced identity drift."
)
summary_rows = []
if main_df is not None and "variant" in main_df.columns and "eer" in main_df.columns:
    s = main_df.groupby("variant").agg(mean_EER=("eer","mean"), std_EER=("eer","std"))
    if "auc" in main_df.columns:
        s_auc = main_df.groupby("variant").agg(mean_AUC=("auc","mean"), std_AUC=("auc","std"))
        s = s.join(s_auc)
    s = s.reset_index()
    s.to_csv(TAB_DIR / "model_performance_summary.csv", index=False)

pd.DataFrame(manifest).to_csv(TAB_DIR / "final_figure_manifest.csv", index=False)
(Path(TAB_DIR) / "novelty_positioning_sentence.txt").write_text(novelty_sentence + "\n")

final_report = {
    "root": str(ROOT),
    "previous_run_used": str(PREV_OUT),
    "new_output_folder": str(OUT),
    "best_variant": str(best_variant),
    "figures_generated": manifest,
    "novelty_sentence": novelty_sentence,
    "finished": datetime.now().isoformat()
}
(Path(OUT) / "run04_complete_summary.json").write_text(json.dumps(final_report, indent=2))

print("\n" + "=" * 80)
print("FINAL Q1 POST-SCREENSHOT SUITE COMPLETE")
print("=" * 80)
print("Figures saved to:", FIG_DIR)
print("Tables saved to:", TAB_DIR)
print("Run03 reproduction saved to:", RUN03)
print("Novelty sentence:")
print(novelty_sentence)
print("Generated figures:")
for m in manifest:
    print(" -", m["figure_file"], ":", m["purpose"])
print("=" * 80)
