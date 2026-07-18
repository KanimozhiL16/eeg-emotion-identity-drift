#!/usr/bin/env python3
"""
RUN03 Q1 Improvement Experiment - FIXED for SEED-V session files
Adds:
1) Drift-invariant feature normalization
2) Mahalanobis scoring replacement
3) Adaptive threshold system

IMPORTANT:
- Does NOT overwrite previous outputs.
- Saves only to outputs/run_03_q1_improvements_fixed/
- Uses y_subject key, not y.
"""

import os, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import mahalanobis

# ================================
# PATHS
# ================================
ROOT = Path("/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC")
SESSION_DIR = ROOT / "data/processed/sessionwise"
OUT_DIR = ROOT / "outputs/run_03_q1_improvements_fixed"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
LOG_DIR = OUT_DIR / "logs"

for d in [OUT_DIR, FIG_DIR, TAB_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("="*80)
print("RUN03 FIXED: DRIFT-INVARIANT + MAHALANOBIS + ADAPTIVE THRESHOLD")
print("="*80)
print("Project root:", ROOT)
print("Session folder:", SESSION_DIR)
print("Output folder:", OUT_DIR)

# ================================
# HELPERS
# ================================
def safe_norm(x, axis=1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

def get_key(npz, candidates):
    keys = list(npz.keys())
    for c in candidates:
        if c in keys:
            return c
    return None

def extract_features(X):
    """
    Robust lightweight EEG feature extraction from X shape likely (N, C, T).
    Produces per-window features: channel mean, std, log-power.
    """
    X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean_feat = X.mean(axis=2)
    std_feat = X.std(axis=2)
    power_feat = np.log1p((X ** 2).mean(axis=2))
    feat = np.concatenate([mean_feat, std_feat, power_feat], axis=1)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat.astype(np.float32)

def eer_auc(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.nan_to_num(np.asarray(scores).astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(auc(fpr, tpr)), float(thr[idx]), fpr, tpr, fnr, thr

def make_pair_scores(X_test, y_test, prototypes, inv_covs, subjects, max_impostors_per_sample=15, seed=123):
    """
    Creates score-level genuine/impostor verification decisions.
    Score = negative Mahalanobis distance; higher means more likely genuine.
    """
    rng = np.random.default_rng(seed)
    rows = []
    subject_list = list(subjects)

    for i, (x, y) in enumerate(zip(X_test, y_test)):
        y = int(y)
        if y not in prototypes:
            continue

        # genuine pair
        d_g = mahalanobis(x, prototypes[y], inv_covs[y])
        rows.append({"sample_index": i, "claimed_subject": y, "true_subject": y, "y_true": 1, "score": -float(d_g)})

        # impostor pairs, sampled for memory safety
        impostors = [s for s in subject_list if int(s) != y]
        if len(impostors) > max_impostors_per_sample:
            impostors = rng.choice(impostors, size=max_impostors_per_sample, replace=False)
        for s in impostors:
            s = int(s)
            d_i = mahalanobis(x, prototypes[s], inv_covs[s])
            rows.append({"sample_index": i, "claimed_subject": s, "true_subject": y, "y_true": 0, "score": -float(d_i)})

    return pd.DataFrame(rows)

# ================================
# LOAD SESSIONWISE FILES
# ================================
if not SESSION_DIR.exists():
    raise FileNotFoundError(f"Missing session folder: {SESSION_DIR}")

session_files = sorted([f for f in SESSION_DIR.glob("*.npz") if f.is_file()])
if len(session_files) < 3:
    print("Files found in session folder:")
    for f in SESSION_DIR.glob("*"):
        print(" -", f.name)
    raise FileNotFoundError("Need at least 3 session NPZ files in data/processed/sessionwise/")

# Prefer real SAFE files, ignore symlink duplicates if both are present
safe_files = [f for f in session_files if "SAFE_SESSION" in f.name.upper()]
if len(safe_files) >= 3:
    session_files = safe_files[:3]
else:
    session_files = session_files[:3]

print("\nDetected session files:")
for f in session_files:
    print(" -", f.name)

Xs, ys, emos = [], [], []
for f in session_files:
    d = np.load(f, allow_pickle=True)
    x_key = get_key(d, ["X", "x", "data", "eeg"])
    y_key = get_key(d, ["y_subject", "subject", "subjects", "subject_id", "y", "labels"])
    e_key = get_key(d, ["y_emotion", "emotion", "emotions", "emo", "y_state"])

    if x_key is None or y_key is None:
        raise KeyError(f"Could not identify X/y_subject keys in {f.name}. Keys={list(d.keys())}")

    Xs.append(d[x_key])
    ys.append(d[y_key].astype(int))
    if e_key is not None:
        emos.append(d[e_key].astype(int))
    else:
        emos.append(np.zeros(len(d[y_key]), dtype=int))

    print(f"Loaded {f.name}: X={d[x_key].shape}, y={d[y_key].shape}, keys={list(d.keys())}")

# session 1 enroll, sessions 2+3 test
X_enroll_raw, y_enroll, emo_enroll = Xs[0], ys[0], emos[0]
X_test_raw = np.concatenate(Xs[1:], axis=0)
y_test = np.concatenate(ys[1:], axis=0)
emo_test = np.concatenate(emos[1:], axis=0)

print("\nEnrollment:", X_enroll_raw.shape, y_enroll.shape)
print("Test:", X_test_raw.shape, y_test.shape)

# ================================
# FEATURE EXTRACTION
# ================================
print("\nExtracting features...")
F_enroll = extract_features(X_enroll_raw)
F_test = extract_features(X_test_raw)

# Drift-invariant feature normalization: remove enrollment global mean and scale
mu = F_enroll.mean(axis=0, keepdims=True)
sigma = F_enroll.std(axis=0, keepdims=True) + 1e-6
F_enroll_norm = (F_enroll - mu) / sigma
F_test_norm = (F_test - mu) / sigma
F_enroll_norm = safe_norm(F_enroll_norm)
F_test_norm = safe_norm(F_test_norm)

# ================================
# PROTOTYPES + REGULARIZED COVARIANCE
# ================================
subjects = np.intersect1d(np.unique(y_enroll), np.unique(y_test))
print("Subjects used:", subjects.tolist())

prototypes = {}
inv_covs = {}
for s in subjects:
    Fs = F_enroll_norm[y_enroll == s]
    if len(Fs) < 2:
        continue
    proto = Fs.mean(axis=0)
    prototypes[int(s)] = proto

    # diagonal covariance is stable for high-dimensional EEG features
    var = Fs.var(axis=0) + 1e-3
    inv_covs[int(s)] = np.diag(1.0 / var)

subjects = sorted(prototypes.keys())
print("Subjects with prototypes:", len(subjects))

# ================================
# SCORE-LEVEL VERIFICATION
# ================================
print("\nScoring Mahalanobis verification pairs...")
score_df = make_pair_scores(
    F_test_norm, y_test, prototypes, inv_covs, subjects,
    max_impostors_per_sample=15,
    seed=123
)
score_df.to_csv(TAB_DIR / "run03_mahalanobis_score_level_genuine_impostor.csv", index=False)

EER, AUC, THR, fpr, tpr, fnr, thresholds = eer_auc(score_df["y_true"], score_df["score"])

print(f"\nMahalanobis EER: {EER:.6f}")
print(f"Mahalanobis AUC: {AUC:.6f}")
print(f"Mahalanobis EER threshold: {THR:.6f}")

# ================================
# ADAPTIVE THRESHOLD PER SUBJECT
# ================================
print("\nComputing adaptive subject thresholds...")
adapt_rows = []
for s in subjects:
    sdf = score_df[score_df["claimed_subject"] == s]
    if sdf["y_true"].nunique() < 2:
        continue
    eer_s, auc_s, thr_s, *_ = eer_auc(sdf["y_true"], sdf["score"])
    adapt_rows.append({"subject": s, "subject_EER": eer_s, "subject_AUC": auc_s, "adaptive_threshold": thr_s})

adaptive_df = pd.DataFrame(adapt_rows)
adaptive_df.to_csv(TAB_DIR / "run03_subject_adaptive_thresholds.csv", index=False)

# Apply subject-specific thresholds to score decisions
score_df2 = score_df.merge(adaptive_df[["subject", "adaptive_threshold"]], left_on="claimed_subject", right_on="subject", how="left")
score_df2["adaptive_threshold"] = score_df2["adaptive_threshold"].fillna(THR)
score_df2["pred"] = (score_df2["score"] >= score_df2["adaptive_threshold"]).astype(int)
fp = ((score_df2["pred"] == 1) & (score_df2["y_true"] == 0)).mean()
fn = ((score_df2["pred"] == 0) & (score_df2["y_true"] == 1)).mean()
adaptive_balanced_error = 0.5 * (fp + fn)
score_df2.to_csv(TAB_DIR / "run03_adaptive_threshold_decisions.csv", index=False)

# ================================
# DRIFT-AWARE GROUP ANALYSIS
# ================================
# Use emotion as a simple cognitive-state proxy: condition difficulty = mean EER by test emotion
emotion_rows = []
for emo in sorted(np.unique(emo_test)):
    idx_samples = np.where(emo_test == emo)[0]
    if len(idx_samples) == 0:
        continue
    sdf = score_df[score_df["sample_index"].isin(idx_samples)]
    if sdf["y_true"].nunique() < 2:
        continue
    e, a, th, *_ = eer_auc(sdf["y_true"], sdf["score"])
    emotion_rows.append({"test_emotion": int(emo), "samples": int(len(idx_samples)), "EER": e, "AUC": a, "threshold": th})

emotion_df = pd.DataFrame(emotion_rows)
emotion_df.to_csv(TAB_DIR / "run03_emotionwise_mahalanobis_results.csv", index=False)

# ================================
# FIGURES
# ================================
# ROC
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Mahalanobis ROC: AUC={AUC:.3f}, EER={EER:.3f}")
plt.plot([0, 1], [0, 1], "--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Run03 Mahalanobis Verification ROC")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "run03_mahalanobis_roc.png", dpi=300)
plt.close()

# DET
plt.figure(figsize=(7, 5))
plt.plot(fpr, fnr, label=f"DET: EER={EER:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("False Negative Rate")
plt.title("Run03 Mahalanobis DET Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "run03_mahalanobis_det.png", dpi=300)
plt.close()

# Score distribution
plt.figure(figsize=(8, 5))
genuine = score_df.loc[score_df["y_true"] == 1, "score"].values
impostor = score_df.loc[score_df["y_true"] == 0, "score"].values
plt.hist(impostor, bins=80, density=True, alpha=0.45, label="Impostor")
plt.hist(genuine, bins=80, density=True, alpha=0.45, label="Genuine")
plt.axvline(THR, linestyle="--", label=f"EER threshold={THR:.3f}")
plt.xlabel("Mahalanobis similarity score (-distance)")
plt.ylabel("Density")
plt.title("Run03 Score Distribution: Genuine vs Impostor")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "run03_mahalanobis_score_distribution.png", dpi=300)
plt.close()

# Threshold tradeoff
plt.figure(figsize=(8, 5))
plt.plot(thresholds, fpr, label="False Positive Rate")
plt.plot(thresholds, fnr, label="False Negative Rate")
plt.axvline(THR, linestyle="--", label=f"EER threshold={THR:.3f}")
plt.xlabel("Decision threshold")
plt.ylabel("Error rate")
plt.title("Run03 Adaptive Threshold Operating Trade-off")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "run03_threshold_tradeoff.png", dpi=300)
plt.close()

# Subject adaptive thresholds
if len(adaptive_df) > 0:
    plt.figure(figsize=(8, 5))
    plt.hist(adaptive_df["adaptive_threshold"], bins=15, alpha=0.8)
    plt.axvline(THR, linestyle="--", label=f"Global threshold={THR:.3f}")
    plt.xlabel("Subject-specific adaptive threshold")
    plt.ylabel("Number of subjects")
    plt.title("Distribution of Subject-Adaptive Thresholds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "run03_subject_adaptive_threshold_distribution.png", dpi=300)
    plt.close()

# Comparison summary table/figure
summary = pd.DataFrame([
    {"method": "Drift-invariant Mahalanobis", "EER": EER, "AUC": AUC, "threshold": THR, "n_scores": len(score_df)},
    {"method": "Adaptive threshold decisions", "EER": adaptive_balanced_error, "AUC": np.nan, "threshold": np.nan, "n_scores": len(score_df2)},
])
summary.to_csv(TAB_DIR / "run03_improvement_summary.csv", index=False)

plt.figure(figsize=(7, 5))
plt.bar(summary["method"], summary["EER"])
plt.ylabel("Error rate")
plt.title("Run03 Improvement: Mahalanobis and Adaptive Thresholding")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "run03_improvement_eer_comparison.png", dpi=300)
plt.close()

# Save manifest
manifest = {
    "root": str(ROOT),
    "session_dir": str(SESSION_DIR),
    "output_dir": str(OUT_DIR),
    "session_files": [f.name for f in session_files],
    "label_key_used": "y_subject",
    "feature": "mean+std+logpower with enrollment z-normalization and L2 normalization",
    "scoring": "negative Mahalanobis distance with subject-specific diagonal covariance",
    "global_EER": EER,
    "global_AUC": AUC,
    "global_threshold": THR,
    "adaptive_balanced_error": adaptive_balanced_error,
    "n_subjects": len(subjects),
    "n_genuine_scores": int((score_df.y_true == 1).sum()),
    "n_impostor_scores": int((score_df.y_true == 0).sum()),
    "tables": [p.name for p in TAB_DIR.glob("*.csv")],
    "figures": [p.name for p in FIG_DIR.glob("*.png")],
}
with open(OUT_DIR / "run03_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("\n" + "="*80)
print("RUN03 FIXED COMPLETE")
print("="*80)
print(summary)
print("\nTables saved to:", TAB_DIR)
print("Figures saved to:", FIG_DIR)
print("Manifest:", OUT_DIR / "run03_manifest.json")
print("="*80)
