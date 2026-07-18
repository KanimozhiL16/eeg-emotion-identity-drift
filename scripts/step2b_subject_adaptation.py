#!/usr/bin/env python3
"""
step2b_subject_adaptation.py  -- H5 paired test (subject-adaptive thresholding).

run07 saved adaptation EER as a session x fraction grid (not paired by subject),
so a paired test isn't possible from it. This script recomputes a clean,
leakage-free per-subject comparison from the score-level file:

  For each probe session and each enrolled (claimed) identity:
    * split that identity's genuine+impostor scores 50/50 (stratified) into
      calibration / test  (fixed seed)
    * GLOBAL threshold  = EER threshold from POOLED calibration of all subjects
    * SUBJECT threshold = EER threshold from that subject's own calibration
    * evaluate balanced error (FMR+FNMR)/2 on the subject's TEST half
  -> paired Wilcoxon (global vs subject-adaptive) across the 16 subjects.

This is a transparent, reviewer-friendly operationalisation of "subject-adaptive
thresholds reduce drift-induced error" (Hypothesis H5).

USAGE (from project root, env active):
    cd /home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
    source p4_seedv_env/bin/activate
    python -u scripts/step2b_subject_adaptation.py 2>&1 | tee outputs/run_11_statistics/step2b_log.txt
"""
import os, glob, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score

def has_outputs(p): return os.path.isdir(os.path.join(p, "outputs"))
_hp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT = next((c for c in [os.getcwd(), _hp,
            "/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
            "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if has_outputs(c)), os.getcwd())
OUT = os.path.join(ROOT, "outputs", "run_11_statistics"); os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)
print("="*78); print("STEP 2b  subject-adaptive threshold (H5)  | root:", ROOT); print("="*78)

f = glob.glob(os.path.join(ROOT, "outputs", "**",
              "score_level_session_verification_scores.csv"), recursive=True)
if not f:
    raise SystemExit("score_level_session_verification_scores.csv not found")
df = pd.read_csv(sorted(f, key=len)[0])
print("columns:", list(df.columns))
CY, CS, CSUB, CSES = "y_true", "score", "claimed_subject", "probe_session"

def eer_threshold(y, s):
    fpr, tpr, thr = roc_curve(y, s); fnr = 1 - tpr
    return thr[np.nanargmin(np.abs(fpr - fnr))]

def balanced_error(y, s, t):
    y = np.asarray(y); pred = (np.asarray(s) >= t).astype(int)
    imp, gen = y == 0, y == 1
    fmr  = pred[imp].mean() if imp.any() else np.nan          # impostor accepted
    fnmr = (1 - pred[gen]).mean() if gen.any() else np.nan    # genuine rejected
    return np.nanmean([fmr, fnmr])

def split(idx, y):
    """stratified 50/50 calib/test indices."""
    calib, test = [], []
    for lab in (0, 1):
        ii = idx[y[idx] == lab]; RNG.shuffle(ii)
        h = len(ii)//2; calib += list(ii[:h]); test += list(ii[h:])
    return np.array(calib), np.array(test)

rows = []
for ses, g in df.groupby(CSES):
    g = g.reset_index(drop=True)
    y = g[CY].values.astype(int); s = g[CS].values.astype(float)
    # orient score so higher = genuine
    if roc_auc_score(y, s) < 0.5: s = -s
    # pooled calib (per-subject stratified) -> global threshold
    calib_all = []
    per_subj = {}
    for sub, sg in g.groupby(CSUB):
        idx = sg.index.values
        c, t = split(idx, y)
        if (y[t] == 1).sum() < 2 or (y[t] == 0).sum() < 2: continue
        per_subj[sub] = (c, t); calib_all += list(c)
    calib_all = np.array(calib_all)
    gthr = eer_threshold(y[calib_all], s[calib_all])
    for sub, (c, t) in per_subj.items():
        sthr = eer_threshold(y[c], s[c]) if (y[c]==1).sum()>=2 and (y[c]==0).sum()>=2 else gthr
        be_g = balanced_error(y[t], s[t], gthr)
        be_s = balanced_error(y[t], s[t], sthr)
        rows.append({"session": ses, "subject": sub,
                     "balanced_error_global": be_g, "balanced_error_subject": be_s})

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "subject_level_adaptation_eer.csv"), index=False)
print("\nper-subject rows:", len(res))

def report(d, label):
    a, b = d["balanced_error_global"].values, d["balanced_error_subject"].values
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    if len(a) < 3: print(f"  [{label}] too few subjects"); return
    w, p = stats.wilcoxon(a, b)
    print(f"  [{label}] n={len(a)} | global={a.mean():.4f} -> subject={b.mean():.4f} "
          f"| mean reduction={ (a-b).mean():.4f} ({100*(a-b).mean()/a.mean():.1f}%) "
          f"| Wilcoxon W={w:.1f}, p={p:.4g}")

print("\n--- H5 paired Wilcoxon (global vs subject-adaptive threshold) ---")
for ses in sorted(res["session"].unique()):
    report(res[res.session == ses], f"session {ses}")
report(res, "pooled (all sessions)")
print("\nSAVED:", os.path.relpath(os.path.join(OUT, "subject_level_adaptation_eer.csv"), ROOT))
print("="*78)
