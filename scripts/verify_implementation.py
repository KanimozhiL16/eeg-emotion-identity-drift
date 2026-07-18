#!/usr/bin/env python3
"""
verify_implementation.py
------------------------
One-command integrity & reproducibility verifier for the
P4_SEEDV_COGNITIVE_BIOMETRIC project.

Runs four independent checks and prints a PASS/FAIL report:
  1. DATA INTEGRITY   - the processed SEED-V is complete (16 subjects, 114,144 windows)
  2. METRIC RECOMPUTE - RUN03 EER/AUC recomputed from raw genuine/impostor scores
  3. SAVED RESULTS    - RUN07 session-drift numbers match the saved tables
  4. PROVENANCE       - manifest + evidence (hashes, environment) exist

Usage (from the project root, inside the env):
    cd /home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
    source p4_seedv_env/bin/activate
    python scripts/verify_implementation.py

Exit code 0 = all checks passed, 1 = at least one failed.
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

# ---- locate project root robustly --------------------------------------------
# The folder may be reached via several paths (e.g. /home/nvidia/... is a symlink
# to the physical /lp-dev/... on Brev). We do NOT care which path string shows;
# we pick whichever candidate actually CONTAINS the processed data.
def _has_data(p):
    return os.path.isdir(os.path.join(p, "data", "processed", "sessionwise"))

_here_parent = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

CANDIDATES = [
    os.getcwd(),                                                   # where you ran it
    _here_parent,                                                  # script's parent
    "/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",        # logical home path
    "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",             # physical disk path
]

ROOT = next((c for c in CANDIDATES if _has_data(c)), None)
if ROOT is None:
    print("ERROR: could not locate the project data folder "
          "(data/processed/sessionwise) in any known path:")
    for c in CANDIDATES:
        print("   -", c)
    print("Run this script from the project root, e.g.:")
    print("   cd /home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC")
    print("   python scripts/verify_implementation.py")
    sys.exit(1)

TOL = 1e-3            # tolerance for floating-point metric comparison
results = []          # (check_name, passed_bool, detail_str)


def check(name, passed, detail=""):
    results.append((name, bool(passed), detail))
    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] {name}" + (f"  --  {detail}" if detail else ""))


def approx(a, b, tol=TOL):
    return abs(float(a) - float(b)) <= tol


print("=" * 78)
print("P4_SEEDV_COGNITIVE_BIOMETRIC  --  IMPLEMENTATION VERIFIER")
print("Project root:", ROOT)
print("=" * 78)

# ---------------------------------------------------------------------------
# CHECK 1 - DATA INTEGRITY
# ---------------------------------------------------------------------------
try:
    d = os.path.join(ROOT, "data", "processed", "sessionwise")
    files = sorted(glob.glob(os.path.join(d, "*16sub.npz")))
    total = 0
    subj_ok = True
    shape_ok = True
    for f in files:
        z = np.load(f, allow_pickle=True)
        X, ys = z["X"], z["y_subject"]
        total += X.shape[0]
        if sorted(np.unique(ys).tolist()) != list(range(1, 17)):
            subj_ok = False
        if X.shape[1:] != (62, 400):
            shape_ok = False
    ok = (len(files) == 3) and subj_ok and shape_ok and (total == 114144)
    check("1. DATA INTEGRITY", ok,
          f"{len(files)} files, total_windows={total}, subjects_1_16={subj_ok}, "
          f"shape_62x400={shape_ok}")
except Exception as e:
    check("1. DATA INTEGRITY", False, f"error: {e}")

# ---------------------------------------------------------------------------
# CHECK 2 - METRIC RECOMPUTE (RUN03 Mahalanobis EER / AUC from raw scores)
# ---------------------------------------------------------------------------
try:
    run03 = os.path.join(ROOT, "outputs", "run_03_q1_improvements_fixed")
    with open(os.path.join(run03, "run03_manifest.json")) as fh:
        man = json.load(fh)
    rep_eer = man["global_EER"]
    rep_auc = man["global_AUC"]

    df = pd.read_csv(os.path.join(
        run03, "tables", "run03_mahalanobis_score_level_genuine_impostor.csv"))
    fpr, tpr, _ = roc_curve(df["y_true"], df["score"])
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[i] + fnr[i]) / 2.0
    auc_val = auc(fpr, tpr)

    ok = approx(eer, rep_eer) and approx(auc_val, rep_auc)
    check("2. METRIC RECOMPUTE (RUN03)", ok,
          f"EER {eer:.4f} vs {rep_eer:.4f} | AUC {auc_val:.4f} vs {rep_auc:.4f} | "
          f"genuine={int((df.y_true==1).sum())}, impostor={int((df.y_true==0).sum())}")
except Exception as e:
    check("2. METRIC RECOMPUTE (RUN03)", False, f"error: {e}")

# ---------------------------------------------------------------------------
# CHECK 3 - SAVED RESULTS (RUN07 session-drift table sanity)
# ---------------------------------------------------------------------------
try:
    run07 = os.path.join(ROOT, "outputs",
                         "run_07_seedv_session_drift_subject_adaptation", "tables")
    sv = pd.read_csv(os.path.join(run07, "session_verification_results.csv"))
    # locate the cross-session rows by probe_session
    eer_s2 = float(sv.loc[sv.probe_session == 2, "EER"].iloc[0])
    eer_s3 = float(sv.loc[sv.probe_session == 3, "EER"].iloc[0])
    # monotonic degradation across sessions is the core claim
    eer_s1 = float(sv.loc[sv.probe_session == 1, "EER"].iloc[0])
    monotonic = eer_s1 < eer_s2 < eer_s3
    ok = approx(eer_s2, 0.181822) and approx(eer_s3, 0.250755) and monotonic
    check("3. SAVED RESULTS (RUN07)", ok,
          f"S1={eer_s1:.4f} < S2={eer_s2:.4f} < S3={eer_s3:.4f} (monotonic={monotonic})")
except Exception as e:
    check("3. SAVED RESULTS (RUN07)", False, f"error: {e}")

# ---------------------------------------------------------------------------
# CHECK 4 - PROVENANCE (manifest + evidence files present)
# ---------------------------------------------------------------------------
try:
    must_exist = [
        os.path.join(ROOT, "outputs", "run_03_q1_improvements_fixed",
                     "run03_manifest.json"),
        os.path.join(ROOT, "evidence", "file_hashes"),
        os.path.join(ROOT, "evidence", "environment"),
    ]
    missing = [p for p in must_exist if not os.path.exists(p)]
    ok = len(missing) == 0
    check("4. PROVENANCE", ok,
          "all present" if ok else f"missing: {missing}")
except Exception as e:
    check("4. PROVENANCE", False, f"error: {e}")

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print("=" * 78)
passed = sum(1 for _, ok, _ in results if ok)
for name, ok, _ in results:
    print(f"   {'PASS' if ok else 'FAIL'}  -  {name}")
print("-" * 78)
all_ok = passed == len(results)
print(f"OVERALL: {passed}/{len(results)} checks passed  ->  "
      f"{'IMPLEMENTATION VERIFIED' if all_ok else 'VERIFICATION INCOMPLETE'}")
print("=" * 78)
sys.exit(0 if all_ok else 1)
