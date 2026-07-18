#!/usr/bin/env python3
"""
34_effect_cis.py  --  REVIEWER FIX M6 (uncertainty on the load-bearing n=16 effects).

Adds bootstrap 95% CIs to the three numbers the manuscript currently states as point estimates:
  (1) mean per-participant cross-session EER and its coefficient of variation (CV),
  (2) the per-participant EER spread (SD),
  (3) the adaptive re-enrolment reduction (absolute and relative), per future session.

Reads existing outputs (no re-computation of EER):
  outputs/run_17_subject_robustness/subject_robustness.csv   (subject, EER_S1toS2, EER_S1toS3, mean_cross)
  outputs/run_16_adaptation_significance/persubject.csv       (session, subject, EER_baseline, EER_adapted)

Outputs -> outputs/run_21_effect_cis/effect_cis.csv, log.txt
USAGE:
  cd /home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
  source p4_seedv_env/bin/activate
  # ensure subject-robustness table exists (quick):
  python -u scripts/subject_robustness.py 2>&1 | tail -5
  mkdir -p outputs/run_21_effect_cis
  python -u scripts/34_effect_cis.py 2>&1 | tee outputs/run_21_effect_cis/log.txt
"""
import os, glob, numpy as np, pandas as pd

def _has(p): return os.path.isdir(os.path.join(p, "outputs"))
_hp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT = next((c for c in [os.getcwd(), _hp,
      "/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)), os.getcwd())
OUT = os.path.join(ROOT, "outputs", "run_21_effect_cis"); os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42); NB = 10000
print("=" * 78); print("M6  bootstrap 95% CIs for the n=16 effects"); print("=" * 78)

def find(*globs):
    for g in globs:
        h = glob.glob(os.path.join(ROOT, "outputs", "**", g), recursive=True)
        if h: return sorted(h, key=len)[0]
    return None

def ci(vals, stat, nb=NB):
    vals = np.asarray(vals, float); n = len(vals)
    bs = [stat(vals[RNG.integers(0, n, n)]) for _ in range(nb)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); return stat(vals), lo, hi

rows = []
def rec(name, point, lo, hi, extra=""):
    rows.append({"quantity": name, "estimate": point, "ci_lo": lo, "ci_hi": hi, "detail": extra})
    print(f"  {name:38s} = {point:.4f}  95%CI[{lo:.4f}, {hi:.4f}]  {extra}")

# ---- (1)+(2) subject robustness ----
sr = find("subject_robustness.csv")
if sr:
    d = pd.read_csv(sr)
    col = "mean_cross" if "mean_cross" in d.columns else \
          [c for c in d.columns if "mean" in c.lower()][0] if any("mean" in c.lower() for c in d.columns) else None
    if col is None:  # fall back: average the two session columns
        sc = [c for c in d.columns if "eer" in c.lower()]
        d["mean_cross"] = d[sc].mean(1); col = "mean_cross"
    mc = d[col].values
    print(f"\nsubject robustness ({sr.split('outputs/')[-1]}, n={len(mc)}):")
    rec("mean cross-session EER", *ci(mc, np.mean))
    rec("SD of cross-session EER", *ci(mc, lambda v: v.std(ddof=0)))
    rec("coefficient of variation", *ci(mc, lambda v: v.std(ddof=0)/v.mean()))
    print(f"  observed range: [{mc.min():.4f}, {mc.max():.4f}]  (min=S{int(d.iloc[mc.argmin()].get('subject',-1))}, max=S{int(d.iloc[mc.argmax()].get('subject',-1))})")
else:
    print("\n  subject_robustness.csv NOT found -> run scripts/subject_robustness.py first.")

# ---- (3) adaptation reduction ----
ad = find("persubject.csv")
# make sure it's the adaptation one (has EER_baseline/EER_adapted)
cands = glob.glob(os.path.join(ROOT, "outputs", "**", "persubject.csv"), recursive=True)
ad = None
for c in cands:
    try:
        h = pd.read_csv(c, nrows=1)
        if {"EER_baseline", "EER_adapted"}.issubset(h.columns): ad = c; break
    except Exception: pass
if ad:
    a = pd.read_csv(ad)
    print(f"\nadaptation reduction ({ad.split('outputs/')[-1]}):")
    for s in sorted(a["session"].unique()):
        g = a[a.session == s]
        absr = (g["EER_baseline"].values - g["EER_adapted"].values)
        relr = 100 * absr / g["EER_baseline"].values
        rec(f"session {int(s)} abs. reduction", *ci(absr, np.mean), f"n={len(g)}")
        rec(f"session {int(s)} rel. reduction (%)", *ci(relr, np.mean))
else:
    print("\n  adaptation persubject.csv NOT found -> run scripts/step2d_adaptation_significance.py first.")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "effect_cis.csv"), index=False)
print("\nSAVED ->", os.path.relpath(os.path.join(OUT, "effect_cis.csv"), ROOT)); print("=" * 78)
