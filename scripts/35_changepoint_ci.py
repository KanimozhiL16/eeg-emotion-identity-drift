#!/usr/bin/env python3
"""
35_changepoint_ci.py  --  REVIEWER FIX (change-point): finer permutation p, breakpoint CI, clean figure.
                          v2: SELF-LOCATING so it reproduces the manuscript's own change-point.

The manuscript reports the change-point on the identity-drift index (summed absolute band-power drift,
PSD units, NOT normalised) with breakpoint ~0.124 and mean ~0.116. Different CSVs under outputs/ carry
several "drift" columns on different scales; picking the wrong one gives a wrong breakpoint. So this
script SCANS every CSV, lists each (drift-column, EER-column) candidate with its n / mean / min-RSS
breakpoint, and AUTO-SELECTS the one that reproduces the manuscript (mean closest to 0.116 AND
breakpoint closest to 0.124). It then runs the reviewer-requested statistics on THAT column only:
  - permutation test (shuffle EER across conditions), 10,000 perms, one-sided  -> finer p
  - bootstrap 95% CI on the breakpoint LOCATION, 2,000 resamples
  - regenerated Fig 5b with unambiguous labels ("breakpoint" vs "mean drift"), replacing "CDT".

Model: continuous two-phase (piecewise-linear)  y ~ 1 + x + relu(x - c); breakpoint c by min RSS.
Outputs -> outputs/run_18_changepoint_ci/{candidates.csv, changepoint_ci.csv, Fig03b_change_point.png, log.txt}
USAGE:
  cd /home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
  source p4_seedv_env/bin/activate
  mkdir -p outputs/run_18_changepoint_ci
  python -u scripts/35_changepoint_ci.py 2>&1 | tee outputs/run_18_changepoint_ci/log.txt
"""
import os, glob, numpy as np, pandas as pd
import numpy.linalg as la
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

MS_MEAN, MS_BP = 0.116, 0.124          # manuscript reference values to lock onto
def _has(p): return os.path.isdir(os.path.join(p, "outputs"))
_hp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT = next((c for c in [os.getcwd(), _hp,
      "/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)), os.getcwd())
OUT = os.path.join(ROOT, "outputs", "run_18_changepoint_ci"); os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(123)

def fit_rss(x, y, c):
    X = np.c_[np.ones_like(x), x, np.clip(x - c, 0, None)]
    b, *_ = la.lstsq(X, y, rcond=None); r = y - X @ b; return float(r @ r), b
def line_rss(x, y):
    X = np.c_[np.ones_like(x), x]; b, *_ = la.lstsq(X, y, rcond=None); r = y - X @ b; return float(r @ r)
def best_breakpoint(x, y):
    cand = np.unique(np.quantile(x, np.linspace(0.10, 0.90, 60)))
    rss = [fit_rss(x, y, c)[0] for c in cand]; i = int(np.argmin(rss)); return float(cand[i]), rss[i]

# ---- scan every CSV for (drift, EER) candidate column pairs ----
DR = ("drift", "psd", "index"); ER = ("eer",)
def is_drift(c): cl=c.lower(); return any(k in cl for k in DR) and "auc" not in cl
def is_eer(c):   return c.lower() in ("eer","eer_mean","mean_eer") or c.lower().startswith("eer")
rows = []
for f in glob.glob(os.path.join(ROOT, "outputs", "**", "*.csv"), recursive=True):
    try: df = pd.read_csv(f)
    except Exception: continue
    dcols = [c for c in df.columns if is_drift(c) and pd.api.types.is_numeric_dtype(df[c])]
    ecols = [c for c in df.columns if is_eer(c)   and pd.api.types.is_numeric_dtype(df[c])]
    if not dcols or not ecols: continue
    for dc in dcols:
        for ec in ecols:
            d = df[[dc, ec]].dropna()
            if len(d) < 12: continue
            x = d[dc].to_numpy(float); y = d[ec].to_numpy(float); o = np.argsort(x)
            x, y = x[o], y[o]
            bp, rss_pw = best_breakpoint(x, y); rss_l = line_rss(x, y)
            rows.append({"file": os.path.relpath(f, ROOT), "drift_col": dc, "eer_col": ec,
                         "n": len(x), "mean_x": float(x.mean()), "breakpoint": bp,
                         "improve_pct": 100*(1 - rss_pw/rss_l),
                         "score": abs(x.mean()-MS_MEAN) + abs(bp-MS_BP)})
cand = pd.DataFrame(rows).sort_values("score")
cand.to_csv(os.path.join(OUT, "candidates.csv"), index=False)
print("="*94); print("CANDIDATE (drift,EER) COLUMN PAIRS  (target: mean~%.3f, breakpoint~%.3f)"%(MS_MEAN,MS_BP)); print("="*94)
print(cand.head(12).to_string(index=False,
      columns=["file","drift_col","eer_col","n","mean_x","breakpoint","improve_pct"]))
if cand.empty: raise SystemExit("No drift/EER candidate columns found under outputs/.")
best = cand.iloc[0]
print("\nSELECTED -> %s :: %s vs %s  (n=%d, mean=%.4f, breakpoint=%.4f)"
      % (best.file, best.drift_col, best.eer_col, best.n, best.mean_x, best.breakpoint))
if abs(best.mean_x-MS_MEAN) > 0.03 or abs(best.breakpoint-MS_BP) > 0.03:
    print("  !! WARNING: best match is still far from the manuscript's 0.116/0.124 -- inspect candidates.csv"
          "\n     before using these numbers; the correct source table may not be present.")

# ---- load selected column and run reviewer statistics ----
df = pd.read_csv(os.path.join(ROOT, best.file)); d = df[[best.drift_col, best.eer_col]].dropna().sort_values(best.drift_col)
x = d[best.drift_col].to_numpy(float); y = d[best.eer_col].to_numpy(float); n = len(x)
bp, rss_pw = best_breakpoint(x, y); rss_l = line_rss(x, y); improve = 1 - rss_pw/rss_l; obs = rss_l - rss_pw
print("\n"+"="*94); print(f"change-point on selected column | n={n} | breakpoint={bp:.4f} | improvement={100*improve:.1f}%")

NPERM = 10000; ge = 0
for _ in range(NPERM):
    yp = RNG.permutation(y); _, rp = best_breakpoint(x, yp)
    if (line_rss(x, yp) - rp) >= obs: ge += 1
p_perm = (1 + ge) / (NPERM + 1)
print(f"permutation p ({NPERM}) = {p_perm:.5f}  (exceedances={ge})")

NB = 2000; bps = np.empty(NB)
for j in range(NB):
    idx = RNG.integers(0, n, n); xs = x[idx]; ys = y[idx]; o = np.argsort(xs)
    bps[j], _ = best_breakpoint(xs[o], ys[o])
lo, hi = np.percentile(bps, [2.5, 97.5])
print(f"breakpoint 95% CI (bootstrap {NB}) = [{lo:.4f}, {hi:.4f}]")

pd.DataFrame([{"source": best.file, "drift_col": best.drift_col, "eer_col": best.eer_col,
               "n": n, "breakpoint": bp, "bp_ci_lo": lo, "bp_ci_hi": hi, "perm_p": p_perm,
               "n_perm": NPERM, "rss_improve_pct": 100*improve, "mean_drift": float(x.mean())}]
             ).to_csv(os.path.join(OUT, "changepoint_ci.csv"), index=False)

# ---- regenerate figure with unambiguous labels ----
_, bhat = fit_rss(x, y, bp); xs = np.linspace(x.min(), x.max(), 200)
yhat = np.c_[np.ones_like(xs), xs, np.clip(xs - bp, 0, None)] @ bhat
plt.figure(figsize=(5.2, 4))
plt.scatter(x, y, s=10, alpha=0.4, color="#3b6ea5", label="conditions")
plt.plot(xs, yhat, color="#c0392b", lw=2, label="two-phase fit")
plt.axvline(bp, ls=":", color="#c0392b", lw=2, label=f"breakpoint = {bp:.3f}")
plt.axvspan(lo, hi, color="#c0392b", alpha=0.12, label="breakpoint 95% CI")
plt.axvline(x.mean(), ls="--", color="0.4", lw=1.5, label=f"mean drift = {x.mean():.3f}")
plt.xlabel("Identity-drift index (summed band-power drift)"); plt.ylabel("Equal error rate (EER)")
plt.title(f"Two-phase change-point (permutation p = {p_perm:.4f})"); plt.legend(fontsize=7); plt.grid(alpha=.3)
plt.tight_layout(); figp = os.path.join(OUT, "Fig03b_change_point.png"); plt.savefig(figp, dpi=300); plt.close()
print("\nSAVED figure ->", os.path.relpath(figp, ROOT))
print("="*94)
print(f"READ-OFF: breakpoint {bp:.3f} (95% CI {lo:.3f}-{hi:.3f}); permutation p={p_perm:.4f}; "
      f"piecewise fit ~{100*improve:.0f}% lower RSS than a line.  mean drift={x.mean():.3f}.")
print("Confirm the SELECTED line above shows mean~0.116 & breakpoint~0.124 before I edit the paper.")
print("="*94)
