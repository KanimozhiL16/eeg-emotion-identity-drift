#!/usr/bin/env python3
"""
31_mixed_effects_mechanism.py  --  REVIEWER FIX M1 (pseudoreplication).

The manuscript's Table 3 (theta beta=+0.38, p=6.4e-9, R2=0.152) is a STANDARDIZED OLS
over the 250 rows of run_06/.../merged_global_psd_identity_eer.csv. Those rows are NOT
independent: each row is one (session-transition x emotion-pair x seed) condition, EER
pooled over all 16 subjects, so 250 = 125 unique (transition x emotion-pair) x 2 seeds.
The same emotion-pair, session-transition and seed recur across many rows, so plain OLS /
Pearson p-values overstate significance (pseudoreplication).

This script re-tests the theta->EER effect while respecting that structure:
  (0) reproduces the standardized OLS  (must match Table 3: theta~+0.38)
  (1) PRIMARY: linear mixed-effects model, random intercept for emotion-pair
        EER ~ theta+alpha+beta+gamma + (1|emotion_pair)
  (1b) same with random intercept for session-transition
  (2) cluster-robust OLS, SE clustered on emotion-pair and on transition
  (3) seed-collapsed OLS (mean over seeds -> 125 unique conditions)
  (4) block-permutation test for theta (permute EER within transition blocks)
  (5) theta beyond alpha, as a nested likelihood-ratio test under the mixed model
All predictors and EER are z-scored exactly as in step4_biological.py (std, ddof=0).

Outputs -> outputs/run_18_mixedeffects_mechanism/{mechanism_robust.csv, mixedlm_summary.txt, log.txt}

USAGE:
  cd /home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
  source p4_seedv_env/bin/activate
  pip install -q statsmodels --break-system-packages
  mkdir -p outputs/run_18_mixedeffects_mechanism
  python -u scripts/31_mixed_effects_mechanism.py 2>&1 | tee outputs/run_18_mixedeffects_mechanism/log.txt
"""
import os, glob, numpy as np, pandas as pd
import numpy.linalg as la
from scipy import stats

def _has(p): return os.path.isdir(os.path.join(p, "outputs"))
_hp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT = next((c for c in [os.getcwd(), _hp,
      "/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)), os.getcwd())
OUT = os.path.join(ROOT, "outputs", "run_18_mixedeffects_mechanism"); os.makedirs(OUT, exist_ok=True)

# ---- load the exact 250-row design matrix used by step4 ----
cands = glob.glob(os.path.join(ROOT, "outputs", "**", "merged_global_psd_identity_eer.csv"), recursive=True)
if not cands:
    raise SystemExit("ERROR: merged_global_psd_identity_eer.csv not found under outputs/")
CSV = sorted(cands, key=len)[0]
df = pd.read_csv(CSV)
BANDS = ["theta_power_drift", "alpha_power_drift", "beta_power_drift", "gamma_power_drift"]
EE = "EER"
# keep the manuscript variant if several exist
if "variant" in df.columns and df["variant"].nunique() > 1:
    keep = "arcface_supcon_cnn" if (df["variant"] == "arcface_supcon_cnn").any() else df["variant"].mode()[0]
    df = df[df["variant"] == keep].copy(); print(f"  filtered to variant={keep}")
df = df.dropna(subset=BANDS + [EE]).copy()
df["emotion_pair"] = df["enroll_emotion"].astype(str) + "->" + df["test_emotion"].astype(str)
df["transition"]   = df["enroll_session"].astype(str) + "->" + df["test_session"].astype(str)
n = len(df)
nseed = df["seed"].nunique() if "seed" in df.columns else 1
print("=" * 80)
print(f"M1 mechanism re-fit | source={os.path.relpath(CSV, ROOT)}")
print(f"  n={n} rows | emotion_pairs={df['emotion_pair'].nunique()} | "
      f"transitions={df['transition'].nunique()} | seeds={nseed}")
print("=" * 80)

def z(a):
    a = np.asarray(a, float); return (a - a.mean()) / a.std(ddof=0)

Z = pd.DataFrame({b: z(df[b]) for b in BANDS}); Z["EER"] = z(df[EE])
for g in ["emotion_pair", "transition", "seed"]:
    if g in df.columns: Z[g] = df[g].values

rows = []
def rec(model, term, beta, p, extra=""):
    rows.append({"model": model, "term": term, "beta": round(float(beta), 5),
                 "p": float(p), "detail": extra})
    star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else " ns"
    print(f"  [{model:26s}] {term:6s} beta={beta:+.4f}  p={p:.3g} {star}  {extra}")

# ---- (0) standardized OLS (reproduce Table 3) ----
X = np.c_[np.ones(n), Z[BANDS].values]; y = Z["EER"].values
b, *_ = la.lstsq(X, y, rcond=None); resid = y - X @ b; k = X.shape[1]
s2 = (resid @ resid) / (n - k); se = np.sqrt(np.diag(s2 * la.inv(X.T @ X))); t = b / se
p = 2 * (1 - stats.t.cdf(np.abs(t), n - k))
R2 = 1 - (resid @ resid) / np.sum((y - y.mean()) ** 2)
print(f"\n(0) Standardized OLS  R2={R2:.3f}  (manuscript Table 3 reference: R2~0.152, theta beta~+0.38)")
for i, bd in enumerate(BANDS): rec("OLS_naive", bd.split("_")[0], b[i + 1], p[i + 1])

# ---- statsmodels-based re-fits ----
try:
    import statsmodels.formula.api as smf
    HAVE = True
except Exception as e:
    print("\n  statsmodels unavailable ->", e, "\n  (run: pip install statsmodels --break-system-packages)")
    HAVE = False

if HAVE:
    d = Z.rename(columns={"theta_power_drift": "theta", "alpha_power_drift": "alpha",
                          "beta_power_drift": "beta", "gamma_power_drift": "gamma"})
    F = "EER ~ theta + alpha + beta + gamma"

    print("\n(1) PRIMARY  MixedLM  EER ~ bands + (1|emotion_pair)")
    m1 = smf.mixedlm(F, d, groups=d["emotion_pair"]).fit(reml=False)
    for term in ["theta", "alpha", "beta", "gamma"]:
        rec("MixedLM(1|emotion_pair)", term, m1.params[term], m1.pvalues[term])
    with open(os.path.join(OUT, "mixedlm_summary.txt"), "w") as fh: fh.write(str(m1.summary()))

    print("\n(1b) MixedLM  EER ~ bands + (1|transition)")
    m1b = smf.mixedlm(F, d, groups=d["transition"]).fit(reml=False)
    rec("MixedLM(1|transition)", "theta", m1b.params["theta"], m1b.pvalues["theta"])

    print("\n(2) Cluster-robust OLS")
    for cl in ["emotion_pair", "transition"]:
        mo = smf.ols(F, d).fit(cov_type="cluster", cov_kwds={"groups": d[cl]})
        rec(f"OLS_cluster[{cl}]", "theta", mo.params["theta"], mo.pvalues["theta"])

    print("\n(5) Theta beyond alpha (nested LRT under mixed model)")
    mA = smf.mixedlm("EER ~ alpha + beta + gamma", d, groups=d["emotion_pair"]).fit(reml=False)
    LR = 2 * (m1.llf - mA.llf); pLR = stats.chi2.sf(LR, 1)
    rec("MixedLM_theta_LRT", "theta", m1.params["theta"], pLR, "vs no-theta model")

# ---- (3) seed-collapsed OLS ----
if "seed" in df.columns and nseed > 1:
    agg = df.groupby(["transition", "emotion_pair"])[BANDS + [EE]].mean().reset_index()
    m = len(agg)
    Xa = np.c_[np.ones(m), np.column_stack([z(agg[bd]) for bd in BANDS])]
    ya = z(agg[EE]); ba, *_ = la.lstsq(Xa, ya, rcond=None); ra = ya - Xa @ ba; ka = Xa.shape[1]
    sa = (ra @ ra) / (m - ka); sea = np.sqrt(np.diag(sa * la.inv(Xa.T @ Xa))); ta = ba / sea
    pa = 2 * (1 - stats.t.cdf(np.abs(ta), m - ka))
    print(f"\n(3) Seed-collapsed OLS  n={m} unique conditions")
    for i, bd in enumerate(BANDS): rec("OLS_seedcollapsed", bd.split("_")[0], ba[i + 1], pa[i + 1])

# ---- (4) block permutation test for theta ----
rng = np.random.default_rng(42); NPERM = 10000
grp = df.groupby("transition").indices
obs = abs(b[1])
null = np.empty(NPERM)
for j in range(NPERM):
    yp = y.copy()
    for idx in grp.values():
        idx = np.asarray(idx); yp[idx] = y[idx][rng.permutation(idx.size)]
    bb, *_ = la.lstsq(X, yp, rcond=None); null[j] = abs(bb[1])
pperm = (1 + np.sum(null >= obs)) / (NPERM + 1)
print(f"\n(4) Block-permutation (theta), {NPERM} perms within transition")
rec("permutation_theta", "theta", b[1], pperm, f"{NPERM} perms")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "mechanism_robust.csv"), index=False)
print("\nSAVED ->", os.path.relpath(os.path.join(OUT, "mechanism_robust.csv"), ROOT))
print("=" * 80)
print("READ-OFF: report the PRIMARY MixedLM(1|emotion_pair) theta p in Table 3.")
print("If theta stays p<0.05 across MixedLM + cluster-robust + seed-collapsed + permutation,")
print("the mechanism is robust and the claim stands (stated correctly). If it collapses in the")
print("clustered models, soften to 'a correlate that does not survive clustering'.")
print("=" * 80)
