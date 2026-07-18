#!/usr/bin/env python3
r"""
40_m11_wilcoxon_fdr.py -- REVIEWER FIX m11.
Recovers the EXACT Wilcoxon W statistics and adds Benjamini-Hochberg FDR (family = the
affect x session contrasts) for the state/time/both effects, WITHOUT re-scoring: it reads the
per-subject EERs already produced by 32_state_vs_time.py (outputs/run_19_state_vs_time/persubject.csv).
Because it calls scipy.stats.wilcoxon exactly as 32_state_vs_time.py did, the p-values reproduce
the manuscript's reported p; it simply also prints W and the FDR-adjusted q.

RUN (Brev or laptop, no GPU, needs only the CSV):
  cd ~/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
  python -u scripts/40_m11_wilcoxon_fdr.py
If persubject.csv is missing, first run 32_state_vs_time.py.
"""
import os, glob, numpy as np, pandas as pd
from scipy import stats

def _has(p): return os.path.isdir(os.path.join(p, "outputs"))
_hp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT = next((c for c in [os.getcwd(), _hp,
      "/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)), os.getcwd())
CANDS = glob.glob(os.path.join(ROOT, "outputs", "**", "run_19_state_vs_time", "persubject.csv"), recursive=True) \
      + glob.glob(os.path.join(ROOT, "outputs", "run_19_state_vs_time", "persubject.csv"))
assert CANDS, "persubject.csv not found -- run 32_state_vs_time.py first."
CSV = CANDS[0]
OUT = os.path.dirname(CSV)
ps = pd.read_csv(CSV)
print("read:", CSV, "| rows:", len(ps), "| cols:", list(ps.columns))

def cell(sc, ec):   # per-subject mean EER for a 2x2 cell (matches 32_state_vs_time.py)
    return ps[(ps.session_changed == sc) & (ps.emotion_changed == ec)].groupby("subject")["EER"].mean()
b0 = cell(0, 0)   # baseline: same session, same emotion

CONTRASTS = [("state", 0, 1, "same session, different emotion  (cognitive-state drift)"),
             ("time",  1, 0, "different session, same emotion   (elapsed-time drift)"),
             ("both",  1, 1, "different session, different emotion")]
rows = []
for name, sc, ec, desc in CONTRASTS:
    x = cell(sc, ec); j = b0.index.intersection(x.index)
    a, b = x.loc[j].values, b0.loc[j].values
    d = a - b
    try:
        W, p = stats.wilcoxon(a, b)                      # EXACT same call as 32_state_vs_time.py
    except Exception as e:
        W, p = float("nan"), float("nan"); print("  wilcoxon err", name, e)
    rows.append({"effect": name, "desc": desc, "n": int(len(j)),
                 "dEER_mean": float(np.mean(d)), "wilcoxon_W": float(W), "p_raw": float(p)})

res = pd.DataFrame(rows)
# Benjamini-Hochberg FDR across the family of affect x session contrasts
m = len(res)
order = res["p_raw"].rank(method="first").astype(int)
res = res.sort_values("p_raw").reset_index(drop=True)
res["rank"] = np.arange(1, len(res) + 1)
res["q_BH"] = (res["p_raw"] * m / res["rank"]).clip(upper=1.0)
res["q_BH"] = res["q_BH"][::-1].cummin()[::-1]          # enforce monotonic q
# Holm (for reporting robustness)
res["p_Holm"] = (res["p_raw"] * (m - res["rank"] + 1)).clip(upper=1.0)
res["p_Holm"] = res["p_Holm"].cummax()
res["sig_FDR_0.05"] = res["q_BH"] < 0.05

res.to_csv(os.path.join(OUT, "m11_wilcoxon_fdr.csv"), index=False)
pd.set_option("display.width", 160)
print("\n" + "=" * 84)
print("m11  Wilcoxon W + Benjamini-Hochberg FDR  (family = affect x session contrasts, m={})".format(m))
print("=" * 84)
for _, r in res.iterrows():
    print(f"  {r['effect']:5s}  n={int(r['n'])}  dEER={r['dEER_mean']:+.4f}  "
          f"W={r['wilcoxon_W']:.1f}  p={r['p_raw']:.3g}  q(BH)={r['q_BH']:.3g}  "
          f"Holm={r['p_Holm']:.3g}  {'sig@FDR.05' if r['sig_FDR_0.05'] else 'ns@FDR.05'}")
print("-" * 84)
print("  Report in text: 'W(n), p (raw), q (BH-FDR across the K affect x session contrasts)'.")
print("  These p reproduce 32_state_vs_time.py exactly (same scipy.stats.wilcoxon call); W and q are added.")
print("  saved ->", os.path.join(OUT, "m11_wilcoxon_fdr.csv"))
