#!/usr/bin/env python3
r"""
41_state_vs_time_figure.py -- figure for the affect(state) x session(time) decomposition (M2/m14/m11).
Pure plotting: reads what 32_state_vs_time.py + 40_m11_wilcoxon_fdr.py already produced
(outputs/run_19_state_vs_time/{persubject.csv, cells.csv, m11_wilcoxon_fdr.csv}). No re-compute.

Panel (a): mean EER for the 4 cells (baseline / state-only / time-only / both) with per-subject
           points and 95% bootstrap CI error bars.
Panel (b): per-subject ΔEER vs baseline for state / time / both, with 95% CI and BH-FDR stars.

OUT -> outputs/run_19_state_vs_time/figures/state_vs_time.png (+ .pdf)
RUN : cd ~/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC && python -u scripts/41_state_vs_time_figure.py
"""
import os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def _has(p): return os.path.isdir(os.path.join(p, "outputs"))
_hp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT = next((c for c in [os.getcwd(), _hp,
      "/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)), os.getcwd())
def find(name):
    c = glob.glob(os.path.join(ROOT, "outputs", "**", "run_19_state_vs_time", name), recursive=True) \
      + glob.glob(os.path.join(ROOT, "outputs", "run_19_state_vs_time", name))
    return c[0] if c else None
PS = find("persubject.csv"); assert PS, "persubject.csv missing -- run 32_state_vs_time.py first."
OUT = os.path.dirname(PS); FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
ps = pd.read_csv(PS)
fdr = find("m11_wilcoxon_fdr.csv"); fdr = pd.read_csv(fdr) if fdr else None
RNG = np.random.default_rng(0)

CELLS = [("baseline", 0, 0), ("state-only", 0, 1), ("time-only", 1, 0), ("both", 1, 1)]
COLORS = ["#9aa0a6", "#2e7d32", "#5b9bd5", "#8e44ad"]

def persub(sc, ec):   # per-subject mean EER for a cell
    return ps[(ps.session_changed == sc) & (ps.emotion_changed == ec)].groupby("subject")["EER"].mean()

def boot_ci(v, f=np.mean, B=5000):
    v = np.asarray(v, float)
    bs = [f(v[RNG.integers(0, len(v), len(v))]) for _ in range(B)]
    return f(v), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.9))

# ---- Panel (a): cell means + per-subject dots + 95% CI ----
b0 = persub(0, 0)
for i, (name, sc, ec) in enumerate(CELLS):
    v = persub(sc, ec)
    m, lo, hi = boot_ci(v.values)
    axA.bar(i, m, color=COLORS[i], alpha=.85, width=.66, zorder=1)
    axA.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="k", capsize=3, lw=1, zorder=3)
    jx = i + (RNG.random(len(v)) - .5) * 0.28
    axA.scatter(jx, v.values, s=12, color="k", alpha=.45, zorder=2)
axA.set_xticks(range(4)); axA.set_xticklabels([c[0] for c in CELLS], fontsize=8, rotation=12)
axA.set_ylabel("Equal error rate (EER)", fontsize=9)
axA.set_title("(a) Verification error by cell (n=16)", fontsize=9)
axA.grid(axis="y", alpha=.25)

# ---- Panel (b): ΔEER vs baseline for state/time/both with 95% CI + FDR stars ----
def stars(q):
    return "***" if q < 1e-3 else "**" if q < 1e-2 else "*" if q < 0.05 else "n.s."
order = [("state-only", 0, 1, "state"), ("time-only", 1, 0, "time"), ("both", 1, 1, "both")]
for i, (name, sc, ec, key) in enumerate(order):
    v = persub(sc, ec); j = b0.index.intersection(v.index)
    d = (v.loc[j] - b0.loc[j]).values
    m, lo, hi = boot_ci(d)
    col = COLORS[[c[0] for c in CELLS].index(name)]
    axB.bar(i, m, color=col, alpha=.85, width=.6)
    axB.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="k", capsize=3, lw=1)
    q = float(fdr.loc[fdr.effect == key, "q_BH"].iloc[0]) if fdr is not None and (fdr.effect == key).any() else np.nan
    lab = stars(q) if not np.isnan(q) else ""
    axB.text(i, hi + 0.004, lab, ha="center", va="bottom", fontsize=9)
axB.axhline(0, color="k", lw=.8)
axB.set_xticks(range(3)); axB.set_xticklabels(["state\nΔ", "time\nΔ", "both\nΔ"], fontsize=8)
axB.set_ylabel("ΔEER vs baseline", fontsize=9)
axB.set_title("(b) State vs time drift (BH-FDR)", fontsize=9)
axB.grid(axis="y", alpha=.25)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"state_vs_time.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved ->", os.path.join(FIG, "state_vs_time.png"), "and .pdf")
print("Panel (b) stars use BH-FDR q from m11_wilcoxon_fdr.csv (***q<1e-3, **<1e-2, *<0.05).")
