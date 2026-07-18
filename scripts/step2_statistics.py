#!/usr/bin/env python3
"""
step2_statistics.py  -- Scientific Reports statistics for P4 drift study.

Reads the EXISTING output CSVs from previous runs and computes the formal tests
that Scientific Reports reviewers expect (hypotheses H1-H5 in STEP1_Framing):

  H2  session degradation        -> Friedman test + Kendall's W across S1/S2/S3
  H3  spectral correlate         -> per-band Pearson r vs EER + Benjamini-Hochberg FDR
                                    + standardized (z-scored) multiple regression
  H4  predictability/phase change-> two-phase (change-point) regression on drift-vs-EER
  H5  adaptation mitigation      -> paired Wilcoxon (baseline vs adapted EER) + effect size
  H1  cross- vs within-state     -> emotion-wise EER spread (reported)

It is DEFENSIVE: it auto-detects column names by keyword and, if a file or column
is missing, it prints exactly what it could not find and continues. Run it, then
paste the console output so the analysis can be tuned to your exact columns.

USAGE (from the project ROOT, inside the env):
    cd /home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
    source p4_seedv_env/bin/activate
    python -u scripts/step2_statistics.py 2>&1 | tee outputs/run_11_statistics/step2_log.txt

Outputs -> outputs/run_11_statistics/  (statistics_summary.csv + the log above)
"""

import os, sys, glob, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- locate root
def has_outputs(p): return os.path.isdir(os.path.join(p, "outputs"))
_here_parent = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT = next((c for c in [os.getcwd(), _here_parent,
                         "/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
                         "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"]
             if has_outputs(c)), os.getcwd())
OUT = os.path.join(ROOT, "outputs", "run_11_statistics")
os.makedirs(OUT, exist_ok=True)
print("="*78); print("STEP 2 STATISTICS  |  project root:", ROOT); print("="*78)

results = []   # rows for statistics_summary.csv
def record(hypo, test, stat, pval, extra=""):
    results.append({"hypothesis": hypo, "test": test, "statistic": stat,
                    "p_value": pval, "detail": extra})
    p = "n/a" if pval is None else f"{pval:.3g}"
    s = "n/a" if stat  is None else f"{stat:.4g}"
    print(f"  [{hypo}] {test}: stat={s}, p={p}  {extra}")

# ---------------------------------------------------------------- helpers
def find_file(*globs):
    for g in globs:
        hits = glob.glob(os.path.join(ROOT, "outputs", "**", g), recursive=True)
        if hits:
            return sorted(hits, key=len)[0]
    return None

def find_col(df, *keys):
    for k in keys:
        for c in df.columns:
            if k.lower() in str(c).lower():
                return c
    return None

def eer_from_scores(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2: return np.nan
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[i] + fnr[i]) / 2)

def bh_fdr(pvals):
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p)
    adj = np.empty(n); adj[order] = (p[order] * n / (np.arange(n)+1))
    # enforce monotonicity
    for i in range(n-2, -1, -1):
        adj[order[i]] = min(adj[order[i]], adj[order[i+1]])
    return np.clip(adj, 0, 1)

def mean_ci(x, alpha=0.05):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return (np.nanmean(x) if len(x) else np.nan, np.nan, np.nan)
    m = x.mean(); se = x.std(ddof=1)/np.sqrt(len(x))
    h = se * stats.t.ppf(1-alpha/2, len(x)-1)
    return m, m-h, m+h

# ============================================================ H2 session drift
print("\n--- H2: session degradation across S1->S2->S3 ---")
f = find_file("score_level_session_verification_scores.csv")
done_h2 = False
if f:
    print("  using", os.path.relpath(f, ROOT))
    df = pd.read_csv(f)
    print("  columns:", list(df.columns))
    csub = find_col(df, "subject", "subj", "claimed")
    cses = find_col(df, "probe_session", "session")
    cy   = find_col(df, "y_true", "label", "genuine", "is_genuine")
    csc  = find_col(df, "score")
    if all([csub, cses, cy, csc]):
        rows = []
        for (su, se), g in df.groupby([csub, cses]):
            rows.append((su, se, eer_from_scores(g[cy], g[csc])))
        per = pd.DataFrame(rows, columns=["subject","session","eer"])
        mat = per.pivot_table(index="subject", columns="session", values="eer").dropna()
        print("  per-subject EER matrix shape:", mat.shape, "sessions:", list(mat.columns))
        for s in mat.columns:
            m, lo, hi = mean_ci(mat[s].values)
            print(f"    session {s}: mean EER={m:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        if mat.shape[1] >= 3:
            chi, p = stats.friedmanchisquare(*[mat[c].values for c in mat.columns])
            n, k = mat.shape; W = chi / (n*(k-1))   # Kendall's W
            record("H2", "Friedman (session EER)", chi, p, f"KendallW={W:.3f}, n={n}")
            done_h2 = True
        else:
            print("  <3 sessions in matrix -> cannot run Friedman")
    else:
        print(f"  MISSING column(s): subject={csub}, session={cses}, label={cy}, score={csc}")
if not done_h2:
    # fallback: aggregate 3-row table (no per-subject test, just report monotonicity)
    f2 = find_file("session_verification_results.csv")
    if f2:
        d = pd.read_csv(f2); ce = find_col(d, "eer")
        print("  fallback aggregate EERs:", d[ce].tolist() if ce else d.head())
        record("H2", "monotonic check (aggregate)", None, None,
               f"EERs={d[ce].round(4).tolist() if ce else 'n/a'}")

# ============================================================ H3 spectral band
print("\n--- H3: spectral-band correlates of EER (FDR) + regression ---")
f = find_file("merged_global_psd_identity_eer.csv", "merged_region_psd_identity_eer.csv",
              "*psd*identity*eer*.csv", "*band*eer*.csv")
if f:
    print("  using", os.path.relpath(f, ROOT))
    df = pd.read_csv(f); print("  columns:", list(df.columns))
    ceer = find_col(df, "eer")
    bands = [c for c in df.columns if any(b in str(c).lower()
             for b in ["theta","alpha","beta","gamma","delta","drift","psd"]) and c != ceer]
    if ceer and bands:
        names, rs, ps = [], [], []
        for b in bands:
            sub = df[[b, ceer]].dropna()
            if sub[b].nunique() > 2 and len(sub) > 3:
                r, p = stats.pearsonr(sub[b], sub[ceer]); names.append(b); rs.append(r); ps.append(p)
        if ps:
            q = bh_fdr(ps)
            for nme, r, p, qq in sorted(zip(names, rs, ps, q), key=lambda t: t[2]):
                tag = "*" if qq < 0.05 else ""
                record("H3", f"Pearson {nme} vs EER", r, p, f"FDR-q={qq:.3g}{tag}")
            # standardized multiple regression
            X = df[names].dropna(); y = df.loc[X.index, ceer]
            Xz = (X - X.mean())/X.std(ddof=0)
            lr = LinearRegression().fit(Xz, y)
            coefs = dict(zip(names, np.round(lr.coef_, 4)))
            record("H3", "standardized regression R^2", lr.score(Xz, y), None, f"betas={coefs}")
    else:
        print(f"  MISSING: eer col={ceer}, band cols={bands}")
else:
    print("  no PSD-EER merged table found (check run_05/run_06 tables names)")

# ============================================================ H4 change-point
print("\n--- H4: two-phase (change-point) regression on drift-vs-EER ---")
f = find_file("merged_global_psd_identity_eer.csv", "final_eer_vs_drift_cdt_summary.csv",
              "*drift*eer*.csv")
if f:
    print("  using", os.path.relpath(f, ROOT))
    df = pd.read_csv(f)
    cd = find_col(df, "drift", "cdt", "psd"); ce = find_col(df, "eer")
    if cd and ce:
        d = df[[cd, ce]].dropna().sort_values(cd).reset_index(drop=True)
        x = d[cd].values; y = d[ce].values
        def sse_line(x, y):
            if len(x) < 2: return np.inf
            b = np.polyfit(x, y, 1); return np.sum((y - np.polyval(b, x))**2)
        sse1 = sse_line(x, y)
        best = (np.inf, None)
        for i in range(2, len(x)-2):
            s = sse_line(x[:i], y[:i]) + sse_line(x[i:], y[i:])
            if s < best[0]: best = (s, i)
        if best[1]:
            cp = x[best[1]]
            n, k1, k2 = len(x), 2, 4
            F = ((sse1-best[0])/(k2-k1)) / (best[0]/(n-k2)) if best[0] > 0 else np.inf
            pF = 1 - stats.f.cdf(F, k2-k1, n-k2)
            record("H4", "two-phase vs single-line (F)", F, pF,
                   f"changepoint drift~={cp:.4g}, SSE {sse1:.3g}->{best[0]:.3g}")
    else:
        print(f"  MISSING: drift col={cd}, eer col={ce}")
else:
    print("  no drift-vs-EER table found")

# ============================================================ H5 adaptation
print("\n--- H5: subject adaptation mitigation (paired) ---")
f = find_file("subject_adaptation_results.csv", "*adaptation*results*.csv")
if f:
    print("  using", os.path.relpath(f, ROOT))
    df = pd.read_csv(f); print("  columns:", list(df.columns))
    cb = find_col(df, "baseline", "before", "no_adapt")
    ca = find_col(df, "after", "adapted", "with_adapt", "best_eer")
    if cb and ca:
        d = df[[cb, ca]].dropna()
        try:
            w, p = stats.wilcoxon(d[cb], d[ca])
            rbc = 1 - (2*w)/(len(d)*(len(d)+1)/2)   # rough effect-size proxy
            record("H5", "Wilcoxon baseline vs adapted", w, p,
                   f"n={len(d)}, mean dEER={-(d[ca]-d[cb]).mean():.4f}")
        except Exception as e:
            print("  wilcoxon failed:", e)
    else:
        print(f"  MISSING paired cols: baseline={cb}, adapted={ca}  (try adaptation_improvement_summary.csv)")
else:
    print("  no per-subject adaptation table found")

# ============================================================ H1 emotion spread
print("\n--- H1: emotion-wise EER spread ---")
f = find_file("run03_emotionwise_mahalanobis_results.csv", "*emotionwise*results*.csv")
if f:
    df = pd.read_csv(f); ce = find_col(df, "eer")
    if ce:
        m, lo, hi = mean_ci(df[ce].values)
        record("H1", "emotion EER mean/CI", m, None,
               f"range [{df[ce].min():.4f},{df[ce].max():.4f}], 95%CI[{lo:.4f},{hi:.4f}]")

# ---------------------------------------------------------------- save
summary = pd.DataFrame(results)
csv = os.path.join(OUT, "statistics_summary.csv")
summary.to_csv(csv, index=False)
print("\n" + "="*78)
print("SAVED:", os.path.relpath(csv, ROOT))
print(summary.to_string(index=False))
print("="*78)
print("Paste this console output back so any MISSING-column blocks can be tuned.")
