#!/usr/bin/env python3
"""
step4_biological.py  -- Step 4: biological mechanism of EEG identity drift.

Turns the RUN05/RUN06 spectral analysis into the manuscript's mechanism section:

  1. Standardized multiple regression  band-drift -> EER   (proper z(X) AND z(y),
     standardized betas + t-tests + p-values + R^2)   [fixes the v1 beta display]
  2. Region ranking: which cortical regions' drift best predicts EER (FDR)
  3. Three publication figures (300 dpi):
       fig1_theta_vs_eer.png        theta drift vs EER + OLS line
       fig2_band_betas.png          standardized band coefficients (signif marked)
       fig3_region_ranking.png      |r| of region drift vs EER
     (+ fig4_embedding_trajectory.png if PCA trajectory table exists)

Reads existing RUN06 tables; DEFENSIVE (auto-detects columns, skips gracefully).
Outputs -> outputs/run_13_biological/{biological_stats.csv, figures/}

USAGE (project root, env active):
    cd /home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
    source p4_seedv_env/bin/activate
    python -u scripts/step4_biological.py 2>&1 | tee outputs/run_13_biological/step4_log.txt
"""
import os, glob, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def has(p): return os.path.isdir(os.path.join(p, "outputs"))
_hp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT = next((c for c in [os.getcwd(), _hp,
            "/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
            "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if has(c)), os.getcwd())
OUT = os.path.join(ROOT, "outputs", "run_13_biological"); FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
print("="*78); print("STEP 4 BIOLOGY | root:", ROOT); print("="*78)

def findf(*globs):
    for g in globs:
        h = glob.glob(os.path.join(ROOT, "outputs", "**", g), recursive=True)
        if h: return sorted(h, key=len)[0]
    return None
def col(df, *k):
    for key in k:
        for c in df.columns:
            if key.lower() in str(c).lower(): return c
    return None
def bh(p):
    p = np.asarray(p, float); n=len(p); o=np.argsort(p); a=np.empty(n)
    a[o]=p[o]*n/(np.arange(n)+1)
    for i in range(n-2,-1,-1): a[o[i]]=min(a[o[i]],a[o[i+1]])
    return np.clip(a,0,1)

stats_rows = []
def rec(metric, val, p=None, extra=""):
    stats_rows.append({"metric":metric,"value":val,"p_value":p,"detail":extra})
    print(f"  {metric}: {val:.4g}" + (f", p={p:.3g}" if p is not None else "") + (f"  {extra}" if extra else ""))

# ---------------------------------------------- load global PSD-EER table
fg = findf("merged_global_psd_identity_eer.csv")
df = pd.read_csv(fg); print("global table:", os.path.relpath(fg, ROOT)); print("cols:", list(df.columns))
ceer = col(df, "EER")
bands = [c for c in df.columns if any(b in str(c).lower() for b in
         ["theta","alpha","beta","gamma","delta"]) and "drift" in str(c).lower()]
print("band-drift cols:", bands)

# ---------------------------------------------- (1) standardized OLS
print("\n--- (1) Standardized regression band-drift -> EER ---")
sub = df[bands+[ceer]].dropna()
X = sub[bands].values; y = sub[ceer].values
Xz = (X - X.mean(0))/X.std(0, ddof=0); yz = (y - y.mean())/y.std(ddof=0)
Xd = np.c_[np.ones(len(Xz)), Xz]
beta, *_ = np.linalg.lstsq(Xd, yz, rcond=None)
resid = yz - Xd@beta; n,k = Xd.shape
sigma2 = (resid@resid)/(n-k)
covb = sigma2*np.linalg.inv(Xd.T@Xd)
se = np.sqrt(np.diag(covb)); tval = beta/se
pval = 2*(1-stats.t.cdf(np.abs(tval), n-k))
R2 = 1 - (resid@resid)/np.sum((yz-yz.mean())**2)
rec("R2_standardized", R2, None, f"n={n}")
betas = {}
for i,b in enumerate(bands):
    rec(f"beta[{b}]", beta[i+1], pval[i+1]); betas[b]=beta[i+1]

# per-band simple correlations + FDR (for the figure)
rs, ps = [], []
for b in bands:
    r,p = stats.pearsonr(sub[b], sub[ceer]); rs.append(r); ps.append(p)
q = bh(ps)
for b,r,p,qq in zip(bands,rs,ps,q): rec(f"pearson[{b}]", r, p, f"FDR-q={qq:.3g}")

# ---------------------------------------------- (2) region ranking
print("\n--- (2) Region ranking (drift -> EER) ---")
fr = findf("merged_region_psd_identity_eer.csv", "region_psd_drift_cross_emotion.csv")
region_rank = []
if fr:
    rdf = pd.read_csv(fr); print("region table:", os.path.relpath(fr, ROOT)); print("cols:", list(rdf.columns))
    creg = col(rdf, "region"); rceer = col(rdf, "EER")
    if creg and rceer:
        cdrift = col(rdf, "total_psd_drift", "drift")
        for reg, g in rdf.dropna(subset=[cdrift, rceer]).groupby(creg):
            if g[cdrift].nunique()>2 and len(g)>3:
                r,p = stats.pearsonr(g[cdrift], g[rceer]); region_rank.append((reg,r,p,len(g)))
    else:  # regions as columns
        rc = col(rdf,"EER")
        regcols=[c for c in rdf.columns if any(z in str(c).lower() for z in
                 ["frontal","central","temporal","parietal","occipital"])]
        for c in regcols:
            s=rdf[[c,rc]].dropna()
            if s[c].nunique()>2: r,p=stats.pearsonr(s[c],s[rc]); region_rank.append((c,r,p,len(s)))
    if region_rank:
        qs = bh([t[2] for t in region_rank])
        region_rank = [(*t, qq) for t,qq in zip(region_rank, qs)]
        region_rank.sort(key=lambda t: -abs(t[1]))
        for reg,r,p,nn,qq in region_rank:
            rec(f"region[{reg}]", r, p, f"FDR-q={qq:.3g}, n={nn}")
else:
    print("  region table not found")

pd.DataFrame(stats_rows).to_csv(os.path.join(OUT,"biological_stats.csv"), index=False)

# ---------------------------------------------- figures
print("\n--- figures ---")
theta = col(df,"theta_power_drift","theta")
if theta:
    s = df[[theta,ceer]].dropna(); x,yv = s[theta].values, s[ceer].values
    b1 = np.polyfit(x,yv,1); xs=np.linspace(x.min(),x.max(),100)
    plt.figure(figsize=(5,4)); plt.scatter(x,yv,s=8,alpha=0.4,color="#3b6ea5")
    plt.plot(xs,np.polyval(b1,xs),color="#c0392b",lw=2)
    r,p=stats.pearsonr(x,yv)
    plt.title(f"Theta drift vs EER (r={r:.2f}, p={p:.1e})"); plt.xlabel("Theta-band PSD drift"); plt.ylabel("EER")
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig1_theta_vs_eer.png"),dpi=300); plt.close(); print("  fig1 saved")
if betas:
    names=list(betas.keys()); vals=[betas[n] for n in names]
    sig=["*" if pval[i+1]<0.05 else "" for i in range(len(names))]
    plt.figure(figsize=(5,4)); bars=plt.bar(range(len(names)),vals,color="#3b6ea5")
    plt.axhline(0,color="k",lw=.6); plt.xticks(range(len(names)),[n.split("_")[0] for n in names],rotation=30)
    for i,(b,sg) in enumerate(zip(bars,sig)):
        plt.text(i,b.get_height(),sg,ha="center",va="bottom",fontsize=14)
    plt.title(f"Standardized band coefficients (R²={R2:.2f})"); plt.ylabel("Std. β (→EER)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig2_band_betas.png"),dpi=300); plt.close(); print("  fig2 saved")
if region_rank:
    regs=[str(t[0]).split("_")[0] for t in region_rank]; rv=[abs(t[1]) for t in region_rank]
    plt.figure(figsize=(5,4)); plt.barh(regs[::-1],rv[::-1],color="#27ae60")
    plt.xlabel("|Pearson r|  (region drift vs EER)"); plt.title("Cortical region contribution to drift")
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig3_region_ranking.png"),dpi=300); plt.close(); print("  fig3 saved")
ft = findf("spectral_embedding_pca_trajectory_coordinates.csv")
if ft:
    td = pd.read_csv(ft); cx=col(td,"pc1","pca1","x"); cy=col(td,"pc2","pca2","y"); cc=col(td,"emotion","session","subject")
    if cx and cy:
        plt.figure(figsize=(5,4))
        if cc:
            for g,gg in td.groupby(cc): plt.plot(gg[cx],gg[cy],"-o",ms=3,alpha=.7,label=str(g))
            plt.legend(fontsize=6,ncol=2)
        else: plt.plot(td[cx],td[cy],"-o",ms=3)
        plt.title("Identity embedding trajectory (PCA)"); plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig4_embedding_trajectory.png"),dpi=300); plt.close(); print("  fig4 saved")

print("\nSAVED stats:", os.path.relpath(os.path.join(OUT,"biological_stats.csv"),ROOT))
print("figures dir:", os.path.relpath(FIG,ROOT)); print("="*78)
