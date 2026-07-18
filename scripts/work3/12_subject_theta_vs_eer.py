#!/usr/bin/env python3
"""
12_subject_theta_vs_eer.py -- C20 (the strongest missing experiment).
Question: do participants with greater theta drift have higher cross-session EER?
Per subject i: theta_drift_i = mean over channels of |theta_power(S2,3) - theta_power(S1)|
               EER_i         = per-subject cross-session EER (reuse run_07 scoring, same as manuscript)
Then: Spearman(theta_drift_i, EER_i), robust (Theil-Sen) regression, and leave-one-subject-out check.

Reuses scripts/07_seedv_session_drift_subject_adaptation.py (simple_features, make_prototypes,
verification_scores, eer_auc) so EERs match the paper. Band power from Welch on the SAME windows.
Writes: outputs/work3/12_subject_theta/subject_theta_eer.csv + figure.
Run   : python -u scripts/work3/12_subject_theta_vs_eer.py 2>&1 | tee outputs/work3/12_subject_theta.log
"""
import os, glob, importlib.util, numpy as np, pandas as pd
from scipy.signal import welch
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def _has(p): return os.path.isdir(os.path.join(p,"outputs"))
_hp=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
ROOT=next((c for c in [os.getcwd(),_hp,"/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)),os.getcwd())
SRC=os.path.join(ROOT,"scripts","07_seedv_session_drift_subject_adaptation.py")
DATA=os.path.join(ROOT,"data","processed","sessionwise")
OUT=os.path.join(ROOT,"outputs","work3","12_subject_theta"); FIG=os.path.join(OUT,"figures"); os.makedirs(FIG,exist_ok=True)
FS=200
spec=importlib.util.spec_from_file_location("run07",SRC); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

# ---- load raw windows per session (X kept for band power) ----
raw={}
for f in sorted(glob.glob(os.path.join(DATA,"*.npz"))):
    d=np.load(f,allow_pickle=True)
    if not all(k in d for k in ("X","y_subject","y_session")): continue
    sid=int(np.unique(d["y_session"])[0])
    if sid in raw: continue
    raw[sid]=(np.asarray(d["X"],np.float32), np.asarray(d["y_subject"],int))
SUBS=sorted(np.unique(raw[1][1]).tolist())

def theta_power(X):   # (N,C,T)->(N,C) theta 4-8Hz absolute (amplitude-preserving)
    f,P=welch(X.astype(np.float64),fs=FS,nperseg=min(200,X.shape[-1]),axis=-1)
    mask=(f>=4)&(f<8); return P[:,:,mask].sum(-1)
def subj_theta(sid):
    X,y=raw[sid]; tp=theta_power(X); return {int(s):tp[y==s].mean(0) for s in np.unique(y)}  # per subj (C,)
th1=subj_theta(1); th2=subj_theta(2); th3=subj_theta(3)

# ---- per-subject cross-session EER via run_07 (matches manuscript) ----
def feats(sid):
    X,y=raw[sid]
    try: X,y=m.downsample_per_subject(X,y,max_per_subject=900,seed=42)[:2]
    except Exception: pass
    return m.simple_features(X), y
Fe,ye=feats(1); P,owners=m.make_prototypes(Fe,ye); owners=np.asarray(owners)
def persub_eer(sid):
    Fp,yp=feats(sid); sdf=m.verification_scores(Fp,yp,P,owners)
    return {int(c):m.eer_auc(g["y_true"].values,g["score"].values)[0] for c,g in sdf.groupby("probe_subject")}
e2,e3=persub_eer(2),persub_eer(3)

rows=[]
for s in SUBS:
    dth=0.5*(np.abs(th2[s]-th1[s]).mean()+np.abs(th3[s]-th1[s]).mean())   # mean |Δθ| over channels & sessions
    eer=0.5*(e2.get(s,np.nan)+e3.get(s,np.nan))
    rows.append({"subject":s,"theta_drift":round(float(dth),6),"cross_EER":round(float(eer),4)})
df=pd.DataFrame(rows).dropna(); df.to_csv(os.path.join(OUT,"subject_theta_eer.csv"),index=False)
x=df["theta_drift"].to_numpy(float); y=df["cross_EER"].to_numpy(float)

rho,pr=stats.spearmanr(x,y); tau,pt=stats.kendalltau(x,y)
ts=stats.theilslopes(y,x);
# LOSO stability of Spearman
loso=[stats.spearmanr(np.delete(x,i),np.delete(y,i))[0] for i in range(len(x))]
print("="*84); print(f"[C20] subjects n={len(df)}")
print(f"  Spearman rho={rho:+.3f} p={pr:.3g} | Kendall tau={tau:+.3f} p={pt:.3g}")
print(f"  Theil-Sen slope={ts[0]:+.4g} (95% CI {ts[2]:+.4g},{ts[3]:+.4g})")
print(f"  LOSO Spearman range [{min(loso):+.3f}, {max(loso):+.3f}] (sign-stable if all same sign)")
print("  READ-OFF: positive significant Spearman + sign-stable LOSO => theta drift tracks identity EER at subject level.")

plt.figure(figsize=(5,4)); plt.scatter(x,y)
for _,r in df.iterrows(): plt.annotate(f"S{int(r['subject'])}",(r['theta_drift'],r['cross_EER']),fontsize=6)
xs=np.linspace(x.min(),x.max(),50); plt.plot(xs,ts[1]+ts[0]*xs,'r-',lw=1)
plt.xlabel("per-subject theta drift |Δθ|"); plt.ylabel("cross-session EER")
plt.title(f"C20  Spearman ρ={rho:+.2f}, p={pr:.3g}"); plt.tight_layout()
plt.savefig(os.path.join(FIG,"subject_theta_vs_eer.png"),dpi=300); plt.close()
print("DONE ->",os.path.relpath(OUT,ROOT))
