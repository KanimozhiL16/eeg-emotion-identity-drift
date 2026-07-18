#!/usr/bin/env python3
"""
16_aep_spectral_theta.py -- C14 (does theta/T7 drift predict EER on the auditory set?).
On the PhysioNet AEP npz (20 subj, 4 ch; ch order read from file, T7 = temporal electrode) we compute
per-(subject,session) band power, per-subject cross-session band drift, and test whether THETA drift
-- and specifically T7 theta drift -- correlates with per-subject cross-session EER.
Reads : data/cross_dataset_aep/AEP_win2s_step1s_fs256_4ch.npz  (X (N,4,512), y_subject, source_file, ch_names)
Writes: outputs/work3/16_aep_theta/aep_theta.csv
"""
import os, re, numpy as np, pandas as pd
from scipy import signal, stats
from sklearn.metrics import roc_curve

def _has(p): return os.path.isdir(os.path.join(p,"outputs"))
_hp=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
ROOT=next((c for c in [os.getcwd(),_hp,"/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)),os.getcwd())
NPZ=os.path.join(ROOT,"data","cross_dataset_aep","AEP_win2s_step1s_fs256_4ch.npz")
OUT=os.path.join(ROOT,"outputs","work3","16_aep_theta"); os.makedirs(OUT,exist_ok=True)
BANDS=[("delta",1,4),("theta",4,8),("alpha",8,13),("beta",13,30),("gamma",30,45)]
RNG=np.random.default_rng(42)

d=np.load(NPZ,allow_pickle=True); X=np.asarray(d["X"],np.float32); subj=np.asarray(d["y_subject"],int)
src=np.asarray([str(s) for s in d["source_file"]]); fs=int(d["fs"]) if "fs" in d else 256
CH=[str(c) for c in d["ch_names"]]; T7=CH.index("T7") if "T7" in CH else 0
print("AEP channels:",CH,"| T7 at index",T7)
def parse(s):
    m=re.match(r"s(\d+)_ex(\d+)(?:_s(\d+))?",s); return (int(m.group(1)),int(m.group(2)),int(m.group(3)) if m and m.group(3) else 0) if m else (None,None,0)
pid=np.array([parse(s) for s in src]); exp=pid[:,1]; sess=pid[:,2]

def bandpow(win):
    out=np.zeros((4,5))
    for c in range(4):
        f,p=signal.welch(win[c],fs=fs,nperseg=min(256,win.shape[-1]))
        for bi,(_,lo,hi) in enumerate(BANDS): out[c,bi]=p[(f>=lo)&(f<hi)].sum()
    return out
mask=(exp==1)
BP=np.array([bandpow(w) for w in X[mask]]); s_subj=subj[mask]; s_sess=sess[mask]
def feats(win):
    row=[]
    for c in range(4):
        ch=win[c]; row+=[ch.mean(),ch.std(),np.sqrt((ch**2).mean())]
        f,p=signal.welch(ch,fs=fs,nperseg=min(256,ch.shape[-1]))
        for _,lo,hi in BANDS: row.append(float(p[(f>=lo)&(f<hi)].sum()))
    return row
F=np.array([feats(w) for w in X[mask]],float)
def l2(v): n=np.linalg.norm(v,axis=-1,keepdims=True); return v/np.clip(n,1e-12,None)
def eer_of(y,s): fpr,tpr,_=roc_curve(y,s); fnr=1-tpr; i=int(np.nanargmin(np.abs(fpr-fnr))); return float((fpr[i]+fnr[i])/2)
SUBS=sorted(np.unique(s_subj).tolist()); en=(s_sess==1)
mu=F[en].mean(0); sd=F[en].std(0)+1e-8
def persub_cross_eer():
    Fe=(F[en]-mu)/sd; se=s_subj[en]; P=np.array([l2(Fe[se==c].mean(0)) for c in SUBS]); out={}
    cross=np.isin(s_sess,[2,3]); Fp=l2((F[cross]-mu)/sd); sp=s_subj[cross]; S=Fp@P.T
    for j,c in enumerate(SUBS):
        gen=S[sp==c,j]; imp=S[sp!=c,j]
        if len(gen) and len(imp): out[c]=eer_of(np.r_[np.ones(len(gen)),np.zeros(len(imp))],np.r_[gen,imp])
    return out
EER=persub_cross_eer()
rows=[]
for c in SUBS:
    m1=(s_subj==c)&(s_sess==1); mc=(s_subj==c)&np.isin(s_sess,[2,3])
    if m1.sum()<2 or mc.sum()<2 or c not in EER: continue
    bp1=BP[m1].mean(0); bpc=BP[mc].mean(0)
    theta_all=np.abs(bpc[:,1]-bp1[:,1]).mean(); theta_T7=abs(bpc[T7,1]-bp1[T7,1])
    rows.append({"subject":c,"theta_drift_4ch":float(theta_all),"theta_drift_T7":float(theta_T7),"cross_EER":float(EER[c])})
df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,"aep_theta.csv"),index=False)
print("="*84+"\n[C14] AEP theta drift -> cross-session EER (n=%d subjects)"%len(df))
for col in ["theta_drift_4ch","theta_drift_T7"]:
    rho,pr=stats.spearmanr(df[col],df["cross_EER"]); print(f"  {col:18s} Spearman rho={rho:+.3f} p={pr:.3g}")
print("  READ-OFF: positive theta (esp. T7) vs EER supports the temporal-theta account beyond SEED-V.")
print("  If null, keep 'partial replication of degradation'.")
print("DONE ->",os.path.relpath(OUT,ROOT))
