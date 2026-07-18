#!/usr/bin/env python3
"""
14_theta_aware_mitigation.py -- C15 (connect the mechanism to the mitigation).
The paper's mitigation (generic re-enrolment) is not theta-aware. Here we test a THETA-AWARE,
UNSUPERVISED countermeasure motivated directly by the theta finding: per-session, per-channel
standardisation of the theta band (remove the session-level theta shift) applied to the feature
vector before cosine verification. No labels from the future session are used (unlike re-enrolment).

Compare cross-session EER (enrol S1 / verify S2,S3), per participant:
  baseline           : full band-power features, enrolment-set standardisation only (paper style)
  theta-normalised   : additionally remove per-session theta offset (theta-aware, unsupervised)
  (reference)        : generic re-enrolment numbers from the paper (Table adapt) for context
Writes: outputs/work3/14_theta_mitigation/theta_mitigation.csv
Run   : python -u scripts/work3/14_theta_aware_mitigation.py 2>&1 | tee outputs/work3/14_theta_mitigation.log
"""
import os, glob, numpy as np, pandas as pd
from scipy.signal import welch
from scipy import stats
from sklearn.metrics import roc_curve

def _has(p): return os.path.isdir(os.path.join(p,"outputs"))
_hp=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
ROOT=next((c for c in [os.getcwd(),_hp,"/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)),os.getcwd())
DATA=os.path.join(ROOT,"data","processed","sessionwise")
OUT=os.path.join(ROOT,"outputs","work3","14_theta_mitigation"); os.makedirs(OUT,exist_ok=True)
FS=200; BANDS=[("delta",0.5,4),("theta",4,8),("alpha",8,13),("beta",13,30),("gamma",30,45)]
THETA_IDX=1  # order in BANDS

sess={}
for f in sorted(glob.glob(os.path.join(DATA,"*.npz"))):
    d=np.load(f,allow_pickle=True)
    if not all(k in d for k in ("X","y_subject","y_session")): continue
    sid=int(np.unique(d["y_session"])[0])
    if sid in sess: continue
    sess[sid]=(np.asarray(d["X"],np.float32),np.asarray(d["y_subject"],int))
SUBS=sorted(np.unique(sess[1][1]).tolist())

def bandfeat(X):   # (N,C,5) log band power
    f,P=welch(X.astype(np.float64),fs=FS,nperseg=min(200,X.shape[-1]),axis=-1); out=[]
    for _,lo,hi in BANDS:
        mask=(f>=lo)&(f<hi); out.append(np.log(P[:,:,mask].sum(-1)+1e-12))
    return np.stack(out,-1)   # (N,C,5)

B={s:bandfeat(sess[s][0]) for s in sess}; Y={s:sess[s][1] for s in sess}
# enrolment-set standardisation (per channel-band), computed on session 1
mu=B[1].reshape(-1,B[1].shape[1],5).mean(0); sd=B[1].reshape(-1,B[1].shape[1],5).std(0)+1e-8
def feats(s, theta_norm):
    F=(B[s]-mu)/sd
    if theta_norm:  # remove per-session theta offset (unsupervised, uses only that session's own stats)
        th=F[:,:,THETA_IDX]; F=F.copy(); F[:,:,THETA_IDX]=(th-th.mean(0))/(th.std(0)+1e-8)
    return F.reshape(F.shape[0],-1)
def eer(gen,imp):
    y=np.r_[np.ones(len(gen)),np.zeros(len(imp))]; s=np.r_[gen,imp]
    fpr,tpr,_=roc_curve(y,s); fnr=1-tpr; i=np.nanargmin(np.abs(fpr-fnr)); return float((fpr[i]+fnr[i])/2)
def persub_cross_eer(theta_norm):
    F1=feats(1,theta_norm)
    protos={s:(F1[Y[1]==s].mean(0)) for s in SUBS}
    for s in protos: protos[s]=protos[s]/(np.linalg.norm(protos[s])+1e-9)
    out={}
    for te in [2,3]:
        Ft=feats(te,theta_norm); Ft=Ft/(np.linalg.norm(Ft,axis=1,keepdims=True)+1e-9)
        for sub in SUBS:
            q=Ft[Y[te]==sub]
            if len(q)<1: continue
            gen=(q@protos[sub]).tolist(); imp=[]
            for o in SUBS:
                if o!=sub: imp.extend((q@protos[o]).tolist())
            out.setdefault(sub,[]).append(eer(np.array(gen),np.array(imp)))
    return {s:np.mean(v) for s,v in out.items()}

base=persub_cross_eer(False); thn=persub_cross_eer(True)
rows=[{"subject":s,"baseline_EER":round(base[s],4),"theta_norm_EER":round(thn[s],4),
       "delta":round(thn[s]-base[s],4)} for s in SUBS if s in base and s in thn]
df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,"theta_mitigation.csv"),index=False)
w=stats.wilcoxon(df["baseline_EER"],df["theta_norm_EER"])
print("="*84+"\n[C15] theta-aware (unsupervised) normalisation vs baseline")
print(df.to_string(index=False))
print(f"\n  mean baseline={df['baseline_EER'].mean():.4f}  theta-norm={df['theta_norm_EER'].mean():.4f}  "
      f"mean Δ={df['delta'].mean():+.4f}  Wilcoxon p={w.pvalue:.3g}")
print("  reference (paper, supervised re-enrolment): S2 -9.0% p=0.006 ; S3 -10.9% p=2e-4")
print("  READ-OFF: if theta-norm lowers EER (Δ<0, p<0.05), the mitigation is connected to the theta mechanism (answers C15).")
print("  If it does not help, report honestly: theta is a correlate; unsupervised theta-normalisation is insufficient, re-enrolment remains the practical fix.")
print("DONE ->",os.path.relpath(OUT,ROOT))
