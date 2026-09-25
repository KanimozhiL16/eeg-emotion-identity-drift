#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
seediv_coral_baseline.py  --  representative domain-adaptation baseline under the SAME
leakage-controlled, trial-disjoint enrol-then-verify protocol (supervisor comment #10).

CORAL (CORrelation ALignment; Sun et al., AAAI 2016) is a standard, training-free,
UNSUPERVISED domain-adaptation method: it aligns the second-order statistics (covariance)
of the enrolment-session features to those of the target session, using NO target identity
labels. We apply it to the paper's PSD (5-band Welch, 310-D) features and score with the
identical cosine matcher, so the ONLY change versus the paper's baseline is the feature
alignment. This gives a fair "does a representative DA method close the cross-session gap?"
reference row, measured the same leakage-free way (S1 template -> S2/S3 probes; trial-disjoint
by construction because sessions are different days).

RUN (laptop, PowerShell):
    python "seediv_coral_baseline.py"
Then paste the console output; the printed rows are ready to add as a comparison row.
"""
import os, glob, numpy as np
from scipy.signal import welch
from scipy.linalg import sqrtm
from sklearn.metrics import roc_curve, roc_auc_score

RNG=np.random.default_rng(0)
DATA=r"C:\Users\L.KANIMOZHI\Downloads\P4_SEEDIV_LOCAL"
BANDS=[(0.5,4),(4,8),(8,13),(13,30),(30,45)]; CAP=400
print("="*76); print("CORAL domain-adaptation baseline (leakage-controlled) | data:",DATA); print("="*76)

def load(n):
    z=np.load(os.path.join(DATA,f"seediv_session{n}.npz"),allow_pickle=True)
    return z["X"].astype(np.float32), z["y_subject"].astype(int)
def psd(X):
    f,P=welch(X,fs=200,nperseg=200,axis=-1)
    return np.concatenate([np.log(P[:,:,(f>=lo)&(f<hi)].sum(-1)+1e-12) for lo,hi in BANDS],1).astype(np.float64)
def l2(F): return F/(np.linalg.norm(F,axis=1,keepdims=True)+1e-8)
def eer(y,s):
    fpr,tpr,_=roc_curve(y,s); fnr=1-tpr; i=int(np.nanargmin(np.abs(fpr-fnr)))
    return float((fpr[i]+fnr[i])/2), float(roc_auc_score(y,s))
def coral(Xs,Xt,eps=1e-4):
    """Align source Xs covariance to target Xt covariance (unsupervised; no target labels)."""
    d=Xs.shape[1]
    Cs=np.cov(Xs,rowvar=False)+eps*np.eye(d); Ct=np.cov(Xt,rowvar=False)+eps*np.eye(d)
    A=np.real(sqrtm(np.linalg.inv(Cs))@sqrtm(Ct))
    return Xs@A

def cross_session_eer(Fs,ys,Ft,yt,align=False,Ft_ref=None):
    Xs=Fs.copy()
    if align:   # CORAL: align source features to target-session covariance (labels not used)
        Xs=coral(Fs, Ft_ref if Ft_ref is not None else Ft)
    subs=sorted(np.unique(ys).tolist()); yscore=[]; sc=[]
    for s in subs:
        en=np.where(ys==s)[0]
        if len(en)>CAP: en=RNG.choice(en,CAP,replace=False)
        templ=l2(l2(Xs[en]).mean(0)[None,:])[0]
        gi=np.where(yt==s)[0]; ii=np.where(yt!=s)[0]
        if len(gi)<5 or len(ii)<20: continue
        if len(ii)>CAP: ii=RNG.choice(ii,CAP,replace=False)
        for v in l2(Ft[gi])@templ: yscore.append(1); sc.append(float(v))
        for v in l2(Ft[ii])@templ: yscore.append(0); sc.append(float(v))
    return eer(np.array(yscore),np.array(sc))

X1,s1=load(1); X2,s2=load(2); X3,s3=load(3)
F1,F2,F3=psd(X1),psd(X2),psd(X3)
print("\n  Cross-session EER (S1 enrol -> later-session verify), pooled:")
for name,Ft,yt in [("S1->S2",F2,s2),("S1->S3",F3,s3)]:
    b=cross_session_eer(F1,s1,Ft,yt,align=False)
    c=cross_session_eer(F1,s1,Ft,yt,align=True,Ft_ref=Ft)
    print(f"    {name}:  PSD+cosine baseline EER={b[0]:.4f} (AUC {b[1]:.3f})   |   "
          f"CORAL-aligned EER={c[0]:.4f} (AUC {c[1]:.3f})   dEER={c[0]-b[0]:+.4f}")
print("\n  Interpretation: if CORAL EER < baseline, a representative unsupervised DA method")
print("  reduces (but is not expected to fully close) the cross-session gap under the SAME")
print("  leakage-controlled protocol; report the row honestly whatever the direction.")
print("="*76); print("PASTE THIS OUTPUT BACK to add the DA-baseline row to the paper.")
