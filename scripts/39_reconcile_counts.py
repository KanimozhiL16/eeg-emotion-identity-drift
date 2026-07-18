#!/usr/bin/env python3
import os, glob, numpy as np, pandas as pd, numpy.linalg as la
def _has(p): return os.path.isdir(os.path.join(p,"outputs"))
_hp=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
ROOT=next((c for c in [os.getcwd(),_hp,"/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)),os.getcwd())
RNG=np.random.default_rng(123)
def fit(x,y,c):
    X=np.c_[np.ones_like(x),x,np.clip(x-c,0,None)];b,*_=la.lstsq(X,y,rcond=None);r=y-X@b;return float(r@r)
def bp_fine(x,y,ng=400):
    lo,hi=np.quantile(x,[0.08,0.92]);cand=np.linspace(lo,hi,ng);rss=[fit(x,y,c) for c in cand]
    return float(cand[int(np.argmin(rss))])
def ci(x,y,nb=2000):
    o=np.argsort(x);x,y=x[o],y[o];n=len(x);bs=np.empty(nb)
    for j in range(nb):
        idx=RNG.integers(0,n,n);xs=x[idx];ys=y[idx];oo=np.argsort(xs);bs[j]=bp_fine(xs[oo],ys[oo],200)
    return bp_fine(x,y), np.percentile(bs,[2.5,97.5])
def show(f,label):
    df=pd.read_csv(f); print("\n==",label,"==\n ",os.path.relpath(f,ROOT)); print("  cols:",list(df.columns)); print("  n=",len(df))
    for c in df.columns:
        cl=c.lower()
        if cl in ("seed","enroll_session","test_session","enroll_emotion","test_emotion","transition","emotion_pair","variant"):
            u=sorted(map(str,pd.unique(df[c].dropna()))); print("   %-16s %d unique: %s"%(c,len(u),u[:12]))
    ec=[c for c in df.columns if c.lower() in ("enroll_emotion","test_emotion")]
    if len(ec)==2:
        vc=df.groupby(ec).size(); print("   emotion-pairs:",vc.shape[0]," rows/pair (min,med,max):",int(vc.min()),int(vc.median()),int(vc.max()))
    return df
fcp=glob.glob(os.path.join(ROOT,"outputs","**","cross_emotion_results_all_seeds.csv"),recursive=True)[0]
dcp=show(fcp,"CHANGE-POINT table"); dcx="mean_drift"
b,c=ci(dcp[dcx].to_numpy(float),dcp["EER"].to_numpy(float))
print("   NAIVE n=%d  breakpoint=%.4f  95%% CI [%.4f, %.4f]"%(len(dcp),b,c[0],c[1]))
g=dcp.groupby(["enroll_emotion","test_emotion"])[[dcx,"EER"]].mean().reset_index()
b2,c2=ci(g[dcx].to_numpy(float),g["EER"].to_numpy(float))
print("   COLLAPSED n=%d  breakpoint=%.4f  95%% CI [%.4f, %.4f]"%(len(g),b2,c2[0],c2[1]))
freg=glob.glob(os.path.join(ROOT,"outputs","**","merged_global_psd_identity_eer.csv"),recursive=True)
if freg: show(sorted(freg,key=len)[0],"REGRESSION table (Table 3)")
print("\nDONE")
