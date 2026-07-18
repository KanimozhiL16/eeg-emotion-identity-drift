#!/usr/bin/env python3
import os, glob, numpy as np, pandas as pd, numpy.linalg as la
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def _has(p): return os.path.isdir(os.path.join(p,"outputs"))
_hp=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
ROOT=next((c for c in [os.getcwd(),_hp,"/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)),os.getcwd())
OUT=os.path.join(ROOT,"outputs","run_18_changepoint_ci"); os.makedirs(OUT,exist_ok=True)
RNG=np.random.default_rng(123); NPERM=10000; NB=2000
def fit(x,y,c):
    X=np.c_[np.ones_like(x),x,np.clip(x-c,0,None)];b,*_=la.lstsq(X,y,rcond=None);r=y-X@b;return float(r@r),b
def line_rss(x,y):
    X=np.c_[np.ones_like(x),x];b,*_=la.lstsq(X,y,rcond=None);r=y-X@b;return float(r@r)
def bp_fine(x,y,ng=400):
    lo,hi=np.quantile(x,[0.08,0.92]);cand=np.linspace(lo,hi,ng);rss=[fit(x,y,c)[0] for c in cand]
    i=int(np.argmin(rss));return float(cand[i]),rss[i]
def analyse(x,y):
    o=np.argsort(x);x,y=x[o],y[o];n=len(x)
    bp,rpw=bp_fine(x,y);rl=line_rss(x,y);gain=1-rpw/rl;obs=rl-rpw
    ge=sum((line_rss(x,(yp:=RNG.permutation(y)))-bp_fine(x,yp)[1])>=obs for _ in range(NPERM))
    p=(1+ge)/(NPERM+1)
    bs=np.empty(NB)
    for j in range(NB):
        idx=RNG.integers(0,n,n);xs=x[idx];ys=y[idx];oo=np.argsort(xs);bs[j],_=bp_fine(xs[oo],ys[oo],200)
    lo,hi=np.percentile(bs,[2.5,97.5]);_,b=fit(x,y,bp)
    return dict(n=n,mean=float(x.mean()),bp=bp,gain=100*gain,p=p,lo=lo,hi=hi,x=x,y=y,coef=b)
f=sorted(glob.glob(os.path.join(ROOT,"outputs","**","merged_global_psd_identity_eer.csv"),recursive=True),key=len)[0]
df=pd.read_csv(f)
if "variant" in df: df=df[df.variant=="arcface_supcon_cnn"].copy()
dc="total_psd_drift"
d=df[[dc,"EER"]].dropna()
N=analyse(d[dc].to_numpy(float), d["EER"].to_numpy(float))
g=df.dropna(subset=[dc,"EER"]).groupby(["test_session","enroll_emotion","test_emotion"])[[dc,"EER"]].mean().reset_index()
C=analyse(g[dc].to_numpy(float), g["EER"].to_numpy(float))
print("="*90)
print("UNIFIED change-point on Table-3 data (identity-drift index = total_psd_drift, arcface_supcon_cnn)")
print("="*90)
print(" NAIVE per-seed  : n=%3d  breakpoint=%.4f  95%% CI [%.4f, %.4f]  gain=%.1f%%  perm p=%.5f"%(N['n'],N['bp'],N['lo'],N['hi'],N['gain'],N['p']))
print(" SEED-COLLAPSED  : n=%3d  breakpoint=%.4f  95%% CI [%.4f, %.4f]  gain=%.1f%%  perm p=%.5f"%(C['n'],C['bp'],C['lo'],C['hi'],C['gain'],C['p']))
print(" mean drift index=%.4f"%N['mean'])
fig,ax=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
xN=N['x'];bp=N['bp'];b=N['coef'];xx=np.linspace(xN.min(),xN.max(),200)
yy=np.c_[np.ones_like(xx),xx,np.clip(xx-bp,0,None)]@b
ax[0].scatter(xN,N['y'],s=8,alpha=0.25,color="#3b6ea5");ax[0].plot(xx,yy,color="#c0392b",lw=2)
ax[0].axvline(bp,ls=":",color="#c0392b",lw=1.8)
ax[0].set_title("Naive: per-seed rows as independent\n(n=%d, two-phase p=%.4f)"%(N['n'],N['p']),fontsize=9)
ax[0].set_xlabel("Identity-drift index (summed band-power drift)");ax[0].set_ylabel("EER");ax[0].grid(alpha=.3)
xc=C['x'];A=np.c_[np.ones_like(xc),xc];bl,*_=la.lstsq(A,C['y'],rcond=None)
ax[1].scatter(xc,C['y'],s=28,alpha=0.8,color="#2e7d32");ax[1].plot(xc,A@bl,color="#000",lw=2,label="single-line fit")
ax[1].set_title("Controlled: %d independent conditions\n(two-phase n.s., p=%.2f -> monotonic)"%(C['n'],C['p']),fontsize=9)
ax[1].set_xlabel("Identity-drift index (summed band-power drift)");ax[1].grid(alpha=.3);ax[1].legend(fontsize=8)
plt.tight_layout();fp=os.path.join(OUT,"Fig03b_change_point.png");plt.savefig(fp,dpi=300);plt.close()
print("\nSAVED figure ->", os.path.relpath(fp,ROOT))
print("="*90)
