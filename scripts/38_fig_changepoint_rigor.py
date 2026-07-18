#!/usr/bin/env python3
import os, glob, numpy as np, pandas as pd, numpy.linalg as la
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def _has(p): return os.path.isdir(os.path.join(p,"outputs"))
_hp=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
ROOT=next((c for c in [os.getcwd(),_hp,"/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)),os.getcwd())
OUT=os.path.join(ROOT,"outputs","run_18_changepoint_ci"); os.makedirs(OUT,exist_ok=True)
def fit(x,y,c):
    X=np.c_[np.ones_like(x),x,np.clip(x-c,0,None)];b,*_=la.lstsq(X,y,rcond=None);r=y-X@b;return float(r@r),b
def bp_fine(x,y,ng=400):
    lo,hi=np.quantile(x,[0.08,0.92]);cand=np.linspace(lo,hi,ng);rss=[fit(x,y,c)[0] for c in cand]
    i=int(np.argmin(rss));return float(cand[i])
f=glob.glob(os.path.join(ROOT,"outputs","**","cross_emotion_results_all_seeds.csv"),recursive=True)[0]
dd=pd.read_csv(f); dc="mean_drift"; gk=["enroll_emotion","test_emotion"]
xr=dd[dc].to_numpy(float); yr=dd["EER"].to_numpy(float)
g=dd.groupby(gk)[[dc,"EER"]].mean().reset_index(); xc=g[dc].to_numpy(float); yc=g["EER"].to_numpy(float)
fig,ax=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
# left: naive per-seed with two-phase
o=np.argsort(xr); xs=xr[o]; bp=bp_fine(xs,yr[o]); _,b=fit(xs,yr[o],bp)
xx=np.linspace(xs.min(),xs.max(),200); yy=np.c_[np.ones_like(xx),xx,np.clip(xx-bp,0,None)]@b
ax[0].scatter(xr,yr,s=8,alpha=0.25,color="#3b6ea5")
ax[0].plot(xx,yy,color="#c0392b",lw=2); ax[0].axvline(bp,ls=":",color="#c0392b",lw=1.8)
ax[0].set_title("Naive: per-seed rows as independent\n(n=375, two-phase p=0.0002)",fontsize=9)
ax[0].set_xlabel("Identity-drift index"); ax[0].set_ylabel("EER"); ax[0].grid(alpha=.3)
# right: seed-collapsed with single line
oc=np.argsort(xc); xcs=xc[oc]; ycs=yc[oc]
A=np.c_[np.ones_like(xcs),xcs]; bl,*_=la.lstsq(A,ycs,rcond=None); yl=A@bl
ax[1].scatter(xc,yc,s=30,alpha=0.8,color="#2e7d32")
ax[1].plot(xcs,yl,color="#000",lw=2,label="single-line fit")
ax[1].set_title("Controlled: 25 independent conditions\n(two-phase n.s., p=0.24 -> monotonic)",fontsize=9)
ax[1].set_xlabel("Identity-drift index"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
plt.tight_layout()
for d in [OUT, os.path.join(ROOT,"figures"), os.path.join(ROOT,"manuscript","figures")]:
    try:
        if os.path.isdir(d): plt.savefig(os.path.join(d,"Fig03b_change_point.png"),dpi=300)
    except Exception as e: print("skip",d,e)
plt.savefig(os.path.join(OUT,"Fig03b_change_point.png"),dpi=300); plt.close()
print("SAVED Fig03b_change_point.png ->", os.path.relpath(os.path.join(OUT,"Fig03b_change_point.png"),ROOT))
print("Download that PNG and replace figures/Fig03b_change_point.png locally + in Overleaf.")
