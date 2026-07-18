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
def bp_fine(x,y,ngrid=400):
    lo,hi=np.quantile(x,[0.08,0.92]);cand=np.linspace(lo,hi,ngrid)
    rss=[fit(x,y,c)[0] for c in cand];i=int(np.argmin(rss));return float(cand[i]),rss[i]
def analyse(x,y):
    o=np.argsort(x);x,y=x[o],y[o];n=len(x)
    bp,rpw=bp_fine(x,y);rl=line_rss(x,y);gain=1-rpw/rl;obs=rl-rpw
    ge=sum((line_rss(x,(yp:=RNG.permutation(y)))-bp_fine(x,yp)[1])>=obs for _ in range(NPERM))
    p=(1+ge)/(NPERM+1)
    bs=np.empty(NB)
    for j in range(NB):
        idx=RNG.integers(0,n,n);xs=x[idx];ys=y[idx];oo=np.argsort(xs);bs[j],_=bp_fine(xs[oo],ys[oo],200)
    lo,hi=np.percentile(bs,[2.5,97.5])
    _,b=fit(x,y,bp); slope1=b[1]; slope2=b[1]+b[2]
    return dict(n=n,mean=float(x.mean()),bp=bp,gain=100*gain,ratio=rl/rpw,p=p,ci=(lo,hi),
                slope_lo=slope1,slope_hi=slope2,x=x,y=y,coef=b)
def find(colnames, mean_range):
    for f in glob.glob(os.path.join(ROOT,"outputs","**","*.csv"),recursive=True):
        try: df=pd.read_csv(f)
        except Exception: continue
        dc=next((c for c in df.columns if c.lower() in colnames and pd.api.types.is_numeric_dtype(df[c])),None)
        ec=next((c for c in df.columns if c.lower()=="eer" and pd.api.types.is_numeric_dtype(df[c])),None)
        if dc and ec and mean_range[0]<df[dc].dropna().mean()<mean_range[1]:
            return f,df,dc,ec
    return None,None,None,None
print("="*100); print("A)  ANCHOR: identity-drift index  (Fig 5 metric; independent conditions)"); print("="*100)
f,df,dc,ec=find({"drift_index","identity_drift","drift_idx"},(0.09,0.14))
if f is None:
    f,df,dc,ec=find({"mean_drift","total_psd_drift","transition_drift"},(0.09,0.14))
print("table :",os.path.relpath(f,ROOT)); print("cols  :",list(df.columns))
print("using : x=%s  y=%s   rows=%d   has_seed=%s"%(dc,ec,len(df),any(c.lower()=="seed" for c in df.columns)))
print(df[[dc,ec]].describe().loc[["count","mean","min","max"]].to_string())
A=analyse(df[[dc,ec]].dropna()[dc].to_numpy(float), df[[dc,ec]].dropna()[ec].to_numpy(float))
print("\n>>> breakpoint = %.4f   95%% CI [%.4f, %.4f]   n=%d"%(A["bp"],A["ci"][0],A["ci"][1],A["n"]))
print(">>> fit gain  = %.1f%% lower RSS than a single line   (RSS ratio %.3f)"%(A["gain"],A["ratio"]))
print(">>> slopes    : below bp = %.3f  ->  above bp = %.3f   (EER rises %.1fx faster after)"
      %(A["slope_lo"],A["slope_hi"], (A["slope_hi"]/A["slope_lo"] if A["slope_lo"] else float('nan'))))
print(">>> permutation p (%d) = %.5f"%(NPERM,A["p"]))
print("\n"+"="*100); print("B)  RIGOR CHECK: naive per-seed  vs  seed-collapsed (pseudoreplication control)"); print("="*100)
fs=None
for f2 in glob.glob(os.path.join(ROOT,"outputs","**","cross_emotion_results_all_seeds.csv"),recursive=True): fs=f2
if fs:
    dd=pd.read_csv(fs); sd=next((c for c in dd.columns if c.lower()=="seed"),None)
    dcx=next((c for c in dd.columns if c.lower() in ("mean_drift","transition_drift","drift_index")),None)
    gk=[c for c in dd.columns if c.lower() in ("enroll_emotion","test_emotion","enroll_session","test_session","transition","emotion_pair")]
    print("table :",os.path.relpath(fs,ROOT)," seed=",sd," drift=",dcx," group=",gk)
    R=analyse(dd[dcx].to_numpy(float), dd["EER"].to_numpy(float))
    g=dd.groupby(gk)[[dcx,"EER"]].mean().reset_index()
    C=analyse(g[dcx].to_numpy(float), g["EER"].to_numpy(float))
    print(" naive per-seed        : n=%3d  bp=%.3f  gain=%4.1f%%  p=%.5f"%(R["n"],R["bp"],R["gain"],R["p"]))
    print(" seed-collapsed (indep): n=%3d  bp=%.3f  gain=%4.1f%%  p=%.5f"%(C["n"],C["bp"],C["gain"],C["p"]))
x,y,b,bp,ci=A["x"],A["y"],A["coef"],A["bp"],A["ci"]
xs=np.linspace(x.min(),x.max(),200); yh=np.c_[np.ones_like(xs),xs,np.clip(xs-bp,0,None)]@b
plt.figure(figsize=(5.2,4))
plt.scatter(x,y,s=14,alpha=0.5,color="#3b6ea5",label="conditions")
plt.plot(xs,yh,color="#c0392b",lw=2,label="two-phase fit")
plt.axvline(bp,ls=":",color="#c0392b",lw=2,label=f"breakpoint = {bp:.3f}")
plt.axvspan(ci[0],ci[1],color="#c0392b",alpha=0.12,label="breakpoint 95% CI")
plt.axvline(x.mean(),ls="--",color="0.4",lw=1.4,label=f"mean drift = {x.mean():.3f}")
plt.xlabel("Identity-drift index (summed band-power drift)"); plt.ylabel("Equal error rate (EER)")
plt.title(f"Change-point (permutation p = {A['p']:.4f}, n={A['n']})"); plt.legend(fontsize=7); plt.grid(alpha=.3)
plt.tight_layout(); fp=os.path.join(OUT,"Fig03b_change_point.png"); plt.savefig(fp,dpi=300); plt.close()
pd.DataFrame([{"anchor_table":os.path.relpath(f,ROOT),"drift_col":dc,"n":A["n"],"breakpoint":A["bp"],
   "ci_lo":A["ci"][0],"ci_hi":A["ci"][1],"gain_pct":A["gain"],"perm_p":A["p"],"mean":A["mean"],
   "slope_below":A["slope_lo"],"slope_above":A["slope_hi"]}]).to_csv(os.path.join(OUT,"changepoint_final.csv"),index=False)
print("\nSAVED figure -> outputs/run_18_changepoint_ci/Fig03b_change_point.png")
print("SAVED stats  -> outputs/run_18_changepoint_ci/changepoint_final.csv")
print("="*100)
