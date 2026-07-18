#!/usr/bin/env python3
import os, glob, numpy as np, pandas as pd, numpy.linalg as la
def _has(p): return os.path.isdir(os.path.join(p,"outputs"))
_hp=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
ROOT=next((c for c in [os.getcwd(),_hp,"/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)),os.getcwd())
OUT=os.path.join(ROOT,"outputs","run_18_changepoint_ci"); os.makedirs(OUT,exist_ok=True)
RNG=np.random.default_rng(123); NPERM=10000; NB=2000
def fit_rss(x,y,c):
    X=np.c_[np.ones_like(x),x,np.clip(x-c,0,None)];b,*_=la.lstsq(X,y,rcond=None);r=y-X@b;return float(r@r)
def line_rss(x,y):
    X=np.c_[np.ones_like(x),x];b,*_=la.lstsq(X,y,rcond=None);r=y-X@b;return float(r@r)
def bp_search(x,y):
    c=np.unique(np.quantile(x,np.linspace(0.10,0.90,60)));r=[fit_rss(x,y,cc) for cc in c]
    i=int(np.argmin(r));return float(c[i]),r[i]
def full(x,y):
    o=np.argsort(x);x,y=x[o],y[o];n=len(x)
    bp,rpw=bp_search(x,y);rl=line_rss(x,y);imp=1-rpw/rl;obs=rl-rpw
    ge=sum((line_rss(x,(yp:=RNG.permutation(y)))-bp_search(x,yp)[1])>=obs for _ in range(NPERM))
    p=(1+ge)/(NPERM+1)
    bs=np.empty(NB)
    for j in range(NB):
        idx=RNG.integers(0,n,n);xs=x[idx];ys=y[idx];oo=np.argsort(xs);bs[j],_=bp_search(xs[oo],ys[oo])
    lo,hi=np.percentile(bs,[2.5,97.5])
    return dict(n=n,mean_x=float(x.mean()),bp=bp,improve=100*imp,p=p,ci_lo=lo,ci_hi=hi)
DR=("drift","psd","index")
isd=lambda c:any(k in c.lower() for k in DR) and "auc" not in c.lower()
ise=lambda c:c.lower()=="eer" or c.lower().startswith("eer")
rows=[]
for f in glob.glob(os.path.join(ROOT,"outputs","**","*.csv"),recursive=True):
    try: df=pd.read_csv(f)
    except Exception: continue
    dcols=[c for c in df.columns if isd(c) and pd.api.types.is_numeric_dtype(df[c])]
    ecols=[c for c in df.columns if ise(c) and pd.api.types.is_numeric_dtype(df[c])]
    for dc in dcols:
        for ec in ecols:
            d=df[[dc,ec]].dropna()
            if len(d)<12 or not (0.02<d[dc].mean()<0.4): continue
            r=full(d[dc].to_numpy(float),d[ec].to_numpy(float))
            r.update(file=os.path.relpath(f,ROOT),drift=dc,mode="raw")
            rows.append(r)
            seedcol=next((c for c in df.columns if c.lower()=="seed"),None)
            gcols=[c for c in df.columns if c.lower() in ("enroll_emotion","test_emotion",
                    "enroll_session","test_session","transition","emotion_pair","condition")]
            if seedcol and gcols:
                g=df.dropna(subset=[dc,ec]).groupby(gcols)[[dc,ec]].mean().reset_index()
                if len(g)>=12:
                    r2=full(g[dc].to_numpy(float),g[ec].to_numpy(float))
                    r2.update(file=os.path.relpath(f,ROOT),drift=dc,mode="seed-collapsed(n=%d)"%len(g))
                    rows.append(r2)
t=pd.DataFrame(rows)
t["bp_gap"]=(t.bp-0.124).abs()
t=t.sort_values(["bp_gap","p"])
pd.set_option("display.width",200,"display.max_colwidth",60)
print("="*118)
print("CHANGE-POINT DIAGNOSTIC  (manuscript target: breakpoint~0.124, improve~8%, published p=0.001)")
print("="*118)
print(t.to_string(index=False,columns=["file","drift","mode","n","mean_x","bp","improve","p","ci_lo","ci_hi"],
      float_format=lambda v:f"{v:.4f}"))
t.to_csv(os.path.join(OUT,"changepoint_diagnostic.csv"),index=False)
print("\nSAVED -> outputs/run_18_changepoint_ci/changepoint_diagnostic.csv")
print("Read: find the row whose bp~0.124 AND improve~8%. Compare its raw p vs its seed-collapsed p.")
print("="*118)
