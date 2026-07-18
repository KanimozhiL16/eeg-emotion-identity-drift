#!/usr/bin/env python3
"""
17_index_monotonic_cv.py -- CORRECTED C10 / C19 / C18 using the paper's ACTUAL identity-drift index
(total_psd_drift), plus theta-only and 4-band comparisons. Zero-risk, merged CSV only, seconds.
Reason: step 10 used the summed 4-band drift; the manuscript's index is total_psd_drift.
Writes: outputs/work3/17_index_monotonic_cv/index_cv.csv
Run: python -u scripts/work3/17_index_monotonic_cv.py 2>&1 | tee outputs/work3/logs/17_index.log
"""
import os, glob, numpy as np, pandas as pd
from scipy import stats

def _has(p): return os.path.isdir(os.path.join(p,"outputs"))
_hp=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
ROOT=next((c for c in [os.getcwd(),_hp,"/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)),os.getcwd())
OUT=os.path.join(ROOT,"outputs","work3","17_index_monotonic_cv"); os.makedirs(OUT,exist_ok=True)
CSV=sorted(glob.glob(os.path.join(ROOT,"outputs","**","merged_global_psd_identity_eer.csv"),recursive=True),key=len)[0]
df=pd.read_csv(CSV)
if "variant" in df.columns and df["variant"].nunique()>1 and (df["variant"]=="arcface_supcon_cnn").any():
    df=df[df["variant"]=="arcface_supcon_cnn"].copy()
for c in ["enroll_emotion","test_emotion","enroll_session","test_session"]:
    if c in df.columns: df[c]=df[c].astype(str)
df["emotion_pair"]=df.get("enroll_emotion","")+"->"+df.get("test_emotion","")
if "transition" not in df.columns:
    df["transition"]=df.get("enroll_session","")+"->"+df.get("test_session","")
bands=[c for c in ["theta_power_drift","alpha_power_drift","beta_power_drift","gamma_power_drift"] if c in df.columns]
INDEX="total_psd_drift" if "total_psd_drift" in df.columns else None
# seed-collapse to independent conditions
gcols=[c for c in ([INDEX]+bands+["EER"]) if c]
g=df.groupby(["transition","emotion_pair"])[gcols].mean().reset_index()
print("="*84); print(f"source: {os.path.relpath(CSV,ROOT)} | seed-collapsed conditions n={len(g)}")
print(f"paper identity-drift index column: {INDEX}")

def isotonic_r2(x,y):
    try:
        from sklearn.isotonic import IsotonicRegression
        o=np.argsort(x); xs,ys=np.asarray(x)[o],np.asarray(y)[o]
        yh=IsotonicRegression(increasing="auto").fit(xs,ys).predict(xs)
        return 1-np.sum((ys-yh)**2)/np.sum((ys-ys.mean())**2)
    except Exception: return float("nan")
def lin_r2(x,y):
    b=np.polyfit(x,y,1); return 1-np.sum((y-np.polyval(b,x))**2)/np.sum((y-y.mean())**2)

y=g["EER"].to_numpy(float)
print("\n"+"="*84+"\n[C10 & C19] monotonicity on candidate indices")
rows=[]
cands=[]
if INDEX: cands.append((INDEX, g[INDEX].to_numpy(float)))
cands.append(("summed_4band", g[bands].abs().sum(1).to_numpy(float)))
sig=g[bands].std(ddof=0).replace(0,1.0)
cands.append(("sigma_norm_4band",(g[bands].abs()/sig).sum(1).to_numpy(float)))
cands.append(("theta_only", g["theta_power_drift"].to_numpy(float)))
for name,x in cands:
    rho,pr=stats.spearmanr(x,y); tau,pt=stats.kendalltau(x,y)
    r2l=lin_r2(x,y); r2i=isotonic_r2(x,y)
    star="**" if pr<0.05 else "ns"
    print(f"  {name:18s} Spearman rho={rho:+.3f} p={pr:.3g} {star} | Kendall tau={tau:+.3f} p={pt:.3g} | linR2={r2l:.3f} isoR2={r2i:.3f}")
    rows.append({"index":name,"spearman_rho":round(rho,3),"spearman_p":float(pr),
                 "kendall_tau":round(tau,3),"kendall_p":float(pt),"lin_R2":round(r2l,3),"iso_R2":round(r2i,3)})
print("  READ-OFF: use the PAPER index (total_psd_drift) row to judge 'monotonic'. If its Spearman is +ve & p<0.05,")
print("  the monotonic claim holds on the paper's own index; if n.s., soften to a theta-SPECIFIC (not aggregate) effect.")

print("\n"+"="*84+"\n[C18] cross-validated EER prediction (out-of-sample) by feature set")
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    feats={"total_psd_drift":[INDEX] if INDEX else None,"theta_only":["theta_power_drift"],"4band":bands}
    groups={"leave-one-emotion-pair-out":g["emotion_pair"].to_numpy(),"leave-one-transition-out":g["transition"].to_numpy()}
    for fname,cols in feats.items():
        if not cols: continue
        X=g[cols].to_numpy(float)
        for gname,gv in groups.items():
            pred=np.full(len(y),np.nan)
            for u in np.unique(gv):
                tr=gv!=u; te=gv==u
                if tr.sum()<3: continue
                pred[te]=LinearRegression().fit(X[tr],y[tr]).predict(X[te])
            mkeep=~np.isnan(pred)
            mae=mean_absolute_error(y[mkeep],pred[mkeep]); rmse=mean_squared_error(y[mkeep],pred[mkeep])**0.5
            r2=1-np.sum((y[mkeep]-pred[mkeep])**2)/np.sum((y[mkeep]-y[mkeep].mean())**2)
            print(f"  feats={fname:16s} {gname:26s} MAE={mae:.4f} RMSE={rmse:.4f} predR2={r2:+.3f}")
            rows.append({"index":f"CV[{fname}|{gname}]","MAE":round(mae,4),"RMSE":round(rmse,4),"pred_R2":round(r2,3)})
    print("  READ-OFF: 'predictable' needs predR2>0 out-of-sample for at least one sensible feature set; else 'systematically associated'.")
except Exception as e:
    print("  sklearn needed:",e)
pd.DataFrame(rows).to_csv(os.path.join(OUT,"index_cv.csv"),index=False)
print("\nDONE ->",os.path.relpath(OUT,ROOT))
