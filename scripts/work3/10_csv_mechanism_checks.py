#!/usr/bin/env python3
"""
10_csv_mechanism_checks.py  --  ZERO-RISK checks from the existing merged CSV.
Addresses (from work3 comments):
  C9  delta band + broadband/total-drift control in the theta regression
  C10 normalised (sigma-scaled) identity-drift index -> refit
  C18 cross-validated EER prediction (MAE/RMSE/predictive R2)  [keeps or kills "predictable"]
  C19 formal monotonicity (Spearman + isotonic regression)      [substantiates "monotonic"]
  C8  if the CSV holds >1 'variant', theta->EER per matcher      [else run step 13]

Reads: outputs/**/merged_global_psd_identity_eer.csv  (the Table-3 data; 250 rows = 50 cond x 5 seeds)
Writes: outputs/work3/10_csv_checks/*.csv + prints READ-OFFs.
Run:  python -u scripts/work3/10_csv_mechanism_checks.py 2>&1 | tee outputs/work3/10_csv_checks.log
"""
import os, glob, numpy as np, pandas as pd, numpy.linalg as la
from scipy import stats

def _has(p): return os.path.isdir(os.path.join(p, "outputs"))
_hp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ROOT = next((c for c in [os.getcwd(), _hp,
      "/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)), os.getcwd())
OUT = os.path.join(ROOT, "outputs", "work3", "10_csv_checks"); os.makedirs(OUT, exist_ok=True)

cands = glob.glob(os.path.join(ROOT, "outputs", "**", "merged_global_psd_identity_eer.csv"), recursive=True)
if not cands: raise SystemExit("ERROR: merged_global_psd_identity_eer.csv not found under outputs/")
CSV = sorted(cands, key=len)[0]
df = pd.read_csv(CSV)
print("="*84); print("source:", os.path.relpath(CSV, ROOT), "| rows:", len(df)); print("cols:", list(df.columns))
EE = "EER"
def z(a): a = np.asarray(a, float); s = a.std(ddof=0); return (a - a.mean())/(s if s else 1.0)

# keep the manuscript variant for the primary checks
VAR = None
if "variant" in df.columns and df["variant"].nunique() > 1:
    VAR = "arcface_supcon_cnn" if (df["variant"]=="arcface_supcon_cnn").any() else df["variant"].mode()[0]
    dfp = df[df["variant"]==VAR].copy()
else:
    dfp = df.copy()
for c in ["enroll_emotion","test_emotion","enroll_session","test_session"]:
    if c in dfp.columns: dfp[c]=dfp[c].astype(str)
dfp["emotion_pair"] = dfp.get("enroll_emotion","")+"->"+dfp.get("test_emotion","")
dfp["transition"]   = dfp.get("enroll_session","")+"->"+dfp.get("test_session","")

band_cols = [c for c in df.columns if c.endswith("_power_drift")]
print("band drift columns present:", band_cols)
has_delta = any("delta" in c for c in band_cols)

def seed_collapse(d, cols):
    if "seed" in d.columns and d["seed"].nunique()>1:
        return d.groupby(["transition","emotion_pair"])[cols].mean().reset_index()
    return d[["transition","emotion_pair"]+cols].copy()

def ols_std(X_raw, y_raw):
    n=len(y_raw); X=np.c_[np.ones(n), np.column_stack([z(X_raw[:,j]) for j in range(X_raw.shape[1])])]
    y=z(y_raw); b,*_=la.lstsq(X,y,rcond=None); r=y-X@b; k=X.shape[1]
    s2=(r@r)/(n-k); se=np.sqrt(np.diag(s2*la.inv(X.T@X))); t=b/se
    p=2*(1-stats.t.cdf(np.abs(t),n-k)); R2=1-(r@r)/np.sum((y-y.mean())**2)
    return b,p,R2

# ---------- C9: delta + broadband control ----------
print("\n"+"="*84+"\n[C9] delta band + broadband/total-drift control")
reg_bands=[c for c in ["theta_power_drift","alpha_power_drift","beta_power_drift","gamma_power_drift"] if c in band_cols]
allb    =[c for c in ["delta_power_drift"]+reg_bands if c in band_cols]
g=seed_collapse(dfp, list(set(allb+[c for c in ["total_psd_drift"] if c in dfp.columns]))+[EE] if False else allb+[EE])
# base model (paper): theta alpha beta gamma
b,p,R2=ols_std(g[reg_bands].to_numpy(float), g[EE].to_numpy(float))
print(f"  base (4-band) seed-collapsed n={len(g)}  R2={R2:.3f}")
for i,c in enumerate(reg_bands): print(f"    {c:18s} beta={b[i+1]:+.3f}  p={p[i+1]:.3g}")
if has_delta:
    gd=seed_collapse(dfp, allb+[EE]); b2,p2,R22=ols_std(gd[allb].to_numpy(float), gd[EE].to_numpy(float))
    print(f"  +delta model  R2={R22:.3f}")
    for i,c in enumerate(allb): print(f"    {c:18s} beta={b2[i+1]:+.3f}  p={p2[i+1]:.3g}")
    ti=allb.index("theta_power_drift"); print(f"  READ-OFF: theta with delta in-model beta={b2[ti+1]:+.3f} p={p2[ti+1]:.3g}")
else:
    print("  delta_power_drift NOT in CSV -> recompute 5-band drift in step 11 (11_band_recompute_from_windows.py).")
if "total_psd_drift" in dfp.columns:
    gt=seed_collapse(dfp, ["theta_power_drift","total_psd_drift",EE])
    b3,p3,_=ols_std(gt[["theta_power_drift","total_psd_drift"]].to_numpy(float), gt[EE].to_numpy(float))
    print(f"  broadband control: theta beta={b3[1]:+.3f} p={p3[1]:.3g} | total_psd beta={b3[2]:+.3f} p={p3[2]:.3g}")
    print("  READ-OFF: if theta stays significant with total_psd_drift in-model, it is not just broadband distance.")

# ---------- C10: sigma-normalised drift index ----------
print("\n"+"="*84+"\n[C10] normalised identity-drift index")
g5=seed_collapse(dfp, band_cols+[EE])
D_raw = g5[band_cols].abs().sum(axis=1).to_numpy(float)          # paper index (unnormalised)
sig = g5[band_cols].std(ddof=0).replace(0,1.0)
D_norm = (g5[band_cols].abs()/sig).sum(axis=1).to_numpy(float)   # sigma-normalised
for name,D in [("unnormalised",D_raw),("sigma-normalised",D_norm)]:
    rho,pr=stats.spearmanr(D, g5[EE]); print(f"  index={name:16s} Spearman rho={rho:+.3f} p={pr:.3g}")
print("  READ-OFF: if both indices give the same positive monotone EER relation, the metric choice is not driving the result.")
pd.DataFrame({"transition":g5["transition"],"emotion_pair":g5["emotion_pair"],
              "D_unnorm":D_raw,"D_sigmanorm":D_norm,"EER":g5[EE]}).to_csv(os.path.join(OUT,"drift_index_variants.csv"),index=False)

# ---------- C19: monotonicity ----------
print("\n"+"="*84+"\n[C19] monotonicity of drift->EER")
x=D_raw; y=g5[EE].to_numpy(float); o=np.argsort(x); x,y=x[o],y[o]
rho,pr=stats.spearmanr(x,y); tau,pt=stats.kendalltau(x,y)
try:
    from sklearn.isotonic import IsotonicRegression
    iso=IsotonicRegression(increasing="auto").fit(x,y); yhat=iso.predict(x)
    r2_iso=1-np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)
except Exception as e:
    r2_iso=float("nan"); print("  (sklearn isotonic unavailable:",e,")")
b_lin=np.polyfit(x,y,1); r2_lin=1-np.sum((y-np.polyval(b_lin,x))**2)/np.sum((y-y.mean())**2)
print(f"  Spearman rho={rho:+.3f} p={pr:.3g} | Kendall tau={tau:+.3f} p={pt:.3g}")
print(f"  linear R2={r2_lin:.3f} | isotonic R2={r2_iso:.3f}  (close => monotone well described by a rising line)")
print("  READ-OFF: positive significant Spearman+Kendall and isotonic~linear R2 justify 'monotonic'.")

# ---------- C18: cross-validated prediction ----------
print("\n"+"="*84+"\n[C18] cross-validated EER prediction from spectral drift")
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    feat=reg_bands; Xg=g5[feat].to_numpy(float) if all(c in g5 for c in feat) else seed_collapse(dfp,reg_bands+[EE])[feat].to_numpy(float)
    gg=seed_collapse(dfp,reg_bands+[EE]); Xg=gg[reg_bands].to_numpy(float); yg=gg[EE].to_numpy(float)
    groups={"leave-one-emotion-pair-out":gg["emotion_pair"].to_numpy(),
            "leave-one-transition-out":gg["transition"].to_numpy()}
    rows=[]
    for gname,gv in groups.items():
        preds=np.full(len(yg),np.nan)
        for u in np.unique(gv):
            tr=gv!=u; te=gv==u
            if tr.sum()<3: continue
            preds[te]=LinearRegression().fit(Xg[tr],yg[tr]).predict(Xg[te])
        m=~np.isnan(preds); mae=mean_absolute_error(yg[m],preds[m]); rmse=mean_squared_error(yg[m],preds[m])**0.5
        ss=1-np.sum((yg[m]-preds[m])**2)/np.sum((yg[m]-yg[m].mean())**2)
        rows.append({"cv":gname,"n":int(m.sum()),"MAE":round(mae,4),"RMSE":round(rmse,4),"pred_R2":round(ss,3)})
        print(f"  {gname:28s} n={int(m.sum())}  MAE={mae:.4f}  RMSE={rmse:.4f}  predictive R2={ss:+.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT,"cv_prediction.csv"),index=False)
    print("  READ-OFF: predictive R2>0 (out-of-sample) supports 'predictable'; if <=0, reword to 'systematically associated'.")
except Exception as e:
    print("  sklearn needed for CV:",e)

# ---------- C8: per-variant theta (only if CSV has multiple matchers) ----------
print("\n"+"="*84+"\n[C8] theta->EER per matcher (CSV variants)")
if "variant" in df.columns and df["variant"].nunique()>1:
    rows=[]
    for v,dv in df.groupby("variant"):
        dv=dv.copy()
        for c in ["enroll_emotion","test_emotion","enroll_session","test_session"]:
            if c in dv.columns: dv[c]=dv[c].astype(str)
        dv["emotion_pair"]=dv.get("enroll_emotion","")+"->"+dv.get("test_emotion","")
        dv["transition"]=dv.get("enroll_session","")+"->"+dv.get("test_session","")
        gv=seed_collapse(dv, reg_bands+[EE]); b,p,R2=ols_std(gv[reg_bands].to_numpy(float), gv[EE].to_numpy(float))
        ti=reg_bands.index("theta_power_drift")
        rows.append({"variant":v,"n":len(gv),"theta_beta":round(b[ti+1],3),"theta_p":float(p[ti+1]),"R2":round(R2,3)})
        print(f"  {v:22s} n={len(gv)} theta beta={b[ti+1]:+.3f} p={p[ti+1]:.3g} R2={R2:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT,"theta_per_variant.csv"),index=False)
    print("  READ-OFF: consistent positive significant theta across variants => theta generalises across matchers.")
else:
    print("  CSV has a single variant -> run step 13 (13_matcher_theta_consistency.py) to get PSD/EEGNet/ArcFace EERs.")
print("\nDONE ->", os.path.relpath(OUT,ROOT))
