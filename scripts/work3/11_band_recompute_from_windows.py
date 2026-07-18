#!/usr/bin/env python3
"""
11_band_recompute_from_windows.py -- C12 (+C9 delta, +C10).
KEY POINT (C12): the paper computes PSD AFTER per-window channel-wise z-score, which forces unit
variance and turns absolute band power into RELATIVE spectral redistribution. Here we recompute
per-condition 5-band spectral drift on AMPLITUDE-PRESERVING windows (filtered, NOT z-scored) and,
for comparison, on z-scored windows, then pair each with the EXISTING per-condition EER (from the
merged CSV, joined on transition x emotion-pair) and refit the standardized band->EER regression.
Reads : data/processed/sessionwise/*.npz  (keys: X, y_subject, y_session, y_emotion)
        outputs/**/merged_global_psd_identity_eer.csv
Writes: outputs/work3/11_band_recompute/{per_condition_drift.csv, regression_compare.csv}
"""
import os, glob, re, numpy as np, pandas as pd, numpy.linalg as la
from scipy.signal import welch
from scipy import stats

def sess_num(v):
    s=str(v); m=re.search(r'SESSION\s*_?(\d+)', s, re.I)
    if m: return m.group(1)
    m=re.search(r'(\d+)', s); return m.group(1) if m else s

def _has(p): return os.path.isdir(os.path.join(p,"outputs"))
_hp=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
ROOT=next((c for c in [os.getcwd(),_hp,"/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)),os.getcwd())
DATA=os.path.join(ROOT,"data","processed","sessionwise")
OUT=os.path.join(ROOT,"outputs","work3","11_band_recompute"); os.makedirs(OUT,exist_ok=True)
FS=200; BANDS=[("delta",0.5,4),("theta",4,8),("alpha",8,13),("beta",13,30),("gamma",30,45)]
EMO={0:"Disgust",1:"Fear",2:"Sad",3:"Neutral",4:"Happy"}  # SEED-V int->CSV name (paper mapping)
EMO_KEYS=["y_emotion","y_label","emotion","y_emo","labels"]

def bandpow(X, zscore):
    Xp=X.astype(np.float64)
    if zscore:
        m=Xp.mean(-1,keepdims=True); s=Xp.std(-1,keepdims=True); Xp=(Xp-m)/(s+1e-8)
    f,P=welch(Xp,fs=FS,nperseg=min(200,X.shape[-1]),axis=-1)
    out=[]
    for _,lo,hi in BANDS:
        mask=(f>=lo)&(f<hi); out.append(P[:,:,mask].sum(-1))
    return np.stack(out,-1)

sess={}; emo_key=None
for fpath in sorted(glob.glob(os.path.join(DATA,"*.npz"))):
    d=np.load(fpath,allow_pickle=True)
    if not all(k in d for k in ("X","y_subject","y_session")): continue
    if emo_key is None: emo_key=next((k for k in EMO_KEYS if k in d.files),None)
    sid=int(np.unique(d["y_session"])[0])
    if sid in sess: continue
    rec={"X":np.asarray(d["X"],np.float32),"sub":np.asarray(d["y_subject"],int)}
    rec["emo"]=np.asarray(d[emo_key],int) if emo_key else None
    sess[sid]=rec
if emo_key is None:
    raise SystemExit("ERROR: no emotion label key found in sessionwise npz (looked for %s)."%EMO_KEYS)
print("sessions:",sorted(sess),"| emotion key:",emo_key)

def cond_bandpow(zscore):
    tbl={}
    for s,rec in sess.items():
        B=bandpow(rec["X"],zscore)
        for e in np.unique(rec["emo"]):
            m=rec["emo"]==e; tbl[(s,int(e))]=B[m].mean(0)
    return tbl

def build_drift(zscore):
    tbl=cond_bandpow(zscore); rows=[]
    for se in [2,3]:
        for e_en in sorted({k[1] for k in tbl if k[0]==1}):
            for e_te in sorted({k[1] for k in tbl if k[0]==se}):
                if (1,e_en) not in tbl or (se,e_te) not in tbl: continue
                drift=np.abs(tbl[(se,e_te)]-tbl[(1,e_en)]).mean(0)
                rows.append({"transition":f"1->{se}","emotion_pair":f"{EMO[int(e_en)]}->{EMO[int(e_te)]}",
                             **{BANDS[i][0]+"_drift":float(drift[i]) for i in range(5)}})
    return pd.DataFrame(rows)

dz=build_drift(True); dz["preproc"]="zscored"
da=build_drift(False); da["preproc"]="amplitude"
pd.concat([dz,da],ignore_index=True).to_csv(os.path.join(OUT,"per_condition_drift.csv"),index=False)
print("per-condition drift rows:",len(dz),"per preprocessing")

cands=glob.glob(os.path.join(ROOT,"outputs","**","merged_global_psd_identity_eer.csv"),recursive=True)
if not cands: raise SystemExit("merged CSV not found for EER join")
m=pd.read_csv(sorted(cands,key=len)[0])
if "variant" in m.columns and m["variant"].nunique()>1:
    m=m[m["variant"]=="arcface_supcon_cnn"] if (m["variant"]=="arcface_supcon_cnn").any() else m
for c in ["enroll_emotion","test_emotion"]:
    if c in m.columns: m[c]=m[c].astype(str)
m["emotion_pair"]=m.get("enroll_emotion","")+"->"+m.get("test_emotion","")
m["transition"]=("1->"+m["test_session"].map(sess_num)) if "test_session" in m.columns else ""
eer=m.groupby(["transition","emotion_pair"])["EER"].mean().reset_index()
if "theta_power_drift" in m.columns:
    ct=m.groupby(["transition","emotion_pair"])["theta_power_drift"].mean().reset_index()
    chk=dz.merge(ct,on=["transition","emotion_pair"],how="inner")
    if len(chk)>3:
        r=np.corrcoef(chk["theta_drift"],chk["theta_power_drift"])[0,1]
        print(f"    [sanity] recomputed z-scored theta vs CSV theta_power_drift: r={r:+.3f} over {len(chk)} conditions (expect high -> validates emotion mapping + recompute)")

def z(a): a=np.asarray(a,float); s=a.std(ddof=0); return (a-a.mean())/(s if s else 1.0)
def refit(dfp):
    g=dfp.merge(eer,on=["transition","emotion_pair"],how="inner")
    print(f"    [merge] {len(dfp)} drift rows x eer -> matched {len(g)} conditions (expect ~50)")
    bands=["theta_drift","alpha_drift","beta_drift","gamma_drift"]
    if len(g)<8: return None
    n=len(g); X=np.c_[np.ones(n),np.column_stack([z(g[b]) for b in bands])]; y=z(g["EER"].to_numpy(float))
    b,*_=la.lstsq(X,y,rcond=None); r=y-X@b; k=X.shape[1]
    se=np.sqrt(np.diag(((r@r)/(n-k))*la.inv(X.T@X))); p=2*(1-stats.t.cdf(np.abs(b/se),n-k))
    R2=1-(r@r)/np.sum((y-y.mean())**2)
    return n,{bands[i]:(round(b[i+1],3),float(p[i+1])) for i in range(4)},round(R2,3)

print("\n"+"="*84+"\n[C12] theta regression: z-scored vs amplitude-preserving PSD")
res=[]
for tag,dfp in [("zscored (paper)",dz),("amplitude-preserving",da)]:
    out=refit(dfp)
    if out is None: print(f"  {tag}: too few matched conditions"); continue
    n,coef,R2=out; th=coef["theta_drift"]
    print(f"  {tag:22s} n={n} R2={R2}  theta beta={th[0]:+.3f} p={th[1]:.3g} | "+
          " ".join(f"{b.split('_')[0]}={coef[b][0]:+.2f}" for b in coef))
    res.append({"preproc":tag,"n":n,"R2":R2,**{b:coef[b][0] for b in coef},
                **{b+"_p":coef[b][1] for b in coef}})
pd.DataFrame(res).to_csv(os.path.join(OUT,"regression_compare.csv"),index=False)
print("\n  READ-OFF (C12): if theta stays the top POSITIVE band with p<0.05 on amplitude-preserving PSD,")
print("  keep 'band-power drift'. If it weakens/flips, rename to 'normalised spectral redistribution'.")
print("DONE ->",os.path.relpath(OUT,ROOT))
