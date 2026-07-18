#!/usr/bin/env python3
"""
13_matcher_theta_consistency.py -- C8.
Does theta drift predict the per-condition EER of MULTIPLE matchers, not just ArcFace/SupCon?
For each matcher we compute per-condition (transition x emotion-pair) EER and regress it on the
per-condition theta drift; consistent positive significant theta across matchers => theta generalises.
Matchers: PSD (5-band log-power), Lightweight (run_07 simple_features if importable), ArcFace (from CSV).
Writes: outputs/work3/13_matcher_theta/{matcher_theta.csv, matcher_theta_summary.csv}
"""
import os, glob, re, importlib.util, numpy as np, pandas as pd, numpy.linalg as la
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
OUT=os.path.join(ROOT,"outputs","work3","13_matcher_theta"); os.makedirs(OUT,exist_ok=True)
FS=200; EMO_KEYS=["y_emotion","y_label","emotion","y_emo","labels"]
EMO={0:"Disgust",1:"Fear",2:"Sad",3:"Neutral",4:"Happy"}  # SEED-V int->CSV name (paper mapping)
BANDS=[("delta",0.5,4),("theta",4,8),("alpha",8,13),("beta",13,30),("gamma",30,45)]

LW=None
SRC=os.path.join(ROOT,"scripts","07_seedv_session_drift_subject_adaptation.py")
if os.path.exists(SRC):
    try:
        spec=importlib.util.spec_from_file_location("run07",SRC); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); LW=m
    except Exception as e: print("run_07 import failed (lightweight skipped):",e)

sess={}; emo_key=None
for f in sorted(glob.glob(os.path.join(DATA,"*.npz"))):
    d=np.load(f,allow_pickle=True)
    if not all(k in d for k in ("X","y_subject","y_session")): continue
    if emo_key is None: emo_key=next((k for k in EMO_KEYS if k in d.files),None)
    sid=int(np.unique(d["y_session"])[0])
    if sid in sess: continue
    sess[sid]={"X":np.asarray(d["X"],np.float32),"sub":np.asarray(d["y_subject"],int),
               "emo":(np.asarray(d[emo_key],int) if emo_key else None)}
if emo_key is None: raise SystemExit("no emotion key in npz (looked for %s)"%EMO_KEYS)
SUBS=sorted(np.unique(sess[1]["sub"]).tolist())

def psd_feat(X):
    f,P=welch(X.astype(np.float64),fs=FS,nperseg=min(200,X.shape[-1]),axis=-1); out=[]
    for _,lo,hi in BANDS:
        mask=(f>=lo)&(f<hi); out.append(np.log(P[:,:,mask].sum(-1)+1e-12))
    return np.concatenate(out,-1)
def theta_feat(X):
    f,P=welch(X.astype(np.float64),fs=FS,nperseg=min(200,X.shape[-1]),axis=-1); mask=(f>=4)&(f<8)
    return P[:,:,mask].sum(-1)

FEAT={}
for s in sess:
    X=sess[s]["X"]; FEAT[s]={"PSD":psd_feat(X)}
    if LW is not None:
        try: FEAT[s]["Lightweight"]=LW.simple_features(X)
        except Exception: pass
    FEAT[s]["theta"]=theta_feat(X)

def eer(gen,imp):
    from sklearn.metrics import roc_curve
    y=np.r_[np.ones(len(gen)),np.zeros(len(imp))]; s=np.r_[gen,imp]
    fpr,tpr,_=roc_curve(y,s); fnr=1-tpr; i=np.nanargmin(np.abs(fpr-fnr)); return float((fpr[i]+fnr[i])/2)
def cond_eer(matcher, s_te, e_en, e_te):
    en=sess[1]; te=sess[s_te]; Fen=FEAT[1][matcher]; Fte=FEAT[s_te][matcher]; protos={}
    for sub in SUBS:
        mask=(en["sub"]==sub)&(en["emo"]==e_en)
        if mask.sum()<3: return None
        v=Fen[mask].mean(0); protos[sub]=v/(np.linalg.norm(v)+1e-9)
    gen=[]; imp=[]
    for sub in SUBS:
        mask=(te["sub"]==sub)&(te["emo"]==e_te)
        if mask.sum()<1: continue
        q=Fte[mask]; q=q/(np.linalg.norm(q,axis=1,keepdims=True)+1e-9)
        for other in SUBS:
            sc=q@protos[other]; (gen if other==sub else imp).extend(sc.tolist())
    if len(gen)<5 or len(imp)<5: return None
    return eer(np.array(gen),np.array(imp))

EMOS=sorted(np.unique(sess[1]["emo"]).tolist()); rows=[]
for s_te in [2,3]:
    for e_en in EMOS:
        for e_te in EMOS:
            dth=np.abs(FEAT[s_te]["theta"][sess[s_te]["emo"]==e_te].mean(0)
                      -FEAT[1]["theta"][sess[1]["emo"]==e_en].mean(0)).mean()
            row={"transition":f"1->{s_te}","emotion_pair":f"{EMO[int(e_en)]}->{EMO[int(e_te)]}","theta_drift":float(dth)}
            for matcher in [k for k in ["PSD","Lightweight"] if k in FEAT[1]]:
                row["EER_"+matcher]=cond_eer(matcher,s_te,e_en,e_te)
            rows.append(row)
df=pd.DataFrame(rows)

cands=glob.glob(os.path.join(ROOT,"outputs","**","merged_global_psd_identity_eer.csv"),recursive=True)
if cands:
    mc=pd.read_csv(sorted(cands,key=len)[0])
    if "variant" in mc.columns and (mc["variant"]=="arcface_supcon_cnn").any(): mc=mc[mc["variant"]=="arcface_supcon_cnn"]
    for c in ["enroll_emotion","test_emotion"]:
        if c in mc.columns: mc[c]=mc[c].astype(str)
    mc["emotion_pair"]=mc.get("enroll_emotion","")+"->"+mc.get("test_emotion","")
    mc["transition"]="1->"+mc["test_session"].map(sess_num)
    ar=mc.groupby(["transition","emotion_pair"])["EER"].mean().reset_index().rename(columns={"EER":"EER_ArcFace"})
    df=df.merge(ar,on=["transition","emotion_pair"],how="left")
    print(f"    [merge] ArcFace matched {int(df['EER_ArcFace'].notna().sum())}/{len(df)} conditions (expect ~50)")
df.to_csv(os.path.join(OUT,"matcher_theta.csv"),index=False)

def z(a): a=np.asarray(a,float); s=a.std(ddof=0); return (a-a.mean())/(s if s else 1.0)
print("="*84+"\n[C8] theta drift -> per-condition EER, per matcher")
res=[]
for col in [c for c in df.columns if c.startswith("EER_")]:
    g=df[["theta_drift",col]].dropna()
    if len(g)<8: print(f"  {col:16s} too few"); continue
    b=np.polyfit(z(g["theta_drift"]),z(g[col]),1); rho,pr=stats.spearmanr(g["theta_drift"],g[col])
    print(f"  {col:16s} n={len(g)}  std-beta={b[0]:+.3f}  Spearman rho={rho:+.3f} p={pr:.3g}")
    res.append({"matcher":col.replace('EER_',''),"n":len(g),"std_beta":round(b[0],3),"spearman_rho":round(rho,3),"spearman_p":float(pr)})
pd.DataFrame(res).to_csv(os.path.join(OUT,"matcher_theta_summary.csv"),index=False)
print("  READ-OFF: theta positive+significant across PSD/Lightweight/ArcFace => theta generalises across matchers (answers C8).")
print("DONE ->",os.path.relpath(OUT,ROOT))
