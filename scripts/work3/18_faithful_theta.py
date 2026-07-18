#!/usr/bin/env python3
"""
18_faithful_theta.py -- FAITHFUL redo of C12/C20/C8 using pipeline 06's OWN functions.
Root cause of earlier divergence: step 11/12/13 took abs() at CHANNEL level (mean|dP|); the pipeline
(06) takes abs() at REGION level (|region-mean_enroll - region-mean_test|) then averages 5 regions.
This script imports 06 (region_of_channel, compute_channel_bandpower, BANDS) so the theta statistic
is identical to the manuscript, and:
  [C12] region-based theta drift regression vs ArcFace EER, AMPLITUDE-PRESERVING (as stored) vs
        a z-scored-window variant -> amplitude branch should reproduce CSV theta beta ~ +0.43 (self-check),
        the z-scored branch shows what per-window z-scoring would have done.
  [C20] per-subject region-based theta drift vs per-subject cross-session EER (run_07 scoring).
  [C8]  pipeline theta_power_drift (from the merged CSV) vs per-matcher EER (from step 13 output).
Run: python -u scripts/work3/18_faithful_theta.py 2>&1 | tee outputs/work3/logs/18_faithful.log
"""
import os, glob, re, importlib.util, numpy as np, pandas as pd, numpy.linalg as la
from scipy import stats

def _has(p): return os.path.isdir(os.path.join(p,"outputs"))
_hp=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
ROOT=next((c for c in [os.getcwd(),_hp,"/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)),os.getcwd())
OUT=os.path.join(ROOT,"outputs","work3","18_faithful_theta"); os.makedirs(OUT,exist_ok=True)
DATA=os.path.join(ROOT,"data","processed","sessionwise")
def sess_num(v):
    s=str(v); m=re.search(r'SESSION\s*_?(\d+)',s,re.I); return m.group(1) if m else (re.search(r'(\d+)',s).group(1) if re.search(r'(\d+)',s) else s)

# ---- import pipeline 06 (functions only) ----
p6path=os.path.join(ROOT,"scripts","06_q1_levelup_biological_embedding_analysis.py")
spec=importlib.util.spec_from_file_location("p6",p6path); p6=importlib.util.module_from_spec(spec); spec.loader.exec_module(p6)
BANDS=p6.BANDS; BLIST=list(BANDS.keys())
print("06 BANDS:",BANDS,"| REGION_ORDER:",getattr(p6,"REGION_ORDER",None))

# ---- import run_07 for per-subject EER ----
m7=None
s7=os.path.join(ROOT,"scripts","07_seedv_session_drift_subject_adaptation.py")
if os.path.exists(s7):
    sp=importlib.util.spec_from_file_location("m7",s7); m7=importlib.util.module_from_spec(sp); sp.loader.exec_module(m7)

# ---- load sessionwise windows ----
sess={}
for f in sorted(glob.glob(os.path.join(DATA,"*.npz"))):
    d=np.load(f,allow_pickle=True)
    if not all(k in d for k in ("X","y_subject","y_session","y_emotion")): continue
    sid=int(np.unique(d["y_session"])[0])
    if sid in sess: continue
    sess[sid]={"X":np.asarray(d["X"],np.float32),"sub":np.asarray(d["y_subject"],int),
               "emo":np.asarray(d["y_emotion"],int),"ch":[str(c) for c in d["ch_names"]],
               "fs":int(np.array(d["fs"]).item()) if "fs" in d.files else 200}
CH=sess[1]["ch"]; FS=sess[1]["fs"]
REGION=[p6.region_of_channel(c) for c in CH]
REGS=sorted(set(REGION))
print(f"channels={len(CH)} fs={FS} | regions found: {REGS}")
# amplitude check (C12 evidence)
_x=sess[1]["X"][:300].astype(float); print(f"[C12 evidence] stored window per-ch std avg={_x.std(-1).mean():.3f} (1.0 => unit z-scored; <1 => amplitude-preserving)")

def region_theta(Xwin, zscore=False):
    """region-averaged theta power vector (dict region->value), pipeline-faithful."""
    Xw=Xwin.astype(np.float64)
    if zscore:
        Xw=(Xw-Xw.mean(-1,keepdims=True))/(Xw.std(-1,keepdims=True)+1e-8)
    bp=p6.compute_channel_bandpower(Xw, FS)     # dict band-> (C,)
    th=np.asarray(bp["theta"],float)            # per-channel theta power
    out={}
    for r in REGS:
        idx=[i for i,rr in enumerate(REGION) if rr==r]
        if idx: out[r]=float(np.mean(th[idx]))
    return out
def drift_over_regions(a,b):
    rs=set(a)&set(b); return float(np.mean([abs(a[r]-b[r]) for r in rs])) if rs else np.nan

# ============ C12: amplitude vs z-scored, per-condition, vs ArcFace EER ============
print("\n"+"="*84+"\n[C12] pipeline-faithful theta drift vs ArcFace EER: amplitude vs z-scored")
csv=sorted(glob.glob(os.path.join(ROOT,"outputs","**","merged_global_psd_identity_eer.csv"),recursive=True),key=len)[0]
mg=pd.read_csv(csv)
if "variant" in mg.columns and (mg["variant"]=="arcface_supcon_cnn").any(): mg=mg[mg["variant"]=="arcface_supcon_cnn"]
for c in ["enroll_emotion","test_emotion"]:
    if c in mg.columns: mg[c]=mg[c].astype(str)
mg["emotion_pair"]=mg["enroll_emotion"]+"->"+mg["test_emotion"]
mg["transition"]="1->"+mg["test_session"].map(sess_num)
eer=mg.groupby(["transition","emotion_pair"])["EER"].mean().reset_index()
EMOS=sorted(np.unique(sess[1]["emo"]).tolist())
EMONAME={0:"Disgust",1:"Fear",2:"Sad",3:"Neutral",4:"Happy"}
def per_condition_theta(zscore):
    # precompute region theta per (session,emotion)
    tbl={}
    for s in sess:
        for e in EMOS:
            mask=sess[s]["emo"]==e
            if mask.sum()<5: continue
            tbl[(s,e)]=region_theta(sess[s]["X"][mask], zscore=zscore)
    rows=[]
    for st in [2,3]:
        for ee in EMOS:
            for te in EMOS:
                if (1,ee) not in tbl or (st,te) not in tbl: continue
                rows.append({"transition":f"1->{st}","emotion_pair":f"{EMONAME[ee]}->{EMONAME[te]}",
                             "theta_drift":drift_over_regions(tbl[(1,ee)],tbl[(st,te)])})
    return pd.DataFrame(rows)
for tag,zs in [("amplitude (paper)",False),("z-scored windows",True)]:
    d=per_condition_theta(zs).merge(eer,on=["transition","emotion_pair"],how="inner").dropna()
    if len(d)<8: print(f"  {tag}: too few ({len(d)})"); continue
    xz=(d["theta_drift"]-d["theta_drift"].mean())/d["theta_drift"].std(ddof=0)
    yz=(d["EER"]-d["EER"].mean())/d["EER"].std(ddof=0)
    b=np.polyfit(xz,yz,1)[0]; rho,pr=stats.spearmanr(d["theta_drift"],d["EER"])
    print(f"  {tag:20s} n={len(d)}  theta std-beta={b:+.3f}  Spearman rho={rho:+.3f} p={pr:.3g}")
print("  SELF-CHECK: 'amplitude (paper)' theta std-beta should be ~ +0.43 (matches CSV Table 3).")
print("  READ-OFF (C12): if z-scored theta beta is similar, per-window z-scoring would not change the story;")
print("  and since the paper already uses amplitude-preserving power, C12 is a Methods clarification.")

# ============ C20: per-subject region theta drift vs per-subject EER ============
print("\n"+"="*84+"\n[C20] per-subject pipeline-faithful theta drift vs per-subject cross-session EER")
subj_theta={}
SUBS=sorted(np.unique(sess[1]["sub"]).tolist())
for s in SUBS:
    rt={}
    for ss in [1,2,3]:
        mask=sess[ss]["sub"]==s
        if mask.sum()<5: continue
        rt[ss]=region_theta(sess[ss]["X"][mask], zscore=False)
    if all(k in rt for k in (1,2,3)):
        subj_theta[s]=0.5*(drift_over_regions(rt[1],rt[2])+drift_over_regions(rt[1],rt[3]))
# per-subject EER via run_07 (same as manuscript subject_robustness)
eer_sub={}
if m7 is not None:
    def feats(ss):
        X,y=sess[ss]["X"],sess[ss]["sub"]
        try: X,y=m7.downsample_per_subject(X,y,max_per_subject=900,seed=42)[:2]
        except Exception: pass
        return m7.simple_features(X), y
    Fe,ye=feats(1); P,owners=m7.make_prototypes(Fe,ye); owners=np.asarray(owners)
    def pe(ss):
        Fp,yp=feats(ss); sdf=m7.verification_scores(Fp,yp,P,owners)
        return {int(c):m7.eer_auc(g["y_true"].values,g["score"].values)[0] for c,g in sdf.groupby("probe_subject")}
    e2,e3=pe(2),pe(3)
    for s in SUBS:
        if s in e2 and s in e3: eer_sub[s]=0.5*(e2[s]+e3[s])
rows=[{"subject":s,"theta_drift":subj_theta[s],"cross_EER":eer_sub[s]} for s in SUBS if s in subj_theta and s in eer_sub]
df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,"subject_theta_faithful.csv"),index=False)
if len(df)>4:
    x=df["theta_drift"].to_numpy(float); y=df["cross_EER"].to_numpy(float)
    rho,pr=stats.spearmanr(x,y); tau,pt=stats.kendalltau(x,y)
    loso=[stats.spearmanr(np.delete(x,i),np.delete(y,i))[0] for i in range(len(x))]
    print(f"  n={len(df)}  Spearman rho={rho:+.3f} p={pr:.3g} | Kendall tau={tau:+.3f} p={pt:.3g}")
    print(f"  LOSO Spearman range [{min(loso):+.3f},{max(loso):+.3f}]")
    print("  READ-OFF: POSITIVE significant => theta drift tracks identity EER at subject level (supports paper).")
    print("            NEGATIVE/again null => the theta effect is between-condition only, NOT a per-subject identity signal.")
else:
    print("  too few subjects with both theta and EER")

# ============ C8: pipeline theta vs per-matcher EER ============
print("\n"+"="*84+"\n[C8] pipeline theta_power_drift vs per-matcher EER")
mtp=os.path.join(ROOT,"outputs","work3","13_matcher_theta","matcher_theta.csv")
if os.path.exists(mtp) and "theta_power_drift" in mg.columns:
    mt=pd.read_csv(mtp)
    ct=mg.groupby(["transition","emotion_pair"])["theta_power_drift"].mean().reset_index()
    j=mt.merge(ct,on=["transition","emotion_pair"],how="inner")
    print(f"  merged {len(j)} conditions with pipeline theta_power_drift")
    for col in [c for c in j.columns if c.startswith("EER_")]:
        g=j[["theta_power_drift",col]].dropna()
        if len(g)<8: print(f"    {col}: too few"); continue
        rho,pr=stats.spearmanr(g["theta_power_drift"],g[col])
        print(f"    {col:16s} n={len(g)}  Spearman(theta_pipeline, EER)={rho:+.3f} p={pr:.3g}")
    print("  READ-OFF: consistent positive across PSD/Lightweight/ArcFace => pipeline theta generalises across matchers.")
else:
    print("  need outputs/work3/13_matcher_theta/matcher_theta.csv (run step 13 first) and theta_power_drift in CSV.")
print("\nDONE ->",os.path.relpath(OUT,ROOT))
