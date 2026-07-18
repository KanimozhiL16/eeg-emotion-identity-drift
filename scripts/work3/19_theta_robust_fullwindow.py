#!/usr/bin/env python3
"""19: full-window (no subsample) theta recompute -> faithfulness + reproduce + C20."""
import os, glob, re, importlib.util, numpy as np, pandas as pd, numpy.linalg as la
from scipy.signal import welch
from scipy import stats
def _has(p): return os.path.isdir(os.path.join(p,"outputs"))
_hp=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
ROOT=next((c for c in [os.getcwd(),_hp,"/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)),os.getcwd())
OUT=os.path.join(ROOT,"outputs","work3","19_robust"); os.makedirs(OUT,exist_ok=True)
DATA=os.path.join(ROOT,"data","processed","sessionwise")
def sess_num(v):
    s=str(v); m=re.search(r'SESSION\s*_?(\d+)',s,re.I); return m.group(1) if m else (re.search(r'(\d+)',s).group(1) if re.search(r'(\d+)',s) else s)
_sp6=importlib.util.spec_from_file_location("p6",os.path.join(ROOT,"scripts","06_q1_levelup_biological_embedding_analysis.py"))
p6=importlib.util.module_from_spec(_sp6); _sp6.loader.exec_module(p6)
BANDS=p6.BANDS; BLIST=list(BANDS.keys())
m7=None
s7=os.path.join(ROOT,"scripts","07_seedv_session_drift_subject_adaptation.py")
if os.path.exists(s7):
    sp=importlib.util.spec_from_file_location("m7",s7); m7=importlib.util.module_from_spec(sp); sp.loader.exec_module(m7)
sess={}
for f in sorted(glob.glob(os.path.join(DATA,"*.npz"))):
    d=np.load(f,allow_pickle=True)
    if not all(k in d for k in ("X","y_subject","y_session","y_emotion")): continue
    sid=int(np.unique(d["y_session"])[0])
    if sid in sess: continue
    sess[sid]={"X":np.asarray(d["X"],np.float32),"sub":np.asarray(d["y_subject"],int),
               "emo":np.asarray(d["y_emotion"],int),"ch":[str(c) for c in d["ch_names"]],
               "fs":int(np.array(d["fs"]).item()) if "fs" in d.files else 200}
CH=sess[1]["ch"]; FS=sess[1]["fs"]; REGION=[p6.region_of_channel(c) for c in CH]; REGS=sorted(set(REGION))
EMOS=sorted(np.unique(sess[1]["emo"]).tolist()); EMONAME={0:"Disgust",1:"Fear",2:"Sad",3:"Neutral",4:"Happy"}
print(f"channels={len(CH)} fs={FS} regions={REGS}")
def chan_bandpower_full(X, zscore=False):
    Xw=X.astype(np.float64)
    if zscore: Xw=(Xw-Xw.mean(-1,keepdims=True))/(Xw.std(-1,keepdims=True)+1e-8)
    f,psd=welch(Xw,fs=FS,nperseg=min(256,Xw.shape[-1]),axis=-1)
    return {b: psd[:,:,(f>=lo)&(f<hi)].mean(axis=(0,2)) for b,(lo,hi) in BANDS.items()}
def region_bands(X, zscore=False):
    bp=chan_bandpower_full(X, zscore); out={}
    for b in BLIST:
        th=np.asarray(bp[b],float); out[b]={r:float(np.mean(th[[i for i,rr in enumerate(REGION) if rr==r]])) for r in REGS}
    return out
def global_drift(a,b): return {bd: float(np.mean([abs(a[bd][r]-b[bd][r]) for r in REGS])) for bd in BLIST}
cond={}; condZ={}
for s in sess:
    for e in EMOS:
        mask=sess[s]["emo"]==e
        if mask.sum()<5: continue
        cond[(s,e)]=region_bands(sess[s]["X"][mask], False); condZ[(s,e)]=region_bands(sess[s]["X"][mask], True)
rows=[]
for st in [2,3]:
    for ee in EMOS:
        for te in EMOS:
            if (1,ee) not in cond or (st,te) not in cond: continue
            gd=global_drift(cond[(1,ee)],cond[(st,te)]); gz=global_drift(condZ[(1,ee)],condZ[(st,te)])
            rows.append({"transition":f"1->{st}","emotion_pair":f"{EMONAME[ee]}->{EMONAME[te]}",
                         **{f"{b}_drift":gd[b] for b in BLIST}, "thetaZ_drift":gz["theta"]})
dd=pd.DataFrame(rows); dd.to_csv(os.path.join(OUT,"recomputed_fullwindow_drift.csv"),index=False)
csv=sorted(glob.glob(os.path.join(ROOT,"outputs","**","merged_global_psd_identity_eer.csv"),recursive=True),key=len)[0]
mg=pd.read_csv(csv)
if "variant" in mg.columns and (mg["variant"]=="arcface_supcon_cnn").any(): mg=mg[mg["variant"]=="arcface_supcon_cnn"]
for c in ["enroll_emotion","test_emotion"]:
    if c in mg.columns: mg[c]=mg[c].astype(str)
mg["emotion_pair"]=mg["enroll_emotion"]+"->"+mg["test_emotion"]; mg["transition"]="1->"+mg["test_session"].map(sess_num)
agg=mg.groupby(["transition","emotion_pair"]).agg(EER=("EER","mean"),theta_csv=("theta_power_drift","mean")).reset_index()
J=dd.merge(agg,on=["transition","emotion_pair"],how="inner").dropna()
print(f"\nmerged {len(J)} conditions (recomputed full-window vs CSV)")
r_faith=np.corrcoef(J["theta_drift"],J["theta_csv"])[0,1]; rs_faith=stats.spearmanr(J["theta_drift"],J["theta_csv"])[0]
print("="*84); print(f"(A) FAITHFULNESS: full-window theta vs CSV theta_power_drift: Pearson r={r_faith:+.3f} Spearman={rs_faith:+.3f}")
def zc(a): a=np.asarray(a,float); s=a.std(ddof=0); return (a-a.mean())/(s if s else 1)
Xb=np.c_[np.ones(len(J)),np.column_stack([zc(J[f"{b}_drift"]) for b in BLIST])]; y=zc(J["EER"])
b,*_=la.lstsq(Xb,y,rcond=None); resid=y-Xb@b; k=Xb.shape[1]
se=np.sqrt(np.diag(((resid@resid)/(len(J)-k))*la.inv(Xb.T@Xb))); pv=2*(1-stats.t.cdf(np.abs(b/se),len(J)-k))
R2=1-(resid@resid)/np.sum((y-y.mean())**2)
print(f"\n(B) 4-band standardized regression (n={len(J)}, R2={R2:.3f}) -- expect theta~+0.43:")
for i,bd in enumerate(BLIST): print(f"    {bd:6s} beta={b[i+1]:+.3f} p={pv[i+1]:.3g}")
for tag,col in [("amplitude","theta_drift"),("z-scored","thetaZ_drift")]:
    rho,pr=stats.spearmanr(J[col],J["EER"]); print(f"(C) single theta [{tag:9s}] vs EER: Spearman={rho:+.3f} p={pr:.3g}")
print("\n"+"="*84+"\n(D) [C20] per-subject full-window theta vs per-subject EER")
subj={}; SUBS=sorted(np.unique(sess[1]["sub"]).tolist())
for s in SUBS:
    rb={}
    for ss in [1,2,3]:
        mask=sess[ss]["sub"]==s
        if mask.sum()<5: continue
        rb[ss]=region_bands(sess[ss]["X"][mask], False)
    if all(k in rb for k in (1,2,3)): subj[s]=0.5*(global_drift(rb[1],rb[2])["theta"]+global_drift(rb[1],rb[3])["theta"])
eer_sub={}
if m7 is not None:
    def feats(ss):
        X,y=sess[ss]["X"],sess[ss]["sub"]
        try: X,y=m7.downsample_per_subject(X,y,max_per_subject=900,seed=42)[:2]
        except Exception: pass
        return m7.simple_features(X), y
    Fe,ye=feats(1); P,ow=m7.make_prototypes(Fe,ye); ow=np.asarray(ow)
    def pe(ss):
        Fp,yp=feats(ss); sdf=m7.verification_scores(Fp,yp,P,ow)
        return {int(c):m7.eer_auc(g["y_true"].values,g["score"].values)[0] for c,g in sdf.groupby("probe_subject")}
    e2,e3=pe(2),pe(3)
    for s in SUBS:
        if s in e2 and s in e3: eer_sub[s]=0.5*(e2[s]+e3[s])
sr=pd.DataFrame([{"subject":s,"theta_drift":subj[s],"cross_EER":eer_sub[s]} for s in SUBS if s in subj and s in eer_sub])
sr.to_csv(os.path.join(OUT,"subject_theta_fullwindow.csv"),index=False)
if len(sr)>4:
    x=sr["theta_drift"].to_numpy(float); yy=sr["cross_EER"].to_numpy(float)
    rho,pr=stats.spearmanr(x,yy); loso=[stats.spearmanr(np.delete(x,i),np.delete(yy,i))[0] for i in range(len(x))]
    print(f"    n={len(sr)} Spearman={rho:+.3f} p={pr:.3g} | LOSO [{min(loso):+.3f},{max(loso):+.3f}]")
print("DONE ->",os.path.relpath(OUT,ROOT))
