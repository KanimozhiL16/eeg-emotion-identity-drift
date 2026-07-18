import os,glob,re,importlib.util,numpy as np,pandas as pd,numpy.linalg as la
from scipy.signal import welch
from scipy import stats
from sklearn.metrics import roc_curve
def _has(p): return os.path.isdir(os.path.join(p,"outputs"))
_hp=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
ROOT=next((c for c in [os.getcwd(),_hp,"/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC","/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)),os.getcwd())
OUT=os.path.join(ROOT,"outputs","work3","21_swapfixed");os.makedirs(OUT,exist_ok=True)
DATA=os.path.join(ROOT,"data","processed","sessionwise")
def sn(v):
    s=str(v);m=re.search(r'SESSION\s*_?(\d+)',s,re.I);return m.group(1) if m else (re.search(r'(\d+)',s).group(1) if re.search(r'(\d+)',s) else s)
_sp=importlib.util.spec_from_file_location("p6",os.path.join(ROOT,"scripts","06_q1_levelup_biological_embedding_analysis.py"));p6=importlib.util.module_from_spec(_sp);_sp.loader.exec_module(p6)
BANDS=p6.BANDS;BL=list(BANDS.keys())
LW=None
s7=os.path.join(ROOT,"scripts","07_seedv_session_drift_subject_adaptation.py")
if os.path.exists(s7):
    sp=importlib.util.spec_from_file_location("m7",s7);LW=importlib.util.module_from_spec(sp);sp.loader.exec_module(LW)
sess={}
for f in sorted(glob.glob(os.path.join(DATA,"*.npz"))):
    d=np.load(f,allow_pickle=True)
    if not all(k in d for k in ("X","y_subject","y_session","y_emotion")):continue
    sid=int(np.unique(d["y_session"])[0])
    if sid in sess:continue
    sess[sid]={"X":np.asarray(d["X"],np.float32),"sub":np.asarray(d["y_subject"],int),"emo":np.asarray(d["y_emotion"],int),"ch":[str(c) for c in d["ch_names"]],"fs":int(np.array(d["fs"]).item()) if "fs" in d.files else 200}
CH=sess[1]["ch"];FS=sess[1]["fs"];REGION=[p6.region_of_channel(c) for c in CH];REGS=sorted(set(REGION))
EMOS=sorted(np.unique(sess[1]["emo"]).tolist());M={0:"Disgust",1:"Fear",2:"Sad",3:"Neutral",4:"Happy"}
def regbands(X,z=False):
    Xw=X.astype(np.float64)
    if z:Xw=(Xw-Xw.mean(-1,keepdims=True))/(Xw.std(-1,keepdims=True)+1e-8)
    f,psd=welch(Xw,fs=FS,nperseg=min(256,Xw.shape[-1]),axis=-1)
    o={}
    for b,(lo,hi) in BANDS.items():
        th=psd[:,:,(f>=lo)&(f<hi)].mean(axis=(0,2));o[b]={r:float(np.mean(th[[i for i,rr in enumerate(REGION) if rr==r]])) for r in REGS}
    return o
def gdrift(a,b):return {bd:float(np.mean([abs(a[bd][r]-b[bd][r]) for r in REGS])) for bd in BL}
def psd_feat(X):
    f,P=welch(X.astype(np.float64),fs=FS,nperseg=min(256,X.shape[-1]),axis=-1);out=[]
    for lo,hi in BANDS.values():out.append(np.log(P[:,:,(f>=lo)&(f<hi)].sum(-1)+1e-12))
    return np.concatenate(out,-1)
FEAT={}
for s in sess:
    X=sess[s]["X"];FEAT[s]={"PSD":psd_feat(X)}
    if LW is not None:
        try:FEAT[s]["Lightweight"]=LW.simple_features(X)
        except Exception:pass
def eer(g,i):
    y=np.r_[np.ones(len(g)),np.zeros(len(i))];s=np.r_[g,i];fpr,tpr,_=roc_curve(y,s);fnr=1-tpr;k=np.nanargmin(np.abs(fpr-fnr));return float((fpr[k]+fnr[k])/2)
def cond_eer(mat,st,ee,te):
    en,te_=sess[1],sess[st];Fe,Ft=FEAT[1][mat],FEAT[st][mat];pr={}
    for su in np.unique(en["sub"]):
        m=(en["sub"]==su)&(en["emo"]==ee)
        if m.sum()<3:return None
        v=Fe[m].mean(0);pr[su]=v/(np.linalg.norm(v)+1e-9)
    G,I=[],[]
    for su in np.unique(te_["sub"]):
        m=(te_["sub"]==su)&(te_["emo"]==te)
        if m.sum()<1:continue
        q=Ft[m];q=q/(np.linalg.norm(q,axis=1,keepdims=True)+1e-9)
        for o in pr:
            sc=(q@pr[o]).tolist();(G if o==su else I).extend(sc)
    return eer(np.array(G),np.array(I)) if len(G)>4 and len(I)>4 else None
rt={}
for s in sess:
    for e in EMOS:
        m=sess[s]["emo"]==e
        if m.sum()>=5:rt[(s,e)]=regbands(sess[s]["X"][m],False)
rtz={}
for s in sess:
    for e in EMOS:
        m=sess[s]["emo"]==e
        if m.sum()>=5:rtz[(s,e)]=regbands(sess[s]["X"][m],True)
rows=[]
for st in [2,3]:
    for ee in EMOS:
        for te in EMOS:
            if (1,ee) not in rt or (st,te) not in rt:continue
            d=gdrift(rt[(1,ee)],rt[(st,te)]);dz=gdrift(rtz[(1,ee)],rtz[(st,te)])
            row={"transition":f"1->{st}","csv_enroll":M[te],"csv_test":M[ee],  # SWAP-CORRECTED merge keys
                 **{f"{b}_drift":d[b] for b in BL},"thetaZ":dz["theta"]}
            for mat in [k for k in ["PSD","Lightweight"] if k in FEAT[1]]:
                row["EER_"+mat]=cond_eer(mat,st,ee,te)
            rows.append(row)
dd=pd.DataFrame(rows)
mg=pd.read_csv(sorted(glob.glob(os.path.join(ROOT,"outputs","**","merged_global_psd_identity_eer.csv"),recursive=True),key=len)[0])
if "variant" in mg.columns and (mg["variant"]=="arcface_supcon_cnn").any():mg=mg[mg["variant"]=="arcface_supcon_cnn"]
mg["transition"]="1->"+mg["test_session"].map(sn)
ag=mg.groupby(["transition","enroll_emotion","test_emotion"]).agg(EER_ArcFace=("EER","mean"),theta_csv=("theta_power_drift","mean")).reset_index()
J=dd.merge(ag,left_on=["transition","csv_enroll","csv_test"],right_on=["transition","enroll_emotion","test_emotion"],how="inner")
print("merged",len(J),"conditions")
r=np.corrcoef(J["theta_drift"],J["theta_csv"])[0,1];print(f"faithfulness (swap-fixed) r={r:+.3f}")
def zc(a):a=np.asarray(a,float);s=a.std(ddof=0);return (a-a.mean())/(s if s else 1)
print("\n[C12] 4-band regression (my full-window drift) vs ArcFace EER:")
for tag,cols in [("amplitude",[f"{b}_drift" for b in BL]),("z-scored",["thetaZ"]+[f"{b}_drift" for b in BL[1:]])]:
    X=np.c_[np.ones(len(J)),np.column_stack([zc(J[c]) for c in cols])];y=zc(J["EER_ArcFace"])
    b,*_=la.lstsq(X,y,rcond=None);print(f"  {tag:10s} theta beta={b[1]:+.3f}")
print("\n[C8] pipeline theta_csv vs per-matcher EER (swap-fixed):")
for col in ["EER_PSD","EER_Lightweight","EER_ArcFace"]:
    if col in J.columns:
        g=J[["theta_csv",col]].dropna();rho,pp=stats.spearmanr(g["theta_csv"],g[col]);print(f"  {col:16s} n={len(g)} Spearman={rho:+.3f} p={pp:.3g}")
print("DONE")
