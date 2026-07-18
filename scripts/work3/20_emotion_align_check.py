import os,glob,re,itertools,importlib.util,numpy as np,pandas as pd
from scipy.signal import welch
from scipy import stats
def _has(p): return os.path.isdir(os.path.join(p,"outputs"))
_hp=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
ROOT=next((c for c in [os.getcwd(),_hp,"/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC","/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)),os.getcwd())
DATA=os.path.join(ROOT,"data","processed","sessionwise")
def sess_num(v):
    s=str(v);m=re.search(r'SESSION\s*_?(\d+)',s,re.I);return m.group(1) if m else (re.search(r'(\d+)',s).group(1) if re.search(r'(\d+)',s) else s)
_sp=importlib.util.spec_from_file_location("p6",os.path.join(ROOT,"scripts","06_q1_levelup_biological_embedding_analysis.py"))
p6=importlib.util.module_from_spec(_sp);_sp.loader.exec_module(p6)
BANDS=p6.BANDS
sess={}
for f in sorted(glob.glob(os.path.join(DATA,"*.npz"))):
    d=np.load(f,allow_pickle=True)
    if not all(k in d for k in ("X","y_subject","y_session","y_emotion")):continue
    sid=int(np.unique(d["y_session"])[0])
    if sid in sess:continue
    sess[sid]={"X":np.asarray(d["X"],np.float32),"emo":np.asarray(d["y_emotion"],int),"ch":[str(c) for c in d["ch_names"]],"fs":int(np.array(d["fs"]).item()) if "fs" in d.files else 200}
CH=sess[1]["ch"];FS=sess[1]["fs"];REGION=[p6.region_of_channel(c) for c in CH];REGS=sorted(set(REGION))
EMOS=sorted(np.unique(sess[1]["emo"]).tolist())
def theta_region(X):
    f,psd=welch(X.astype(np.float64),fs=FS,nperseg=min(256,X.shape[-1]),axis=-1)
    lo,hi=BANDS["theta"];th=psd[:,:,(f>=lo)&(f<hi)].mean(axis=(0,2))
    return {r:float(np.mean(th[[i for i,rr in enumerate(REGION) if rr==r]])) for r in REGS}
cond={}
for s in sess:
    for e in EMOS:
        m=sess[s]["emo"]==e
        if m.sum()>=5: cond[(s,e)]=theta_region(sess[s]["X"][m])
recs=[]
for st in[2,3]:
    for ee in EMOS:
        for te in EMOS:
            if (1,ee) in cond and (st,te) in cond:
                recs.append({"transition":f"1->{st}","ee":ee,"te":te,
                    "theta_drift":float(np.mean([abs(cond[(1,ee)][r]-cond[(st,te)][r]) for r in REGS]))})
dd=pd.DataFrame(recs)
csv=sorted(glob.glob(os.path.join(ROOT,"outputs","**","merged_global_psd_identity_eer.csv"),recursive=True),key=len)[0]
mg=pd.read_csv(csv)
if "variant" in mg.columns and (mg["variant"]=="arcface_supcon_cnn").any():mg=mg[mg["variant"]=="arcface_supcon_cnn"]
mg["transition"]="1->"+mg["test_session"].map(sess_num)
mg["ep"]=mg["enroll_emotion"].astype(str)+"|"+mg["test_emotion"].astype(str)
ck=mg.groupby(["transition","ep"])["theta_power_drift"].mean().reset_index()
NAMES=sorted(set(mg["enroll_emotion"].astype(str))|set(mg["test_emotion"].astype(str)))
print("emotion int ids:",EMOS,"| CSV emotion names:",NAMES)
best=(-2,None,None)
for perm in itertools.permutations(NAMES):
    mp=dict(zip(EMOS,perm))
    for swap in [False,True]:
        d2=dd.copy()
        if not swap: d2["ep"]=d2["ee"].map(mp)+"|"+d2["te"].map(mp)
        else: d2["ep"]=d2["te"].map(mp)+"|"+d2["ee"].map(mp)
        j=d2.merge(ck,on=["transition","ep"],how="inner")
        if len(j)<40: continue
        r=np.corrcoef(j["theta_drift"],j["theta_power_drift"])[0,1]
        if r>best[0]: best=(r,mp,swap)
print(f"BEST faithfulness r={best[0]:+.3f}  mapping={best[1]}  swap_enroll_test={best[2]}")
print("READ-OFF: r>0.85 => it WAS an emotion-label mapping issue (mapping shown is correct); theta is faithfully")
print("reproducible and I redo C8/C12/C20 with this mapping. r still low => theta does NOT reproduce on full data.")
