#!/usr/bin/env python3
"""
15_reve_linear_probe.py -- C7 (fair test of the foundation model).
The paper uses REVE frozen + cosine, which is unfair vs a trained EEGNet. The minimally fair test is
a LINEAR PROBE on the frozen embedding. We add: frozen REVE + LDA linear probe (fit on session-1
identities), then cosine-verify in the probe space, and compare to frozen-cosine (paper).
(Full fine-tuning is a heavier follow-up; a linear probe is the standard 'is identity linearly
separable in the frozen space' control.)
Writes: outputs/work3/15_reve_probe/reve_probe.csv
Run   : pip install -q transformers scikit-learn --break-system-packages
        python -u scripts/work3/15_reve_linear_probe.py 2>&1 | tee outputs/work3/15_reve_probe.log
"""
import os, glob, numpy as np, pandas as pd, torch
from sklearn.metrics import roc_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def _has(p): return os.path.isdir(os.path.join(p,"outputs"))
_hp=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
ROOT=next((c for c in [os.getcwd(),_hp,"/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)),os.getcwd())
DATA=os.path.join(ROOT,"data","processed","sessionwise")
OUT=os.path.join(ROOT,"outputs","work3","15_reve_probe"); os.makedirs(OUT,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"; MAXPS=120

from transformers import AutoModel
pos_bank=AutoModel.from_pretrained("brain-bzh/reve-positions",trust_remote_code=True).to(DEV).eval()
model=AutoModel.from_pretrained("brain-bzh/reve-base",trust_remote_code=True).to(DEV).eval()

rng=np.random.default_rng(42); sess={}; ch=None
for f in sorted(glob.glob(os.path.join(DATA,"*.npz"))):
    d=np.load(f,allow_pickle=True)
    if not all(k in d for k in ("X","y_subject","y_session","ch_names")): continue
    sid=int(np.unique(d["y_session"])[0])
    if sid in sess: continue
    ch=[str(c) for c in d["ch_names"]]; X=np.asarray(d["X"],np.float32); y=np.asarray(d["y_subject"],int)
    keep=np.concatenate([rng.permutation(np.where(y==s)[0])[:MAXPS] for s in np.unique(y)])
    sess[sid]=(X[keep],y[keep])
SUBS=sorted(np.unique(sess[1][1]).tolist()); S2R={s:i for i,s in enumerate(SUBS)}
KEEP=[i for i,c in enumerate(ch) if _hasattr(pos_bank,c)] if False else None
KI=[]; KN=[]
for i,c in enumerate(ch):
    try:
        p=pos_bank([c]); n=int(p.shape[0]) if hasattr(p,"shape") else len(p)
        if n>=1: KI.append(i); KN.append(c)
    except Exception: pass
KI=np.asarray(KI,int)

@torch.no_grad()
def embed(X):
    pos=pos_bank(KN); pos=pos.to(DEV); outs=[]
    for i in range(0,len(X),64):
        xb=torch.tensor(X[i:i+64][:,KI,:],dtype=torch.float32,device=DEV)
        pb=pos.unsqueeze(0).expand(xb.size(0),-1,-1); o=model(xb,pb)
        o=o.last_hidden_state if hasattr(o,"last_hidden_state") else (o[0] if isinstance(o,(tuple,list)) else o)
        o=o.mean((1,2)) if o.dim()==4 else (o.mean(1) if o.dim()==3 else o); outs.append(o.float().cpu().numpy())
    E=np.concatenate(outs,0); return E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-9)
emb={k:(embed(sess[k][0]),sess[k][1]) for k in sess}

def eer_auc(y,s):
    fpr,tpr,_=roc_curve(y,s); fnr=1-tpr; i=np.nanargmin(np.abs(fpr-fnr)); return float((fpr[i]+fnr[i])/2)
def verify(space):  # space: dict day-> (features, labels)
    P=np.vstack([space[1][0][space[1][1]==s].mean(0,keepdims=True) for s in SUBS]); P=P/(np.linalg.norm(P,axis=1,keepdims=True)+1e-9)
    res={}
    for day in [1,2,3]:
        if day==1:
            E,y=space[1]; idx=np.arange(len(E)); rng.shuffle(idx); te=idx[len(idx)//2:]; G,lab=E[te],y[te]
        else: G,lab=space[day]
        Sc=G@P.T; gi=np.array([S2R[t] for t in lab]); gen=Sc[np.arange(len(G)),gi]
        mm=np.ones_like(Sc,bool); mm[np.arange(len(G)),gi]=False
        res[day]=eer_auc(np.r_[np.ones(len(gen)),np.zeros(mm.sum())],np.r_[gen,Sc[mm]])
    return res

frozen=verify(emb)
# linear probe: LDA fit on session-1 identities, transform all sessions, re-verify
lda=LinearDiscriminantAnalysis(n_components=min(len(SUBS)-1,emb[1][0].shape[1])).fit(emb[1][0],emb[1][1])
probe={k:(lda.transform(emb[k][0]),emb[k][1]) for k in emb}
for k in probe: probe[k]=(probe[k][0]/(np.linalg.norm(probe[k][0],axis=1,keepdims=True)+1e-9),probe[k][1])
probed=verify(probe)

rows=[]
for day,tag in [(1,"within"),(2,"S1->S2"),(3,"S1->S3")]:
    rows.append({"protocol":tag,"REVE_frozen_cosine":round(frozen[day],4),"REVE_linear_probe":round(probed[day],4)})
    print(f"  {tag:8s} frozen={frozen[day]:.4f}  linear-probe={probed[day]:.4f}")
pd.DataFrame(rows).to_csv(os.path.join(OUT,"reve_probe.csv"),index=False)
print("  READ-OFF (C7): report both. If the linear probe still degrades across sessions, the fair test confirms")
print("  frozen foundation features do not solve drift; if it closes the gap, soften the REVE claim accordingly.")
print("DONE ->",os.path.relpath(OUT,ROOT))
