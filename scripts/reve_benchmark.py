#!/usr/bin/env python3
"""
reve_benchmark.py  -- Revision experiment R2: foundation-model (REVE) baseline under our protocol.

Answers the reviewer's #1 request: does a modern pretrained EEG foundation encoder reduce cross-
session drift? We use REVE (brain-bzh/reve-base, NeurIPS 2025) as a FROZEN feature extractor, enrol
session-1 prototypes, verify sessions 2 and 3 by cosine, and report global EER/AUC per session --
directly comparable to the PSD and EEGNet rows in Table 1.

Requires: transformers, torch (GPU). First run downloads the model (~hundreds of MB).
Outputs -> outputs/run_18_reve_benchmark/ : reve_eer.csv
USAGE:  pip install -q transformers --break-system-packages ; python scripts/reve_benchmark.py
"""
import os, glob, numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import torch

ROOT="/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"
if not os.path.isdir(ROOT): ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
DATA=os.path.join(ROOT,"data","processed","sessionwise"); OUT=os.path.join(ROOT,"outputs","run_18_reve_benchmark"); os.makedirs(OUT,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"; MAXPS=120  # windows per subject per session (runtime cap)
print("="*74); print(f"R2  REVE foundation-model benchmark  (device={DEV})"); print("="*74)

# ---- load REVE (frozen) ----
try:
    from transformers import AutoModel
    pos_bank=AutoModel.from_pretrained("brain-bzh/reve-positions",trust_remote_code=True).to(DEV).eval()
    model  =AutoModel.from_pretrained("brain-bzh/reve-base",trust_remote_code=True).to(DEV).eval()
except Exception as e:
    print("ERROR loading REVE -- install transformers and ensure network access:\n ",e); raise SystemExit(1)

# ---- load SEED-V sessionwise data (subsampled) ----
rng=np.random.default_rng(42); sess={}; ch_names=None
for f in sorted(glob.glob(os.path.join(DATA,"*.npz"))):
    d=np.load(f,allow_pickle=True)
    if not all(k in d for k in ("X","y_subject","y_session","ch_names")): continue
    sid=int(np.unique(d["y_session"])[0])
    if sid in sess: continue
    ch_names=[str(c) for c in d["ch_names"]]; X=np.asarray(d["X"],np.float32); y=np.asarray(d["y_subject"],int)
    keep=[]
    for s in np.unique(y):
        idx=np.where(y==s)[0]; rng.shuffle(idx); keep.append(idx[:MAXPS])
    keep=np.concatenate(keep)
    sess[sid]=(X[keep], y[keep])
    print(f"  session {sid}: {X[keep].shape}")
SUBS=sorted(np.unique(sess[1][1]).tolist()); S2R={s:i for i,s in enumerate(SUBS)}

# ---- align channels to REVE position vocabulary (drop names REVE can't place, e.g. CB1/CB2) ----
def _npos(p): return int(p.shape[0]) if hasattr(p,"shape") else len(p)
KEEP_IDX=[]; KEEP_NAMES=[]
for i,c in enumerate(ch_names):
    try:
        if _npos(pos_bank([c]))>=1: KEEP_IDX.append(i); KEEP_NAMES.append(c)
    except Exception:
        pass
KEEP_IDX=np.asarray(KEEP_IDX,int)
print(f"  channels kept by REVE: {len(KEEP_NAMES)}/{len(ch_names)}  (dropped: {[c for c in ch_names if c not in KEEP_NAMES]})")

_DBG=[True]
@torch.no_grad()
def embed(X):
    pos=pos_bank(KEEP_NAMES); pos=pos.to(DEV)
    outs=[]
    for i in range(0,len(X),64):
        xb=torch.tensor(X[i:i+64][:,KEEP_IDX,:],dtype=torch.float32,device=DEV)
        pb=pos.unsqueeze(0).expand(xb.size(0),-1,-1)
        o=model(xb,pb)
        o=o.last_hidden_state if hasattr(o,"last_hidden_state") else (o[0] if isinstance(o,(tuple,list)) else o)
        if _DBG[0]:
            print(f"  [debug] raw REVE output shape = {tuple(o.shape)}  -> pooling all axes except batch+embed_dim"); _DBG[0]=False
        if o.dim()==4:   o=o.mean((1,2))   # (B, channels, patches, embed_dim) -> (B, embed_dim)
        elif o.dim()==3: o=o.mean(1)        # (B, tokens, embed_dim) -> (B, embed_dim)
        outs.append(o.float().cpu().numpy())
    E=np.concatenate(outs,0)
    if E.shape[1]<32: print(f"  [debug] pooled embedding dim = {E.shape[1]} (looks wrong if <32)")
    return E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-9)

emb={k:(embed(sess[k][0]),sess[k][1]) for k in sess}
P=np.vstack([emb[1][0][emb[1][1]==s].mean(0,keepdims=True) for s in SUBS]); P=P/(np.linalg.norm(P,axis=1,keepdims=True)+1e-9)
def eer_auc(y,s):
    fpr,tpr,_=roc_curve(y,s); fnr=1-tpr; i=np.nanargmin(np.abs(fpr-fnr)); return float((fpr[i]+fnr[i])/2), float(roc_auc_score(y,s))

import csv; rows=[]
print("\n  session :  REVE global EER  | AUC")
for day in [1,2,3]:
    if day==1:  # within-session reference: 50/50 split of session 1
        E,y=emb[1]; idx=np.arange(len(E)); rng.shuffle(idx); te=idx[len(idx)//2:]
        G=E[te]; lab=y[te]
    else:
        G,lab=emb[day]
    Sc=G@P.T; gi=np.array([S2R[t] for t in lab])
    gen=Sc[np.arange(len(G)),gi]; mmask=np.ones_like(Sc,bool); mmask[np.arange(len(G)),gi]=False
    e,a=eer_auc(np.r_[np.ones(len(gen)),np.zeros(mmask.sum())],np.r_[gen,Sc[mmask]])
    tag={1:"within-session",2:"S1->S2",3:"S1->S3"}[day]
    rows.append({"protocol":tag,"REVE_EER":round(e,4),"REVE_AUC":round(a,4)}); print(f"  {tag:>13} :   {e:.4f}       | {a:.4f}")

with open(os.path.join(OUT,"reve_eer.csv"),"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=["protocol","REVE_EER","REVE_AUC"]); w.writeheader(); w.writerows(rows)
print("="*74)
print("Compare to Table 1: PSD 0.183/0.229/0.272 ; EEGNet 0.011/0.190/0.245.")
print("If REVE cross-session EER is also high, drift is NOT solved by a foundation encoder (key revision point).")
print("PASS -- saved outputs/run_18_reve_benchmark/reve_eer.csv"); print("="*74)
