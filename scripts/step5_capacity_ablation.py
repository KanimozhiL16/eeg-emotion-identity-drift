#!/usr/bin/env python3
"""
step5_capacity_ablation.py  -- "drift, not capacity" ablation (rebuts Ozdenizci 2019).

Trains EEGNet at increasing model capacity and shows the CROSS-SESSION EER gap does
NOT close as capacity grows -> the limit is drift, not representational capacity.

For each capacity: train identity classifier on session 1, build per-subject
prototypes (S1), verify by cosine on S1-heldout / S2 / S3 -> EER. Reports params,
within-session EER, cross-session EER, and the within->cross GAP per capacity.

USAGE (project root, env active; GPU auto):
    cd /home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
    source p4_seedv_env/bin/activate
    mkdir -p outputs/run_14_capacity
    python -u scripts/step5_capacity_ablation.py 2>&1 | tee outputs/run_14_capacity/log.txt
Outputs -> outputs/run_14_capacity/capacity_ablation.csv
"""
import os, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import torch, torch.nn as nn

RNG = np.random.default_rng(0); MAXP = 400
def has(p): return os.path.isdir(os.path.join(p,"outputs"))
_hp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
ROOT = next((c for c in [os.getcwd(),_hp,
        "/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
        "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if has(c)), os.getcwd())
OUT = os.path.join(ROOT,"outputs","run_14_capacity"); os.makedirs(OUT,exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("="*78); print("STEP 5 CAPACITY ABLATION | root:",ROOT,"| device:",dev); print("="*78)

def load(n):
    f=glob.glob(os.path.join(ROOT,"**",f"*SESSION{n}_16sub.npz"),recursive=True)
    z=np.load(sorted(f,key=len)[0],allow_pickle=True); return z["X"].astype(np.float32), z["y_subject"].astype(int)
X1,y1=load(1); X2,y2=load(2); X3,y3=load(3); SUBS=sorted(np.unique(y1).tolist())
mu=X1.mean((0,2),keepdims=True); sd=X1.std((0,2),keepdims=True)+1e-6
nrm=lambda X:((X-mu)/sd).astype(np.float32)
def sub(X,y,cap):
    k=[];
    for s in np.unique(y):
        idx=np.where(y==s)[0]
        if len(idx)>cap: idx=RNG.choice(idx,cap,replace=False)
        k+=list(idx)
    k=np.array(sorted(k)); return X[k],y[k]
def l2(F): return F/(np.linalg.norm(F,axis=1,keepdims=True)+1e-8)
def protos(F,y): return l2(np.stack([l2(F[y==s]).mean(0) for s in SUBS]))
def eer(F_te,y_te,P):
    F=l2(F_te); S=F@P.T
    gen=S[np.arange(len(F)),[SUBS.index(t) for t in y_te]]
    m=np.ones_like(S,bool); m[np.arange(len(F)),[SUBS.index(t) for t in y_te]]=False
    yv=np.r_[np.ones(len(gen)),np.zeros(m.sum())]; sv=np.r_[gen,S[m]]
    fpr,tpr,_=roc_curve(yv,sv); fnr=1-tpr
    return float((fpr[np.nanargmin(np.abs(fpr-fnr))]+fnr[np.nanargmin(np.abs(fpr-fnr))])/2), float(roc_auc_score(yv,sv))

class EEGNet(nn.Module):
    def __init__(s,C=62,ncl=16,F1=8,D=2,F2=16,k=64):
        super().__init__()
        s.f1=nn.Sequential(nn.Conv2d(1,F1,(1,k),padding=(0,k//2),bias=False),nn.BatchNorm2d(F1))
        s.dw=nn.Sequential(nn.Conv2d(F1,F1*D,(C,1),groups=F1,bias=False),nn.BatchNorm2d(F1*D),nn.ELU(),nn.AvgPool2d((1,4)),nn.Dropout(0.5))
        s.sep=nn.Sequential(nn.Conv2d(F1*D,F1*D,(1,16),padding=(0,8),groups=F1*D,bias=False),nn.Conv2d(F1*D,F2,(1,1),bias=False),nn.BatchNorm2d(F2),nn.ELU(),nn.AvgPool2d((1,8)),nn.Dropout(0.5))
        s.flat=nn.Flatten(); s.head=nn.LazyLinear(ncl)
    def embed(s,x): return s.flat(s.sep(s.dw(s.f1(x))))
    def forward(s,x): return s.head(s.embed(x))
def tt(X): return torch.tensor(X[:,None,:,:],dtype=torch.float32)

CAPS = {"tiny":(4,1,4),"small":(8,2,16),"base":(16,2,32),"large":(16,4,64),"xlarge":(32,4,128)}
rows=[]
for name,(F1,D,F2) in CAPS.items():
    net=EEGNet(F1=F1,D=D,F2=F2).to(dev); opt=torch.optim.Adam(net.parameters(),1e-3); lf=nn.CrossEntropyLoss()
    Xtr,ytr=sub(nrm(X1),y1,1200); lab=np.array([SUBS.index(s) for s in ytr])
    net(tt(Xtr[:8]).to(dev)); npar=sum(p.numel() for p in net.parameters())
    for ep in range(20):
        net.train(); perm=RNG.permutation(len(Xtr))
        for i in range(0,len(perm),256):
            b=perm[i:i+256]; opt.zero_grad(); loss=lf(net(tt(Xtr[b]).to(dev)),torch.tensor(lab[b]).to(dev)); loss.backward(); opt.step()
    net.eval()
    @torch.no_grad()
    def emb(X):
        X=nrm(X); o=[]
        for i in range(0,len(X),512): o.append(net.embed(tt(X[i:i+512]).to(dev)).cpu().numpy())
        return np.concatenate(o).astype(np.float32)
    F1e=emb(X1)
    en,te=[],[]
    for s in SUBS:
        idx=np.where(y1==s)[0]; RNG.shuffle(idx); h=len(idx)//2; en+=list(idx[:h]); te+=list(idx[h:])
    en,te=np.array(en),np.array(te)
    Ph=protos(F1e[en],y1[en]); Xs,ys=sub(F1e[te],y1[te],MAXP); e_in,a_in=eer(Xs,ys,Ph)
    Pf=protos(F1e,y1)
    Xs2,ys2=sub(emb(X2),y2,MAXP); e2,_=eer(Xs2,ys2,Pf)
    Xs3,ys3=sub(emb(X3),y3,MAXP); e3,_=eer(Xs3,ys3,Pf)
    gap=((e2+e3)/2)-e_in
    rows.append({"capacity":name,"params":npar,"EER_within":round(e_in,4),
                 "EER_S2":round(e2,4),"EER_S3":round(e3,4),"within_to_cross_gap":round(gap,4)})
    print(f"  {name:6s} params={npar:8d} | within={e_in:.4f}  S2={e2:.4f}  S3={e3:.4f}  GAP={gap:.4f}")

df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,"capacity_ablation.csv"),index=False)
print("\n"+"="*78); print(df.to_string(index=False))
print("\nINTERPRETATION: if EER_within stays low while S2/S3 and GAP do NOT shrink as")
print("params grow, the cross-session limit is DRIFT, not capacity (rebuts Ozdenizci 2019).")
print("SAVED:", os.path.relpath(os.path.join(OUT,"capacity_ablation.csv"),ROOT)); print("="*78)
