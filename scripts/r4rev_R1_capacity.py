#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R1: CAPACITY ABLATION, LEAKAGE-FREE (whole-trial-disjoint), faithful to step5_capacity_ablation.py.
Same 5 EEGNet capacities and training recipe; the ONLY change is a fair split:
  S1 trials are partitioned into TRAIN / ENROL / TEST trials (disjoint).
  - encoder trained ONLY on TRAIN-trial windows (no eval trial in training)
  - per-subject prototypes built from ENROL-trial windows
  - within-session EER = TEST-trial windows vs ENROL prototypes (trial-disjoint)
  - cross-session EER = S2 / S3 windows vs the SAME ENROL prototypes (matched budget)
Runs 3 seeds/capacity -> mean +/- SD (also answers the single-seed objection).
USAGE:
  cd /lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
  source fignn_env/bin/activate 2>/dev/null || source p4_seedv_env/bin/activate
  python -u r4rev_R1_capacity.py 2>&1 | tee r1_capacity_log.txt
Outputs -> outputs/run_reanalysis/R1_capacity_trialdisjoint.csv
"""
import os, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import torch, torch.nn as nn
def has(p): return os.path.isdir(os.path.join(p,"outputs"))
ROOT=next((c for c in ["/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
        "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC", os.getcwd()] if has(c)), os.getcwd())
OUT=os.path.join(ROOT,"outputs","run_reanalysis"); os.makedirs(OUT,exist_ok=True)
DATA=os.path.join(ROOT,"data","processed","sessionwise")
dev="cuda" if torch.cuda.is_available() else "cpu"
CAPS={"tiny":(4,1,4),"small":(8,2,16),"base":(16,2,32),"large":(16,4,64),"xlarge":(32,4,128)}
MAXP=400; SEEDS=[0,1,2]
print("="*80); print("R1 CAPACITY LEAKAGE-FREE | root:",ROOT,"| dev:",dev); print("="*80)

def load(n):
    f=sorted(glob.glob(os.path.join(DATA,f"*SESSION{n}_16sub.npz")),key=len)[0]
    z=np.load(f,allow_pickle=True)
    return z["X"].astype(np.float32), z["y_subject"].astype(int), z["y_trial"].astype(int)
X1,y1,t1=load(1); X2,y2,t2=load(2); X3,y3,t3=load(3); SUBS=sorted(np.unique(y1).tolist())
mu=X1.mean((0,2),keepdims=True); sd=X1.std((0,2),keepdims=True)+1e-6
nrm=lambda X:((X-mu)/sd).astype(np.float32)
def l2(F): return F/(np.linalg.norm(F,axis=1,keepdims=True)+1e-8)
def protos(F,y): return l2(np.stack([l2(F[y==s]).mean(0) for s in SUBS]))
def eer(F_te,y_te,P):
    F=l2(F_te); S=F@P.T
    gen=S[np.arange(len(F)),[SUBS.index(t) for t in y_te]]
    m=np.ones_like(S,bool); m[np.arange(len(F)),[SUBS.index(t) for t in y_te]]=False
    yv=np.r_[np.ones(len(gen)),np.zeros(m.sum())]; sv=np.r_[gen,S[m]]
    fpr,tpr,_=roc_curve(yv,sv); fnr=1-tpr; i=int(np.nanargmin(np.abs(fpr-fnr)))
    return float((fpr[i]+fnr[i])/2), float(roc_auc_score(yv,sv))
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
def cap(idx,n,rng): idx=np.asarray(idx); return rng.choice(idx,n,replace=False) if len(idx)>n else idx

TRIALS=sorted(np.unique(t1).tolist())   # 15 trials
rows=[]
for name,(F1,D,F2) in CAPS.items():
    per=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed); torch.manual_seed(seed)
        tr=TRIALS.copy(); rng.shuffle(tr)
        train_tr=set(tr[:9]); enrol_tr=set(tr[9:12]); test_tr=set(tr[12:])   # 9/3/3 disjoint
        tr_mask=np.isin(t1,list(train_tr))
        Xtr=nrm(X1[tr_mask]); ytr=y1[tr_mask]
        # cap ~1200/subject for training (as in original)
        keep=[]
        for s in SUBS:
            si=np.where(ytr==s)[0]
            keep+=list(rng.choice(si,min(1200,len(si)),replace=False))
        keep=np.array(sorted(keep)); Xtr=Xtr[keep]; lab=np.array([SUBS.index(s) for s in ytr[keep]])
        net=EEGNet(F1=F1,D=D,F2=F2).to(dev); opt=torch.optim.Adam(net.parameters(),1e-3); lf=nn.CrossEntropyLoss()
        net(tt(Xtr[:8]).to(dev)); npar=sum(p.numel() for p in net.parameters())
        for ep in range(20):
            net.train(); perm=rng.permutation(len(Xtr))
            for i in range(0,len(perm),256):
                b=perm[i:i+256]; opt.zero_grad(); loss=lf(net(tt(Xtr[b]).to(dev)),torch.tensor(lab[b]).to(dev)); loss.backward(); opt.step()
        net.eval()
        @torch.no_grad()
        def emb(X):
            X=nrm(X); o=[]
            for i in range(0,len(X),512): o.append(net.embed(tt(X[i:i+512]).to(dev)).cpu().numpy())
            return np.concatenate(o).astype(np.float32)
        # prototypes from ENROL trials of S1
        en=np.where(np.isin(t1,list(enrol_tr)))[0]
        Fen=emb(X1[en]); P=protos(Fen,y1[en])
        # within-session (trial-disjoint): TEST trials of S1
        teix=np.where(np.isin(t1,list(test_tr)))[0]
        Ft=emb(X1[teix]); yt=y1[teix]
        keep2=np.concatenate([cap(np.where(yt==s)[0],MAXP,rng) for s in SUBS])
        e_in,_=eer(Ft[keep2],yt[keep2],P)
        # cross-session vs same prototypes
        F2e=emb(X2); k2=np.concatenate([cap(np.where(y2==s)[0],MAXP,rng) for s in SUBS]); e2,_=eer(F2e[k2],y2[k2],P)
        F3e=emb(X3); k3=np.concatenate([cap(np.where(y3==s)[0],MAXP,rng) for s in SUBS]); e3,_=eer(F3e[k3],y3[k3],P)
        per.append((e_in,e2,e3));
        print(f"  {name:6s} seed{seed} params={npar} | within(TD)={e_in:.4f}  S2={e2:.4f}  S3={e3:.4f}")
    per=np.array(per); m=per.mean(0); s_=per.std(0,ddof=1)
    gap=((m[1]+m[2])/2)-m[0]
    rows.append(dict(capacity=name,params=npar,
        within_TD_mean=round(m[0],4),within_TD_sd=round(s_[0],4),
        S2_mean=round(m[1],4),S2_sd=round(s_[1],4),S3_mean=round(m[2],4),S3_sd=round(s_[2],4),
        gap=round(gap,4)))
    print(f"  >> {name:6s} within(TD)={m[0]:.4f}±{s_[0]:.4f}  S2={m[1]:.4f}±{s_[1]:.4f}  S3={m[2]:.4f}±{s_[2]:.4f}  gap={gap:+.4f}")
df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,"R1_capacity_trialdisjoint.csv"),index=False)
print("\n"+"="*80); print(df.to_string(index=False))
print("\nINTERPRET: compare within_TD to the leaky within (0.011-0.016 in Table S-C).")
print("If within_TD is high (~0.2-0.3) and does NOT fall as params grow, the 'capacity")
print("drives within-session to ~0' trend was a leakage artefact; the honest claim is that")
print("cross-session EER (S2/S3) stays flat across a 38x parameter range.")
print("SAVED:", os.path.join(OUT,"R1_capacity_trialdisjoint.csv")); print("="*80)
