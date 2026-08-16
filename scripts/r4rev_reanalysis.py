#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REVIEWER REANALYSIS R2, R3, R4 (read-only on data; writes outputs/run_reanalysis/).
  R2  2x2 emotion x session, FULLY TRIAL-DISJOINT in every cell, matched budgets (PSD+cosine).
  R3  theta regression ADJUSTED for session transition (S1->S2 vs S1->S3); per-transition slopes.
  R4  manipulation check with BALANCED accuracy + TRIAL-level label-permutation null.
USAGE:
  cd /lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
  source fignn_env/bin/activate 2>/dev/null || source p4_seedv_env/bin/activate
  python -u r4rev_reanalysis.py 2>&1 | tee reanalysis_log.txt
"""
import os, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.signal import welch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
RNG=np.random.default_rng(0)
ROOT=next((c for c in ["/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
        "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",os.getcwd()] if os.path.isdir(c)),os.getcwd())
DATA=os.path.join(ROOT,"data","processed","sessionwise")
OUT=os.path.join(ROOT,"outputs","run_reanalysis"); os.makedirs(OUT,exist_ok=True)
BANDS=[(0.5,4),(4,8),(8,13),(13,30),(30,45)]; EMO={0:"disgust",1:"fear",2:"sad",3:"neutral",4:"happy"}
CAP=200
print("="*80); print("REANALYSIS | root:",ROOT); print("="*80)

def load(n):
    f=sorted(glob.glob(os.path.join(DATA,f"*SESSION{n}_16sub.npz")),key=len)[0]
    z=np.load(f,allow_pickle=True)
    return (z["X"].astype(np.float32),z["y_subject"].astype(int),
            z["y_trial"].astype(int),z["y_emotion"].astype(int))
def psd(X):
    f,P=welch(X,fs=200,nperseg=200,axis=-1)
    return np.concatenate([np.log(P[:,:,(f>=lo)&(f<hi)].sum(-1)+1e-12) for lo,hi in BANDS],1).astype(np.float32)
def l2(F): return F/(np.linalg.norm(F,axis=1,keepdims=True)+1e-8)
def eer(y,s):
    fpr,tpr,_=roc_curve(y,s); fnr=1-tpr; i=int(np.nanargmin(np.abs(fpr-fnr)))
    return float((fpr[i]+fnr[i])/2), float(roc_auc_score(y,s))
def capidx(idx):
    idx=np.asarray(idx); return RNG.choice(idx,CAP,replace=False) if len(idx)>CAP else idx

# =================================================================== R2
def R2():
    print("\n"+"#"*80); print("# R2  2x2 emotion x session, FULLY TRIAL-DISJOINT, matched budgets"); print("#"*80)
    X1,s1,t1,e1=load(1); X2,s2,t2,e2=load(2); X3,s3,t3,e3=load(3)
    SUBS=sorted(np.unique(s1).tolist()); F1=psd(X1); F2=psd(X2); F3=psd(X3)
    emos=sorted(np.unique(e1).tolist())
    rows=[]   # per (subject, cell) EER
    for s in SUBS:
        # enrol emotion e0: split its S1 trials into enrol vs held-out (trial-disjoint)
        for e0 in emos:
            tr=sorted(np.unique(t1[(s1==s)&(e1==e0)]).tolist())
            if len(tr)<2: continue
            RNG.shuffle(tr); en_tr={tr[0]}; ho_tr=set(tr[1:])   # enrol clip vs held-out clip(s), same emotion
            en=np.where((s1==s)&(e1==e0)&(np.isin(t1,list(en_tr))))[0]
            if len(en)<5: continue
            en=capidx(en); templ=l2(l2(F1[en]).mean(0)[None,:])[0]
            def cell(Fp,sp,ep,tp,cross,diff):
                if not diff:  # same emotion e0
                    gi=np.where((sp==s)&(ep==e0)&(~np.isin(tp,list(en_tr)) if not cross else np.ones(len(sp),bool)))[0]
                    ii=np.where((sp!=s)&(ep==e0))[0]
                else:         # different emotion
                    gi=np.where((sp==s)&(ep!=e0))[0]; ii=np.where((sp!=s)&(ep!=e0))[0]
                if len(gi)<5 or len(ii)<20: return None
                gi=capidx(gi); ii=capidx(ii)
                sg=l2(Fp[gi])@templ; si=l2(Fp[ii])@templ
                y=np.r_[np.ones(len(sg)),np.zeros(len(si))]; sc=np.r_[sg,si]
                return eer(y,sc)[0]
            base=cell(F1,s1,e1,t1,False,False)      # same-session, same-emotion (trial-disjoint: held-out clip)
            eOnly=cell(F1,s1,e1,t1,False,True)       # same-session, diff-emotion
            sOnly=cell(np.vstack([F2,F3]),np.r_[s2,s3],np.r_[e2,e3],np.r_[t2,t3],True,False)  # cross-session same-emo
            both =cell(np.vstack([F2,F3]),np.r_[s2,s3],np.r_[e2,e3],np.r_[t2,t3],True,True)    # cross-session diff-emo
            if None not in (base,eOnly,sOnly,both):
                rows.append(dict(subject=s,enrol_emo=EMO[e0],baseline=base,emo_only=eOnly,sess_only=sOnly,both=both))
    df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,"R2_2x2_trialdisjoint_percell.csv"),index=False)
    # per-subject average across enrol emotions
    g=df.groupby("subject")[["baseline","emo_only","sess_only","both"]].mean().reset_index()
    g.to_csv(os.path.join(OUT,"R2_2x2_persubject.csv"),index=False)
    print(f"per-subject cells (n={len(g)}):")
    for c in ["baseline","emo_only","sess_only","both"]:
        print(f"  {c:10s} mean EER {g[c].mean():.4f} ± {g[c].std(ddof=1):.4f}")
    dE=g["emo_only"]-g["baseline"]; dS=g["sess_only"]-g["baseline"]
    wE=wilcoxon(g["emo_only"],g["baseline"]); wS=wilcoxon(g["sess_only"],g["baseline"])
    print(f"  EMOTION mismatch  delta EER = {dE.mean():+.4f}  Wilcoxon p={wE.pvalue:.4g}")
    print(f"  SESSION mismatch  delta EER = {dS.mean():+.4f}  Wilcoxon p={wS.pvalue:.4g}")
    print("  (compare to paper's window-level +0.047 emotion / +0.077 session)")

# =================================================================== R3
def R3():
    print("\n"+"#"*80); print("# R3  theta regression ADJUSTED for session transition"); print("#"*80)
    import statsmodels.formula.api as smf
    cand=glob.glob(os.path.join(ROOT,"**","merged_global_psd_identity_eer.csv"),recursive=True)
    if not cand: print("  [MISS] merged_global_psd_identity_eer.csv not found; paste its path."); return
    df=pd.read_csv(sorted(cand,key=len)[0]); print("  file:",sorted(cand,key=len)[0]); print("  cols:",list(df.columns))
    low={c.lower():c for c in df.columns}
    def col(*keys):
        for k in keys:
            for lc,orig in low.items():
                if k in lc: return orig
        return None
    theta=col("theta"); alpha=col("alpha"); beta=col("beta"); gamma=col("gamma")
    eerc=col("eer"); tsess=col("test_session","test_sess"); esess=col("enroll_session","enrol_session")
    eem=col("enroll_emotion","enrol_emotion"); tem=col("test_emotion")
    print(f"  detected: theta={theta} alpha={alpha} beta={beta} gamma={gamma} EER={eerc} test_session={tsess}")
    if None in (theta,eerc,tsess):
        print("  [ADAPT NEEDED] paste the column list above."); return
    keys=[c for c in [esess,tsess,eem,tem] if c]
    d=df.groupby(keys,as_index=False)[[c for c in [theta,alpha,beta,gamma,eerc] if c]].mean()
    d=d.rename(columns={theta:"theta",eerc:"EER",tsess:"trans"})
    for c0,nm in [(alpha,"alpha"),(beta,"beta"),(gamma,"gamma")]:
        if c0: d=d.rename(columns={c0:nm})
    d["trans"]=d["trans"].astype(str)
    print(f"  n conditions = {len(d)}; transitions: {sorted(d['trans'].unique())}")
    # (1) unadjusted multiband
    m1=smf.ols("EER ~ theta"+("".join([f" + {b}" for b in ['alpha','beta','gamma'] if b in d])),data=d).fit()
    print(f"  [unadjusted]  theta beta={m1.params['theta']:+.4f}  p={m1.pvalues['theta']:.4g}")
    # (2) transition-adjusted
    m2=smf.ols("EER ~ theta + C(trans)"+("".join([f" + {b}" for b in ['alpha','beta','gamma'] if b in d])),data=d).fit()
    print(f"  [+transition] theta beta={m2.params['theta']:+.4f}  p={m2.pvalues['theta']:.4g}   <-- key")
    # (3) per-transition slopes
    for tv in sorted(d["trans"].unique()):
        dt=d[d["trans"]==tv]
        if len(dt)>=6:
            mt=smf.ols("EER ~ theta",data=dt).fit()
            print(f"  [{tv}] theta beta={mt.params['theta']:+.4f}  p={mt.pvalues['theta']:.4g}  (n={len(dt)})")
    # (4) within-transition centred
    d["theta_c"]=d.groupby("trans")["theta"].transform(lambda x:x-x.mean())
    d["EER_c"]=d.groupby("trans")["EER"].transform(lambda x:x-x.mean())
    m4=smf.ols("EER_c ~ theta_c",data=d).fit()
    print(f"  [within-transition centred] theta beta={m4.params['theta_c']:+.4f}  p={m4.pvalues['theta_c']:.4g}")
    d.to_csv(os.path.join(OUT,"R3_theta_conditions.csv"),index=False)
    print("  INTERPRET: if theta stays significant with C(trans) and within-transition, it is NOT merely S2-vs-S3.")

# =================================================================== R4
def R4():
    print("\n"+"#"*80); print("# R4  manipulation check: BALANCED accuracy + TRIAL-level permutation"); print("#"*80)
    X1,s1,t1,e1=load(1); F1=psd(X1); SUBS=sorted(np.unique(s1).tolist()); rows=[]
    for s in SUBS:
        idx=np.where(s1==s)[0]; X=F1[idx]; y=e1[idx]; tr=t1[idx]
        trs=np.unique(tr)
        if len(np.unique(y))<2 or len(trs)<5: continue
        # leave-one-trial-out CV -> balanced accuracy
        preds=np.zeros(len(y),int)
        for hd in trs:
            te=tr==hd; trn=~te
            if len(np.unique(y[trn]))<2: preds[te]=y[trn][0] if trn.any() else 0; continue
            clf=LinearDiscriminantAnalysis().fit(X[trn],y[trn]); preds[te]=clf.predict(X[te])
        bacc=balanced_accuracy_score(y,preds)
        # trial-level permutation null: permute emotion labels ACROSS trials
        tr2emo={int(tt):int(e1[idx][tr==tt][0]) for tt in trs}
        obs=bacc; nperm=200; ge=0
        for _ in range(nperm):
            perm=RNG.permutation(list(tr2emo.values()))
            pmap=dict(zip(tr2emo.keys(),perm)); yp=np.array([pmap[int(tt)] for tt in tr])
            pr=np.zeros(len(yp),int)
            for hd in trs:
                te=tr==hd; trn=~te
                if len(np.unique(yp[trn]))<2: pr[te]=yp[trn][0] if trn.any() else 0; continue
                pr[te]=LinearDiscriminantAnalysis().fit(X[trn],yp[trn]).predict(X[te])
            if balanced_accuracy_score(yp,pr)>=obs: ge+=1
        pval=(ge+1)/(nperm+1)
        rows.append(dict(subject=s,balanced_acc=round(bacc,4),perm_p=round(pval,4)))
        print(f"  S{s:02d}: balanced-acc={bacc:.3f}  trial-perm p={pval:.3f}")
    df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,"R4_manipcheck_balanced.csv"),index=False)
    sig=(df["perm_p"]<0.05).sum()
    print(f"  SUMMARY: mean balanced-acc {df['balanced_acc'].mean():.3f}; {sig}/{len(df)} participants p<0.05 (trial-permutation)")

for fn in (R2,R3,R4):
    try: fn()
    except Exception as ex: print(f"\n[{fn.__name__} ERROR] {type(ex).__name__}: {ex}")
print("\nSAVED:",OUT); print("="*80)
