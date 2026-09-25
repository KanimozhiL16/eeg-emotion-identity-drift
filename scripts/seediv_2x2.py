#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
seediv_2x2.py  --  SEED-IV replication of the leakage-controlled dissociation
============================================================================
Second-emotional-dataset replication of the P4 differentiating experiment, using the
SAME matcher and SAME trial-disjoint protocol as the SEED-V analysis (r4rev_reanalysis.py
R2 + r4_clip_control.py + R4). Read-only on data; writes results next to the npz.

Matcher (identical to paper step3 / SEED-V):
  Welch nperseg=200, 5 bands [(0.5-4),(4-8),(8-13),(13-30),(30-45)], log band-power,
  62*5 = 310-D descriptor, L2 normalise, cosine similarity, EER via ROC. CAP=200 win/pool.

Analyses:
  A  Data + clip-structure audit (trial -> emotion, clips per emotion, trial-disjointness).
  B  2x2  emotion-mismatch x session-mismatch, FULLY TRIAL-DISJOINT in every cell,
        matched budgets -> per-subject EER in 4 cells + Wilcoxon vs baseline.
  C  CLIP CONTROL (headline): within Session 1, identity fixed, template from ONE enrol clip;
        (B) same-emotion different-clip  vs  (C) different-emotion. Paired Wilcoxon.
        C>B (sig) => affect effect survives clip control (not pure film-clip confound).
  D  Manipulation check: LDA leave-one-trial-out balanced accuracy + trial-level
        label-permutation null (can PSD read emotion at all, without window leakage?).
  E  Participant-aware mixed model on the 2x2 cell EERs:
        EER ~ sess_mismatch * emo_mismatch, random intercept per subject (if statsmodels).

INPUT: seediv_session{1,2,3}.npz built by 09D_v2 (must contain y_emotion + y_trial).
RUN (Windows PowerShell, laptop):
    python "seediv_2x2.py"
Then paste the FULL console output back.
"""
import os, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.signal import welch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

RNG = np.random.default_rng(0)
DATA = r"C:\Users\L.KANIMOZHI\Downloads\P4_SEEDIV_LOCAL"   # LOCAL (non-OneDrive) folder written by 09D_v2
OUT  = os.path.join(DATA, "run_seediv_2x2"); os.makedirs(OUT, exist_ok=True)
BANDS = [(0.5,4),(4,8),(8,13),(13,30),(30,45)]
EMO = {0:"neutral", 1:"sad", 2:"fear", 3:"happy"}   # SEED-IV
CAP = 200
print("="*80); print("SEED-IV 2x2 / CLIP-CONTROL / MANIP-CHECK | data:", DATA); print("="*80)

def load(n):
    f = os.path.join(DATA, f"seediv_session{n}.npz")
    z = np.load(f, allow_pickle=True)
    keys = list(z.files)
    for req in ("X","y_subject","y_session","y_emotion","y_trial"):
        if req not in keys:
            raise SystemExit(f"[STOP] {f} lacks key '{req}'. Re-run 09D_v2 (keys found: {keys}).")
    return (z["X"].astype(np.float32), z["y_subject"].astype(int),
            z["y_trial"].astype(int), z["y_emotion"].astype(int))

def psd(X):
    f, P = welch(X, fs=200, nperseg=200, axis=-1)
    out = [np.log(P[:, :, (f>=lo)&(f<hi)].sum(-1) + 1e-12) for lo, hi in BANDS]
    return np.concatenate(out, axis=1).astype(np.float32)          # (N, 310)
def l2(F): return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i]+fnr[i])/2), float(roc_auc_score(y, s))
def capidx(idx):
    idx = np.asarray(idx)
    return RNG.choice(idx, CAP, replace=False) if len(idx) > CAP else idx

# ---- load all sessions ----
X1,s1,t1,e1 = load(1); X2,s2,t2,e2 = load(2); X3,s3,t3,e3 = load(3)
F1,F2,F3 = psd(X1), psd(X2), psd(X3)
SUBS = sorted(np.unique(s1).tolist())
emos = sorted(np.unique(e1).tolist())

# =================================================================== A  AUDIT
print("\n"+"#"*80); print("# A  DATA + CLIP-STRUCTURE AUDIT"); print("#"*80)
print(f"subjects (S1) = {SUBS}  (n={len(SUBS)})")
for n,(sx,tx,ex,Fx) in [(1,(s1,t1,e1,F1)),(2,(s2,t2,e2,F2)),(3,(s3,t3,e3,F3))]:
    print(f"session {n}: windows={len(sx)}  trials={sorted(np.unique(tx).tolist())}")
    for em in sorted(np.unique(ex).tolist()):
        cl = sorted(np.unique(tx[ex==em]).tolist())
        print(f"    emotion {em} ({EMO.get(em,em):8s}): {len(cl)} clips -> trials {cl}")
# trial-disjointness across sessions
cross = set(np.unique(t1)) & set(np.unique(t2)) & set(np.unique(t3))
print(f"trial ids shared across all 3 sessions (session-LOCAL ids expected): {sorted(cross)}")
print("  -> cross-session cells enrol S1 / verify S2+S3 = different DAYS, so disjoint by design.")

# =================================================================== B  2x2
print("\n"+"#"*80); print("# B  2x2  emotion-mismatch x session-mismatch (TRIAL-DISJOINT, matched budgets)"); print("#"*80)
rows=[]
for s in SUBS:
    for e0 in emos:
        tr = sorted(np.unique(t1[(s1==s)&(e1==e0)]).tolist())
        if len(tr) < 2: continue
        RNG.shuffle(tr); en_tr={tr[0]}
        en = np.where((s1==s)&(e1==e0)&(np.isin(t1,list(en_tr))))[0]
        if len(en) < 5: continue
        en = capidx(en); templ = l2(l2(F1[en]).mean(0)[None,:])[0]
        def cell(Fp,sp,ep,tp,cross_sess,diff_emo):
            if not diff_emo:
                gi = np.where((sp==s)&(ep==e0)&(np.ones(len(sp),bool) if cross_sess
                                                else ~np.isin(tp,list(en_tr))))[0]
                ii = np.where((sp!=s)&(ep==e0))[0]
            else:
                gi = np.where((sp==s)&(ep!=e0))[0]; ii = np.where((sp!=s)&(ep!=e0))[0]
            if len(gi)<5 or len(ii)<20: return None
            gi=capidx(gi); ii=capidx(ii)
            sg=l2(Fp[gi])@templ; si=l2(Fp[ii])@templ
            y=np.r_[np.ones(len(sg)),np.zeros(len(si))]; sc=np.r_[sg,si]
            return eer(y,sc)[0]
        base = cell(F1,s1,e1,t1,False,False)                                    # same sess, same emo, diff clip
        eOnly= cell(F1,s1,e1,t1,False,True)                                     # same sess, diff emo
        sOnly= cell(np.vstack([F2,F3]),np.r_[s2,s3],np.r_[e2,e3],np.r_[t2,t3],True,False)  # cross sess, same emo
        both = cell(np.vstack([F2,F3]),np.r_[s2,s3],np.r_[e2,e3],np.r_[t2,t3],True,True)   # cross sess, diff emo
        if None not in (base,eOnly,sOnly,both):
            rows.append(dict(subject=s,enrol_emo=EMO[e0],baseline=base,emo_only=eOnly,
                             sess_only=sOnly,both=both))
df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,"B_2x2_percell.csv"),index=False)
g=df.groupby("subject")[["baseline","emo_only","sess_only","both"]].mean().reset_index()
g.to_csv(os.path.join(OUT,"B_2x2_persubject.csv"),index=False)
print(f"per-subject cells available (n={len(g)}):")
for c in ["baseline","emo_only","sess_only","both"]:
    print(f"  {c:10s} mean EER {g[c].mean():.4f} +/- {g[c].std(ddof=1):.4f}")
if len(g)>=5:
    wE=wilcoxon(g["emo_only"],g["baseline"]); wS=wilcoxon(g["sess_only"],g["baseline"])
    wB=wilcoxon(g["both"],g["baseline"])
    print(f"  EMOTION mismatch  dEER={ (g['emo_only']-g['baseline']).mean():+.4f}  Wilcoxon p={wE.pvalue:.4g}")
    print(f"  SESSION mismatch  dEER={ (g['sess_only']-g['baseline']).mean():+.4f}  Wilcoxon p={wS.pvalue:.4g}")
    print(f"  BOTH     mismatch dEER={ (g['both']-g['baseline']).mean():+.4f}  Wilcoxon p={wB.pvalue:.4g}")
    print("  (SEED-V window-level reference: +0.047 emotion / +0.077 session; emotion marginal p~0.086)")

# =================================================================== C  CLIP CONTROL (headline)
print("\n"+"#"*80); print("# C  CLIP CONTROL (within S1, identity fixed): same-emo/diff-clip vs diff-emotion"); print("#"*80)
clips_per_emo={int(e):sorted(np.unique(t1[e1==e]).tolist()) for e in emos}
usable=[e for e,cl in clips_per_emo.items() if len(cl)>=2]
print("emotions with >=2 clips:", [EMO[e] for e in usable])
crows=[]
for e in usable:
    clips=clips_per_emo[e]; enrol_clip=clips[0]; same_e_other=clips[1:]
    for s in SUBS:
        en=np.where((s1==s)&(e1==e)&(t1==enrol_clip))[0]
        if len(en)<5: continue
        en=capidx(en); templ=l2(l2(F1[en]).mean(0)[None,:])[0]
        gB=np.where((s1==s)&(e1==e)&(np.isin(t1,same_e_other)))[0]
        gC=np.where((s1==s)&(e1!=e))[0]
        iB=np.where((s1!=s)&(e1==e)&(np.isin(t1,same_e_other)))[0]
        iC=np.where((s1!=s)&(e1!=e))[0]
        for cond,gi,ii in [("B_sameEmo_diffClip",gB,iB),("C_diffEmo",gC,iC)]:
            if len(gi)<5 or len(ii)<20: continue
            gi=capidx(gi); ii=capidx(ii)
            for v in l2(F1[gi])@templ: crows.append((cond,s,1,float(v)))
            for v in l2(F1[ii])@templ: crows.append((cond,s,0,float(v)))
cdf=pd.DataFrame(crows,columns=["condition","subject","y_true","score"])
cdf.to_csv(os.path.join(OUT,"C_clipctrl_scores.csv"),index=False)
for c in ["B_sameEmo_diffClip","C_diffEmo"]:
    d=cdf[cdf.condition==c]
    if len(d): e_,a_=eer(d.y_true.values,d.score.values); print(f"  pooled {c:20s}: EER={e_:.4f} AUC={a_:.4f} (n={len(d)})")
per=[]
for s in SUBS:
    r={"subject":s}; ok=True
    for c in ["B_sameEmo_diffClip","C_diffEmo"]:
        d=cdf[(cdf.condition==c)&(cdf.subject==s)]
        if d.y_true.nunique()<2: ok=False; break
        r[c]=eer(d.y_true.values,d.score.values)[0]
    if ok: per.append(r)
pdf=pd.DataFrame(per); pdf.to_csv(os.path.join(OUT,"C_clipctrl_persubject.csv"),index=False)
if len(pdf)>=5:
    b=pdf["B_sameEmo_diffClip"].values; c=pdf["C_diffEmo"].values
    W,p=wilcoxon(c,b)
    print(f"  same-emo/diff-clip (B) mean EER = {b.mean():.4f} +/- {b.std(ddof=1):.4f}")
    print(f"  diff-emotion       (C) mean EER = {c.mean():.4f} +/- {c.std(ddof=1):.4f}")
    print(f"  paired Wilcoxon C vs B: W={W:.1f} p={p:.4g}  (mean dEER={(c-b).mean():+.4f})")
    if c.mean()>b.mean() and p<0.05:
        print("  => affect effect SURVIVES clip control (not purely film-clip/sensory).")
    elif p>=0.05:
        print("  => NOT separable from clip in this within-session single-enrol-clip test (report honestly).")

# =================================================================== D  MANIP CHECK
print("\n"+"#"*80); print("# D  MANIPULATION CHECK: LDA leave-one-trial-out balanced acc + trial-perm null"); print("#"*80)
mrows=[]
for s in SUBS:
    idx=np.where(s1==s)[0]; X=F1[idx]; y=e1[idx]; tr=t1[idx]; trs=np.unique(tr)
    if len(np.unique(y))<2 or len(trs)<5: continue
    preds=np.zeros(len(y),int)
    for hd in trs:
        te=tr==hd; trn=~te
        if len(np.unique(y[trn]))<2: preds[te]=y[trn][0] if trn.any() else 0; continue
        preds[te]=LinearDiscriminantAnalysis().fit(X[trn],y[trn]).predict(X[te])
    bacc=balanced_accuracy_score(y,preds)
    tr2emo={int(tt):int(y[tr==tt][0]) for tt in trs}
    obs=bacc; nperm=100; ge=0
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
    mrows.append(dict(subject=s,balanced_acc=round(bacc,4),perm_p=round(pval,4)))
    print(f"  S{s:02d}: balanced-acc={bacc:.3f}  trial-perm p={pval:.3f}")
mdf=pd.DataFrame(mrows); mdf.to_csv(os.path.join(OUT,"D_manipcheck.csv"),index=False)
if len(mdf):
    sig=(mdf["perm_p"]<0.05).sum()
    print(f"  SUMMARY: mean balanced-acc {mdf['balanced_acc'].mean():.3f} "
          f"(chance={1/len(emos):.3f}); {sig}/{len(mdf)} subjects p<0.05")

# =================================================================== E  MIXED MODEL
print("\n"+"#"*80); print("# E  PARTICIPANT-AWARE MIXED MODEL on 2x2 cell EERs"); print("#"*80)
try:
    import statsmodels.formula.api as smf
    long=[]
    for _,r in df.iterrows():
        long.append((r.subject,0,0,r.baseline))   # sess0 emo0
        long.append((r.subject,0,1,r.emo_only))    # sess0 emo1
        long.append((r.subject,1,0,r.sess_only))   # sess1 emo0
        long.append((r.subject,1,1,r.both))        # sess1 emo1
    L=pd.DataFrame(long,columns=["subject","sess_mismatch","emo_mismatch","EER"])
    L.to_csv(os.path.join(OUT,"E_mixedmodel_long.csv"),index=False)
    m=smf.mixedlm("EER ~ sess_mismatch * emo_mismatch",L,groups=L["subject"]).fit(reml=True)
    print(m.summary().tables[1])
    print("  (session_mismatch = cross-day; emo_mismatch = different elicited emotion)")
except Exception as ex:
    print("  [mixed model skipped]",type(ex).__name__,ex,"-- pip install statsmodels to enable")

print("\nSAVED CSVs in:",OUT); print("="*80)
print("PASTE THIS ENTIRE OUTPUT BACK for interpretation.")
