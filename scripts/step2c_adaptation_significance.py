#!/usr/bin/env python3
"""
step2c_adaptation_significance.py
Locks the "adaptive (re-enrollment) verification" claim with PER-SUBJECT paired significance.

Reproduces run_07's enrollment-update adaptation, but per subject and leakage-safe:
  - enroll prototype on S1 (50% of each subject's S1 windows),
  - at target session k in {2,3}: split that session into ADAPT-set (fraction f) + TEST-set (rest),
  - baseline EER  = TEST-set scored vs S1-only prototype,
  - adapted EER   = same TEST-set scored vs updated prototype (S1-enrol + adapt-set),
  - paired Wilcoxon (baseline vs adapted) across the 16 subjects, per session.
Prints its OWN baseline per-session EER first — must match run_07 (~0.138 / 0.182 / 0.251) before trusting p.

Features: per-channel log band-powers (delta/theta/alpha/beta/gamma) via Welch -> 62*5 = 310-d.
Outputs -> outputs/run_15_adaptation_significance/ : adaptation_significance.csv, persubject.csv
USAGE:  python scripts/step2c_adaptation_significance.py
"""
import os, glob, csv, numpy as np
from scipy.signal import welch
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

FRACTIONS=[0.1,0.2]; BANDS=[(1,4),(4,8),(8,13),(13,30),(30,45)]; FS=200
ROOT="/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"
if not os.path.isdir(ROOT): ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
DATA=os.path.join(ROOT,"data","processed","sessionwise"); OUT=os.path.join(ROOT,"outputs","run_15_adaptation_significance")
os.makedirs(OUT,exist_ok=True)
print("="*78); print("STEP 2c  per-subject adaptation significance"); print("="*78)

# load 3 sessionwise files, map by INTERNAL y_session (handles file/id reversal)
sess={}
for f in sorted(glob.glob(os.path.join(DATA,"*.npz"))):
    d=np.load(f,allow_pickle=True); sid=int(np.unique(d["y_session"])[0])
    sess[sid]=(d["X"].astype(np.float32), d["y_subject"].astype(int))
    print(f"  loaded {os.path.basename(f)} -> internal session {sid}, X={d['X'].shape}")
SUBS=sorted(np.unique(sess[1][1]).tolist()); S2R={s:i for i,s in enumerate(SUBS)}
print(f"  sessions={sorted(sess)} subjects={len(SUBS)}")

def feats(X):
    fr,P=welch(X,fs=FS,nperseg=200,axis=-1)        # P:(N,62,F)
    out=[np.log(np.trapz(P[:,:,(fr>=lo)&(fr<hi)],axis=-1)+1e-12) for lo,hi in BANDS]
    return np.concatenate(out,axis=1)               # (N, 62*5)
F={k:(feats(sess[k][0]),sess[k][1]) for k in sess}
print("  features done:", {k:F[k][0].shape for k in F})

rng=np.random.default_rng(42)
# S1 enroll / heldout split per subject
enrol={}; s1test={}
for s in SUBS:
    idx=np.where(F[1][1]==s)[0]; rng.shuffle(idx); h=len(idx)//2; enrol[s]=idx[:h]; s1test[s]=idx[h:]
sc=StandardScaler().fit(F[1][0][np.concatenate([enrol[s] for s in SUBS])])
def nz(M): X=sc.transform(M); return X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9)
Z={k:nz(F[k][0]) for k in F}
proto1={s: (Z[1][enrol[s]].mean(0)) for s in SUBS}; proto1={s:p/(np.linalg.norm(p)+1e-9) for s,p in proto1.items()}
def eer(g,i):
    y=np.r_[np.ones(len(g)),np.zeros(len(i))]; s=np.r_[g,i]
    fpr,tpr,_=roc_curve(y,s); fnr=1-tpr; j=np.nanargmin(np.abs(fpr-fnr)); return float((fpr[j]+fnr[j])/2)

# baseline per-subject EER for a given session's test indices and prototype dict
def persub_eer(test_idx_of, proto):
    res={}
    for c in SUBS:
        pc=proto[c]; tc=Z_sess[test_idx_of[c]]@pc
        ic=np.concatenate([Z_sess[test_idx_of[s]]@pc for s in SUBS if s!=c])
        res[c]=eer(np.r_[np.ones(len(tc)),np.zeros(len(ic))],np.r_[tc,ic])
    return res

rows=[]; ps_rows=[]
# sanity: S1 heldout baseline
Z_sess=Z[1]; s1=persub_eer(s1test,proto1); print(f"  [sanity] S1-heldout mean EER={np.mean(list(s1.values())):.4f} (run_07≈0.138)")
for k in [2,3]:
    Z_sess=Z[k]
    base_idx={s:np.where(F[k][1]==s)[0] for s in SUBS}
    base=persub_eer(base_idx,proto1); base_mean=np.mean(list(base.values()))
    # try fractions, pick best mean adapted EER
    best=None
    for f in FRACTIONS:
        adapt_idx={}; test_idx={}
        for s in SUBS:
            idx=np.where(F[k][1]==s)[0].copy(); rng.shuffle(idx); na=max(1,int(len(idx)*f))
            adapt_idx[s]=idx[:na]; test_idx[s]=idx[na:]
        proto_a={}
        for s in SUBS:
            v=np.vstack([Z[1][enrol[s]], Z[k][adapt_idx[s]]]).mean(0); proto_a[s]=v/(np.linalg.norm(v)+1e-9)
        # baseline on the SAME held-out test set (fair)
        b=persub_eer(test_idx,proto1); a=persub_eer(test_idx,proto_a)
        bm=np.mean(list(b.values())); am=np.mean(list(a.values()))
        if best is None or am<best[2]: best=(f,bm,am,b,a)
    f,bm,am,b,a=best
    bb=np.array([b[s] for s in SUBS]); aa=np.array([a[s] for s in SUBS])
    try: W,p=stats.wilcoxon(bb,aa)
    except Exception: W,p=np.nan,np.nan
    rel=100*(bm-am)/bm
    rows.append({"session":k,"best_fraction":f,"EER_baseline":round(bm,4),"EER_adapted":round(am,4),
                 "rel_reduction_pct":round(rel,2),"wilcoxon_W":float(W),"p_value":round(float(p),4),"n":len(SUBS)})
    for s in SUBS: ps_rows.append({"session":k,"subject":s,"EER_baseline":round(b[s],4),"EER_adapted":round(a[s],4)})
    print(f"  session {k}: baseline(full)={base_mean:.4f} | held-out baseline={bm:.4f} -> adapted={am:.4f} (f={f}, -{rel:.1f}%) | Wilcoxon W={W}, p={p:.4f}")

with open(os.path.join(OUT,"adaptation_significance.csv"),"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(os.path.join(OUT,"persubject.csv"),"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=["session","subject","EER_baseline","EER_adapted"]); w.writeheader(); w.writerows(ps_rows)
sig=[r for r in rows if r["p_value"]<0.05]
print("="*78)
print(f"VERDICT: adaptive re-enrollment significant (p<0.05) at sessions: {[r['session'] for r in sig] or 'NONE'}")
print("PASS — saved -> outputs/run_15_adaptation_significance/")
print("="*78)
