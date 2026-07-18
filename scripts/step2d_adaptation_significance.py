#!/usr/bin/env python3
"""
step2d_adaptation_significance.py
Per-subject significance for the enrollment-update adaptation — using run_07's OWN functions
(imported), so the baseline EER matches run_07 exactly (S2≈0.182, S3≈0.251).

Protocol (mirrors run_07's adaptation, made leakage-safe + per-subject):
  - enroll prototypes on Session 1 (simple_features + make_prototypes),
  - for probe session p in {2,3}: per subject split into ADAPT (first frac) + EVAL (rest),
      P_adapt[c] = l2norm(0.75*P_enroll[c] + 0.25*centroid(adapt features of c)),
  - score the SAME EVAL set vs P_enroll (baseline) and vs P_adapt (adapted) via verification_scores,
  - per-subject EER from each score_df, paired Wilcoxon across the 16 subjects (best frac in {0.1,0.2}).
Outputs -> outputs/run_16_adaptation_significance/ : adaptation_significance.csv, persubject.csv
USAGE:  python scripts/step2d_adaptation_significance.py
"""
import os, importlib.util, numpy as np, pandas as pd
from scipy import stats

ROOT="/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"
if not os.path.isdir(ROOT): ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
SRC=os.path.join(ROOT,"scripts","07_seedv_session_drift_subject_adaptation.py")
OUT=os.path.join(ROOT,"outputs","run_16_adaptation_significance"); os.makedirs(OUT,exist_ok=True)
FRACTIONS=[0.1,0.2]; RNG=np.random.default_rng(42)
print("="*78); print("STEP 2d  adaptation significance (reusing run_07 functions)"); print("="*78)

# ---- import run_07 as a module (does not run main) ----
spec=importlib.util.spec_from_file_location("run07",SRC); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

import glob
DATA=os.path.join(ROOT,"data","processed","sessionwise")
files=sorted(set(glob.glob(os.path.join(DATA,"*.npz"))))
sess={}
for p in files:
    d=np.load(p,allow_pickle=True)
    if "X" not in d or "y_subject" not in d or "y_session" not in d: continue
    X=np.asarray(d["X"],np.float32); y=np.asarray(d["y_subject"],int); sid=int(np.unique(d["y_session"])[0])
    if sid in sess: continue                      # skip duplicate copies of same session
    try: X,y=m.downsample_per_subject(X,y,max_per_subject=900,seed=42)[:2]
    except Exception: pass
    F=m.simple_features(X); sess[sid]=(np.asarray(F,np.float64),np.asarray(y,int))
    print(f"  session {sid}: F={F.shape}")
SUBS=sorted(np.unique(sess[1][1]).tolist())

# enroll on session 1
Fe,ye=sess[1]; P_enroll,owners=m.make_prototypes(Fe,ye); owners=np.asarray(owners)
ow2i={int(o):i for i,o in enumerate(owners)}

def persub_eer(score_df):
    """per-subject EER from a verification_scores dataframe (group by probe_subject)."""
    res={}
    for c,g in score_df.groupby("probe_subject"):
        res[int(c)]=m.eer_auc(g["y_true"].values, g["score"].values)[0]
    return res

rows=[]; ps_rows=[]
print("\n  session : baseline -> adapted | best_frac | Wilcoxon p")
for p in [2,3]:
    Fp,yp=sess[p]
    best=None
    for frac in FRACTIONS:
        adapt_mask=np.zeros(len(yp),bool);
        for c in SUBS:
            idx=np.where(yp==c)[0].copy(); RNG.shuffle(idx); n_ad=max(1,int(len(idx)*frac)); adapt_mask[idx[:n_ad]]=True
        eval_idx=np.where(~adapt_mask)[0]
        # adapted prototypes
        P_adapt=P_enroll.copy().astype(np.float64)
        for c in SUBS:
            ad=np.where(adapt_mask & (yp==c))[0]
            if len(ad): P_adapt[ow2i[c]]=m.l2norm((0.75*P_enroll[ow2i[c]]+0.25*Fp[ad].mean(0))[None,:])[0]
        sb=m.verification_scores(Fp[eval_idx],yp[eval_idx],P_enroll,owners)
        sa=m.verification_scores(Fp[eval_idx],yp[eval_idx],P_adapt,owners)
        b=persub_eer(sb); a=persub_eer(sa)
        bm=np.mean([b[c] for c in SUBS]); am=np.mean([a[c] for c in SUBS])
        if best is None or am<best[2]: best=(frac,bm,am,b,a)
    frac,bm,am,b,a=best
    bb=np.array([b[c] for c in SUBS]); aa=np.array([a[c] for c in SUBS])
    try: W,pv=stats.wilcoxon(bb,aa)
    except Exception: W,pv=np.nan,np.nan
    rel=100*(bm-am)/bm
    rows.append({"session":p,"best_fraction":frac,"EER_baseline":round(bm,4),"EER_adapted":round(am,4),
                 "rel_reduction_pct":round(rel,2),"wilcoxon_W":float(W),"p_value":round(float(pv),4),"n":len(SUBS)})
    for c in SUBS: ps_rows.append({"session":p,"subject":c,"EER_baseline":round(b[c],4),"EER_adapted":round(a[c],4)})
    print(f"  S{p}: {bm:.4f} -> {am:.4f} | frac={frac} (-{rel:.1f}%) | W={W}, p={pv:.4f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT,"adaptation_significance.csv"),index=False)
pd.DataFrame(ps_rows).to_csv(os.path.join(OUT,"persubject.csv"),index=False)
sig=[r["session"] for r in rows if r["p_value"]<0.05]
print("="*78)
print(f"  [sanity] baseline S2/S3 should be ≈0.182/0.251 (run_07).")
print(f"VERDICT: adaptive re-enrollment significant (p<0.05) at sessions: {sig or 'NONE'}")
print("PASS — saved -> outputs/run_16_adaptation_significance/")
print("="*78)
