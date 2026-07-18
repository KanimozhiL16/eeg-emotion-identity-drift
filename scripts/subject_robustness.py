#!/usr/bin/env python3
"""
subject_robustness.py  -- Revision experiment R1: per-subject cross-session robustness.

Answers the reviewer request for subject-level analysis (hardest/easiest subjects, inter-subject
variance, drift distribution). Reuses run_07's OWN feature/scoring functions so the numbers are
consistent with the manuscript (no new pipeline).

Per subject, cross-session EER is computed for S1->S2 and S1->S3 (enrol on session 1).
Outputs -> outputs/run_17_subject_robustness/ : subject_robustness.csv, figures/fig_subject_eer.png
USAGE:  python scripts/subject_robustness.py
"""
import os, glob, importlib.util, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT="/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"
if not os.path.isdir(ROOT): ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
SRC=os.path.join(ROOT,"scripts","07_seedv_session_drift_subject_adaptation.py")
DATA=os.path.join(ROOT,"data","processed","sessionwise")
OUT=os.path.join(ROOT,"outputs","run_17_subject_robustness"); FIG=os.path.join(OUT,"figures"); os.makedirs(FIG,exist_ok=True)
print("="*74); print("R1  per-subject cross-session robustness (run_07 functions)"); print("="*74)

spec=importlib.util.spec_from_file_location("run07",SRC); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

# load sessionwise data by internal session id
sess={}
for f in sorted(glob.glob(os.path.join(DATA,"*.npz"))):
    d=np.load(f,allow_pickle=True)
    if not all(k in d for k in ("X","y_subject","y_session")): continue
    sid=int(np.unique(d["y_session"])[0])
    if sid in sess: continue
    X=np.asarray(d["X"],np.float32); y=np.asarray(d["y_subject"],int)
    try: X,y=m.downsample_per_subject(X,y,max_per_subject=900,seed=42)[:2]
    except Exception: pass
    sess[sid]=(m.simple_features(X), y)
SUBS=sorted(np.unique(sess[1][1]).tolist()); S2R={s:i for i,s in enumerate(SUBS)}
Fe,ye=sess[1]; P,owners=m.make_prototypes(Fe,ye); owners=np.asarray(owners)

def persub_eer(day):
    Fp,yp=sess[day]
    sdf=m.verification_scores(Fp,yp,P,owners)
    return {int(c): m.eer_auc(g["y_true"].values, g["score"].values)[0] for c,g in sdf.groupby("probe_subject")}

e2,e3=persub_eer(2),persub_eer(3)
rows=[{"subject":s,"EER_S1toS2":round(e2[s],4),"EER_S1toS3":round(e3[s],4),
       "mean_cross":round((e2[s]+e3[s])/2,4)} for s in SUBS]
df=pd.DataFrame(rows).sort_values("mean_cross")
df.to_csv(os.path.join(OUT,"subject_robustness.csv"),index=False)

mc=df["mean_cross"].values
print(df.to_string(index=False))
print(f"\nmean cross-session EER: {mc.mean():.4f} +/- {mc.std():.4f} (SD); range [{mc.min():.4f}, {mc.max():.4f}]")
print(f"easiest subject: S{int(df.iloc[0]['subject'])} (EER {df.iloc[0]['mean_cross']:.3f}) | "
      f"hardest: S{int(df.iloc[-1]['subject'])} (EER {df.iloc[-1]['mean_cross']:.3f})")
print(f"coefficient of variation: {mc.std()/mc.mean():.2f}")

# figure: sorted per-subject bars (S2, S3) + boxplot
fig,ax=plt.subplots(1,2,figsize=(10,4))
x=np.arange(len(df)); w=0.4
ax[0].bar(x-w/2,df["EER_S1toS2"],w,label="S1$\\to$S2")
ax[0].bar(x+w/2,df["EER_S1toS3"],w,label="S1$\\to$S3")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"S{int(s)}" for s in df["subject"]],rotation=90,fontsize=7)
ax[0].set_ylabel("cross-session EER"); ax[0].set_title("Per-subject cross-session EER (sorted)"); ax[0].legend(fontsize=8)
ax[1].boxplot([df["EER_S1toS2"],df["EER_S1toS3"],df["mean_cross"]],labels=["S1->S2","S1->S3","mean"])
ax[1].set_ylabel("EER"); ax[1].set_title("Inter-subject variability"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_subject_eer.png"),dpi=300); plt.close()
print("="*74); print("PASS -- saved outputs/run_17_subject_robustness/ (subject_robustness.csv, figures/fig_subject_eer.png)"); print("="*74)
