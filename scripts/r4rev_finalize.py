#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINALISE the trial-disjoint reanalysis so it can be made PRIMARY everywhere.
Reads the CSVs already written by r4rev_reanalysis.py (no EEG reprocessing):
  outputs/run_reanalysis/R2_2x2_persubject.csv   (subject, baseline, emo_only, sess_only, both)
  outputs/run_reanalysis/R3_theta_conditions.csv (50 conditions: theta, alpha, beta, gamma, EER, trans)
Produces:
  1) participant-aware MixedLM (EER ~ S*E + 1|subject) + FDR on the 3 paired contrasts, on the TRIAL-DISJOINT 2x2.
  2) theta transition table: beta + 95% CI for unadjusted / +transition / within-transition / S1->S2 / S1->S3.
USAGE:
  cd /lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
  source fignn_env/bin/activate 2>/dev/null || source p4_seedv_env/bin/activate
  python -u r4rev_finalize.py 2>&1 | tee finalize_log.txt
"""
import os, glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
OUT=None
for c in ["/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC","/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",os.getcwd()]:
    if os.path.isdir(os.path.join(c,"outputs","run_reanalysis")): OUT=os.path.join(c,"outputs","run_reanalysis"); break
print("="*78); print("FINALISE reanalysis | dir:",OUT); print("="*78)

# ---------- 1) MixedLM + FDR on trial-disjoint 2x2 ----------
print("\n# 1) TRIAL-DISJOINT 2x2: participant-aware mixed model + FDR")
p=pd.read_csv(os.path.join(OUT,"R2_2x2_persubject.csv"))
long=[]
for _,r in p.iterrows():
    long+=[(r.subject,0,0,r.baseline),(r.subject,0,1,r.emo_only),(r.subject,1,0,r.sess_only),(r.subject,1,1,r["both"])]
d=pd.DataFrame(long,columns=["subject","S","E","EER"])
m=smf.mixedlm("EER ~ S*E",d,groups=d["subject"]).fit()
print(f"  MixedLM  session beta={m.params['S']:+.4f} p={m.pvalues['S']:.3g} | emotion beta={m.params['E']:+.4f} p={m.pvalues['E']:.3g} | S:E beta={m.params['S:E']:+.4f} p={m.pvalues['S:E']:.3g}")
# paired Wilcoxon contrasts + FDR
pv=[wilcoxon(p.emo_only,p.baseline).pvalue, wilcoxon(p.sess_only,p.baseline).pvalue, wilcoxon(p["both"],p.baseline).pvalue]
rej,q,_,_=multipletests(pv,method="fdr_bh")
for nm,pp,qq in zip(["emotion","session","both"],pv,q):
    print(f"  contrast {nm:8s}: p={pp:.4g}  q(FDR)={qq:.4g}")
print(f"  cells: baseline {p.baseline.mean():.4f}, +emo {p.emo_only.mean():.4f}, +sess {p.sess_only.mean():.4f}, both {p['both'].mean():.4f}")
print(f"  deltas: emotion {(p.emo_only-p.baseline).mean():+.4f}, session {(p.sess_only-p.baseline).mean():+.4f}, both {(p['both']-p.baseline).mean():+.4f}")

# ---------- 2) theta transition table with 95% CI ----------
print("\n# 2) THETA transition table (beta + 95% CI)")
d3=pd.read_csv(os.path.join(OUT,"R3_theta_conditions.csv"))
tcol="trans" if "trans" in d3 else [c for c in d3 if "trans" in c.lower() or "session" in c.lower()][0]
d3=d3.rename(columns={tcol:"trans"}); d3["trans"]=d3["trans"].astype(str)
def row(name,formula,dat,term):
    fit=smf.ols(formula,data=dat).fit(); b=fit.params[term]; ci=fit.conf_int().loc[term]
    print(f"  {name:22s} beta={b:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  p={fit.pvalues[term]:.3g}  n={int(fit.nobs)}")
bands=" + ".join([b for b in ['alpha','beta','gamma'] if b in d3])
row("unadjusted", "EER ~ theta"+(" + "+bands if bands else ""), d3, "theta")
row("+ transition", "EER ~ theta + C(trans)"+(" + "+bands if bands else ""), d3, "theta")
d3["theta_c"]=d3.groupby("trans")["theta"].transform(lambda x:x-x.mean())
d3["EER_c"]=d3.groupby("trans")["EER"].transform(lambda x:x-x.mean())
row("within-transition", "EER_c ~ theta_c", d3, "theta_c")
for tv in sorted(d3["trans"].unique()):
    sub=d3[d3["trans"]==tv]
    if len(sub)>=6: row(f"only {tv[-14:-8] if len(tv)>14 else tv}", "EER ~ theta", sub, "theta")
print("\nDONE. Paste this whole block; these become the PRIMARY mixed-model + theta-table numbers.")
print("="*78)
