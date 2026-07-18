#!/usr/bin/env python3
r"""
manip_fig_only.py -- FAST figure+CSV generator for the M2 manipulation check.
Same decoding as manip_check_emotion_decoding.py (identical features, trial-safe GroupKFold),
but SKIPS the 200x permutation-null loop (that only gives a p-value; significance is already
established by the one-sample t-test, t(15)=5.96, p=2.6e-5). This produces the two Supplementary
figures + CSVs in ~2 min instead of ~30 min.

OUT (Brev): outputs/figs/manip_check_confusion.png, manip_check_persubject.png
            outputs/csv/manip_check_persubject.csv, manip_check_confusion.csv
RUN: cd ~/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC && source p4_seedv_env/bin/activate
     python -u scripts/manip_fig_only.py 2>&1 | tee outputs/manip_fig_only.log
"""
import os, glob, csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import welch
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix

def has(p): return os.path.isdir(os.path.join(p, "outputs"))
ROOT = next((c for c in [os.getcwd(),
      "/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if has(c)), os.getcwd())
DATA = os.path.join(ROOT, "data", "processed", "sessionwise")
FS, BANDS = 200, [(0.5,4),(4,8),(8,13),(13,30),(30,45)]

def bandpow(X):
    f, P = welch(X.astype(np.float64), fs=FS, nperseg=200, axis=-1)
    return np.log(np.concatenate([P[:, :, (f>=lo)&(f<hi)].sum(-1) for lo,hi in BANDS], 1)+1e-12).astype(np.float32)

Xs, ysub, yemo, ytr, yses = [], [], [], [], []
for fp in sorted(glob.glob(os.path.join(DATA, "*16sub.npz"))):
    d = np.load(fp, allow_pickle=True)
    if not all(k in d for k in ("X","y_subject","y_emotion","y_trial","y_session")): continue
    Xs.append(np.asarray(d["X"], np.float32)); ysub.append(np.asarray(d["y_subject"],int))
    yemo.append(np.asarray(d["y_emotion"],int)); ytr.append(np.asarray(d["y_trial"],int))
    yses.append(np.asarray(d["y_session"],int))
X=np.concatenate(Xs); ysub=np.concatenate(ysub); yemo=np.concatenate(yemo)
ytr=np.concatenate(ytr); yses=np.concatenate(yses)
F = bandpow(X); SUBS = sorted(np.unique(ysub).tolist())
print(f"[load] windows={len(F)} subjects={len(SUBS)}")

def decode_subject(s):
    m = ysub==s; Fi, yi = F[m], yemo[m].copy(); groups = yses[m]*100 + ytr[m]
    gk = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    yhat = np.zeros_like(yi)
    for tr_i, te_i in gk.split(Fi, yi, groups):
        clf = make_pipeline(StandardScaler(), LinearSVC(C=0.5, dual="auto", max_iter=5000))
        clf.fit(Fi[tr_i], yi[tr_i]); yhat[te_i] = clf.predict(Fi[te_i])
    return accuracy_score(yi, yhat), confusion_matrix(yi, yhat, labels=[0,1,2,3,4])

accs=[]; C=np.zeros((5,5))
for s in SUBS:
    a,cm = decode_subject(s); accs.append(a); C+=cm
    print(f"  subj {s:2d}: acc = {a*100:5.1f}%")
accs=np.array(accs)
t_stat,t_p = stats.ttest_1samp(accs,0.20)
try: w_stat,w_p = stats.wilcoxon(accs-0.20)
except ValueError: w_stat,w_p=float("nan"),float("nan")
d_eff=(accs.mean()-0.20)/accs.std(ddof=1); n_above=int(np.sum(accs>0.20))
print(f"\nMEAN {accs.mean()*100:.1f}% (SD {accs.std(ddof=1)*100:.1f}); n_above={n_above}/{len(SUBS)}; "
      f"t({len(SUBS)-1})={t_stat:.2f} p={t_p:.3e}; Wilcoxon W={w_stat:.1f} p={w_p:.3e}; d={d_eff:.2f}")

FIG=os.path.join(ROOT,"outputs","figs"); CSVD=os.path.join(ROOT,"outputs","csv")
os.makedirs(FIG,exist_ok=True); os.makedirs(CSVD,exist_ok=True)
EMO=["Disgust","Fear","Sad","Neutral","Happy"]; Cn=C/C.sum(1,keepdims=True)

fig,ax=plt.subplots(figsize=(4.2,3.6)); im=ax.imshow(Cn,cmap="Blues",vmin=0,vmax=1)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(EMO,rotation=45,ha="right",fontsize=8); ax.set_yticklabels(EMO,fontsize=8)
ax.set_xlabel("Predicted",fontsize=9); ax.set_ylabel("True",fontsize=9)
for i in range(5):
    for j in range(5):
        ax.text(j,i,f"{Cn[i,j]:.2f}",ha="center",va="center",fontsize=7,
                color="white" if Cn[i,j]>0.5 else "black")
fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04).ax.tick_params(labelsize=7)
ax.set_title(f"5-way emotion decoding (mean {accs.mean()*100:.1f}%, chance 20%)",fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG,"manip_check_confusion.png"),dpi=300); plt.close(fig)

fig,ax=plt.subplots(figsize=(5.2,3.0)); order=np.argsort(accs)[::-1]
ax.bar(range(len(accs)),accs[order]*100,color="#3b6fb0")
ax.axhline(20,ls="--",lw=1,color="crimson",label="chance (20%)")
ax.axhline(accs.mean()*100,ls="-",lw=1,color="black",label=f"mean ({accs.mean()*100:.1f}%)")
ax.set_xticks(range(len(accs))); ax.set_xticklabels([f"S{SUBS[i]}" for i in order],rotation=90,fontsize=6)
ax.set_ylabel("Emotion-decoding acc (%)",fontsize=9); ax.set_xlabel("Subject (sorted)",fontsize=9)
ax.legend(fontsize=7,frameon=False); fig.tight_layout()
fig.savefig(os.path.join(FIG,"manip_check_persubject.png"),dpi=300); plt.close(fig)

with open(os.path.join(CSVD,"manip_check_persubject.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["subject","acc","chance","above_chance"])
    for i,s in enumerate(SUBS): w.writerow([s,f"{accs[i]:.4f}",0.20,int(accs[i]>0.20)])
    w.writerow([]); w.writerow(["mean",f"{accs.mean():.4f}"]); w.writerow(["sd",f"{accs.std(ddof=1):.4f}"])
    w.writerow(["n_above_chance",n_above]); w.writerow(["one_sample_t",f"{t_stat:.4f}"])
    w.writerow(["one_sample_p",f"{t_p:.3e}"]); w.writerow(["wilcoxon_W",f"{w_stat:.4f}"])
    w.writerow(["wilcoxon_p",f"{w_p:.3e}"]); w.writerow(["cohens_d",f"{d_eff:.4f}"])
with open(os.path.join(CSVD,"manip_check_confusion.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["true\\pred"]+EMO)
    for i in range(5): w.writerow([EMO[i]]+[f"{Cn[i,j]:.4f}" for j in range(5)])

print(f"[figs] saved -> {FIG}/manip_check_confusion.png , manip_check_persubject.png")
print(f"[csv]  saved -> {CSVD}/manip_check_persubject.csv , manip_check_confusion.csv")
print("DONE (permutation-null skipped by design; significance from t-test above).")
