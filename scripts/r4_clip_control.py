#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP-CONTROLLED AFFECT ANALYSIS (reviewer concern: is the "emotion condition" effect
really film-clip / sensory-content confound?).

Design (all WITHIN Session 1 -> removes the session confound; trial-disjoint -> no
window-overlap leakage):
  For each subject s and each enrol-emotion e that has >=2 clips (trials):
     template_s(e) = L2-mean PSD prototype from ONE enrol clip of emotion e
     Probe conditions (scored vs template_s(e), genuine=s, impostor=other subjects):
        (B) SAME emotion e, DIFFERENT clip   -> identity match, clip changed, affect same
        (C) DIFFERENT emotion (other clips)  -> identity match, clip changed, affect changed
  Key test:  EER(C) vs EER(B), paired across participants (Wilcoxon).
     EER(C) > EER(B)  => the emotion-condition effect survives holding identity fixed and
                         changing only affect, i.e. it is NOT purely clip/sensory content.
     EER(C) ~ EER(B)  => the effect is clip/sensory, not affect-specific.

Matcher = the paper's PSD band-power + cosine (step3): Welch nperseg=200, 5 bands, log,
62*5=310-D, L2 cosine, EER via ROC.  Read-only on data; writes results to outputs/run_clipctrl/.
USAGE:
  cd /lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
  source fignn_env/bin/activate 2>/dev/null || source p4_seedv_env/bin/activate
  python -u r4_clip_control.py 2>&1 | tee clipctrl_log.txt
"""
import os, glob, numpy as np, pandas as pd
from scipy.signal import welch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve, roc_auc_score

RNG = np.random.default_rng(0)
ROOT = next((c for c in ["/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
                         "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC", os.getcwd()]
             if os.path.isdir(c)), os.getcwd())
DATA = os.path.join(ROOT, "data", "processed", "sessionwise")
OUT  = os.path.join(ROOT, "outputs", "run_clipctrl"); os.makedirs(OUT, exist_ok=True)
CAP  = 200          # max windows per (subject,trial) for speed
BANDS = [(0.5,4),(4,8),(8,13),(13,30),(30,45)]
EMO = {0:"disgust",1:"fear",2:"sad",3:"neutral",4:"happy"}
print("="*76); print("CLIP-CONTROLLED AFFECT ANALYSIS | root:", ROOT); print("="*76)

# ---- load Session 1 with trial (=clip) + emotion labels ----
def load(n):
    f = sorted(glob.glob(os.path.join(DATA, f"*SESSION{n}_16sub.npz")), key=len)
    z = np.load(f[0], allow_pickle=True)
    return (z["X"].astype(np.float32), z["y_subject"].astype(int),
            z["y_trial"].astype(int), z["y_emotion"].astype(int))
X1, ysub, ytr, yemo = load(1)
SUBS = sorted(np.unique(ysub).tolist())
print(f"Session 1: X={X1.shape}  subjects={len(SUBS)}  trials={sorted(np.unique(ytr))}")

# ================= STEP 1: METADATA / CLIP-STRUCTURE AUDIT =================
print("\n--- STEP 1: clip structure (trial -> emotion), Session 1 ---")
audit = []
for t in sorted(np.unique(ytr)):
    es = np.unique(yemo[ytr == t])
    audit.append((int(t), [EMO.get(int(e), int(e)) for e in es]))
    print(f"  trial {t:2d}: emotion(s) {[EMO.get(int(e),int(e)) for e in es]}")
# clips per emotion (a 'clip' = a trial id here)
clips_per_emo = {int(e): sorted(np.unique(ytr[yemo == e]).tolist()) for e in np.unique(yemo)}
print("\nClips (trials) per emotion:")
for e, cl in clips_per_emo.items():
    print(f"  {EMO.get(e,e):8s}: {len(cl)} clip(s) -> trials {cl}")
usable = [e for e, cl in clips_per_emo.items() if len(cl) >= 2]
print(f"\nEmotions with >=2 clips (usable for clip control): "
      f"{[EMO.get(e,e) for e in usable]}")
pd.DataFrame([(t, ','.join(map(str,es))) for t,es in audit],
             columns=["trial","emotions"]).to_csv(os.path.join(OUT,"clip_audit.csv"), index=False)
if not usable:
    print("\n[STOP] No emotion has >=2 clips within a session; clip-controlled test not "
          "possible with this split. Paste the STEP-1 audit above and we redesign.")
    raise SystemExit(0)

# ================= matcher (paper PSD+cosine) =================
def psd(X):
    f, P = welch(X, fs=200, nperseg=200, axis=-1)
    out = [np.log(P[:, :, (f>=lo)&(f<hi)].sum(-1) + 1e-12) for lo, hi in BANDS]
    return np.concatenate(out, axis=1).astype(np.float32)   # (N, 310)
def l2(F): return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
def cap_idx(idx):
    idx = np.asarray(idx)
    return RNG.choice(idx, CAP, replace=False) if len(idx) > CAP else idx
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i]+fnr[i])/2), float(roc_auc_score(y, s))

# precompute PSD once
F1 = psd(X1)

# ================= STEP 2: within-session clip-controlled EER =================
# per (condition, subject) score pools
rows = []   # (condition, subject, y_true, score)
for e in usable:
    clips = clips_per_emo[e]
    enrol_clip = clips[0]; same_e_other = clips[1:]
    for s in SUBS:
        en = np.where((ysub==s)&(yemo==e)&(ytr==enrol_clip))[0]
        if len(en) < 5: continue
        en = cap_idx(en)
        templ = l2(l2(F1[en]).mean(0)[None,:])[0]           # subject template for (s,e,enrol clip)
        # genuine probes
        gB = np.where((ysub==s)&(yemo==e)&(np.isin(ytr,same_e_other)))[0]   # same emo, diff clip
        gC = np.where((ysub==s)&(yemo!=e))[0]                               # diff emotion
        # impostor probes (other subjects, matched condition)
        iB = np.where((ysub!=s)&(yemo==e)&(np.isin(ytr,same_e_other)))[0]
        iC = np.where((ysub!=s)&(yemo!=e))[0]
        for cond, gi, ii in [("B_sameEmo_diffClip", gB, iB), ("C_diffEmo", gC, iC)]:
            if len(gi) < 5 or len(ii) < 20: continue
            gi = cap_idx(gi); ii = cap_idx(ii)
            sg = l2(F1[gi]) @ templ
            si = l2(F1[ii]) @ templ
            for v in sg: rows.append((cond, s, 1, float(v)))
            for v in si: rows.append((cond, s, 0, float(v)))

df = pd.DataFrame(rows, columns=["condition","subject","y_true","score"])
df.to_csv(os.path.join(OUT,"clipctrl_scores.csv"), index=False)

# pooled EER per condition
print("\n--- STEP 2: pooled EER (within-session, trial-disjoint) ---")
pooled = {}
for c in ["B_sameEmo_diffClip","C_diffEmo"]:
    d = df[df.condition==c]
    e_, a_ = eer(d.y_true.values, d.score.values)
    pooled[c] = (e_, a_, len(d))
    print(f"  {c:22s}: EER={e_:.4f}  AUC={a_:.4f}  (n_scores={len(d)})")

# per-subject EER for paired test
per = []
for s in SUBS:
    row = {"subject": s}
    ok = True
    for c in ["B_sameEmo_diffClip","C_diffEmo"]:
        d = df[(df.condition==c)&(df.subject==s)]
        if d.y_true.nunique() < 2: ok=False; break
        row[c] = eer(d.y_true.values, d.score.values)[0]
    if ok: per.append(row)
perdf = pd.DataFrame(per); perdf.to_csv(os.path.join(OUT,"clipctrl_persubject.csv"), index=False)
print(f"\nPer-subject EER available for {len(perdf)} subjects")
if len(perdf) >= 5:
    b = perdf["B_sameEmo_diffClip"].values; c = perdf["C_diffEmo"].values
    W, p = wilcoxon(c, b)
    print(f"  mean EER  same-emo/diff-clip (B) = {b.mean():.4f} ± {b.std(ddof=1):.4f}")
    print(f"  mean EER  diff-emotion       (C) = {c.mean():.4f} ± {c.std(ddof=1):.4f}")
    print(f"  paired Wilcoxon  C vs B: W={W:.1f}  p={p:.4g}  "
          f"(mean ΔEER = {(c-b).mean():+.4f})")
    print("\nINTERPRETATION:")
    if (c.mean() > b.mean()) and p < 0.05:
        print("  EER(diff-emotion) > EER(same-emotion/diff-clip), significant ->")
        print("  the emotion-condition effect SURVIVES clip control: it is not purely")
        print("  film-clip / sensory-content confound.")
    elif p >= 0.05:
        print("  No significant difference -> within this within-session, single-enrol-clip")
        print("  test the emotion effect is NOT separable from clip; report honestly.")
    else:
        print("  EER(same-emo/diff-clip) >= EER(diff-emotion): unexpected; inspect per-subject.")
print("\nSAVED:", OUT)
print("="*76)
