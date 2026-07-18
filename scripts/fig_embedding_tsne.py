#!/usr/bin/env python3
"""
fig_embedding_tsne.py  --  Identity + per-subject session drift (t-SNE), CCIEP Fig.2B analogue.

FIX over v1: a GLOBAL t-SNE is dominated by between-subject variance, so colouring it
by session shows sessions intermixed (looks like NO drift). Session drift is a
WITHIN-subject effect, so we show it PER SUBJECT.

LAYOUT (3 panels):
  (a) global t-SNE coloured by SUBJECT      -> identity clusters (separable)
  (b) per-subject t-SNE, most STABLE subject, coloured by session   -> sessions overlap
  (c) per-subject t-SNE, most DRIFT-PRONE subject, coloured by session -> sessions separate
The stable/drift subjects are auto-selected from cross-session centroid separation,
matching the paper's finding that drift is UNEVEN across individuals (CV 0.61).

INPUT (on Brev): data/processed/sessionwise/*.npz  keys: X[N,C,T], y_subject, y_session
RUN:  cd /home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
      python -W ignore scripts/fig_embedding_tsne.py 2>&1 | tee outputs/figs/embedding_tsne.log
OUTPUT: outputs/figs/Fig_embedding_tsne.png (600 dpi)
Uses only data you already have (band-power/time features). No retraining, no fabricated numbers.
"""
import os, glob, sys, traceback, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

FS = 200
BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,14),"beta":(14,30),"gamma":(30,45)}
DATA = "data/processed/sessionwise"
OUT  = "outputs/figs"; os.makedirs(OUT, exist_ok=True)
MAX_PER_SUBJECT_SESSION = 120     # cap per subject per session (readability + speed)
MAX_GLOBAL_POINTS       = 6000    # hard cap for the global panel
RNG = np.random.default_rng(0)

def features(X):
    f, P = welch(X, fs=FS, nperseg=min(256, X.shape[-1]), axis=-1)
    bp = np.stack([P[:, :, (f>=lo)&(f<=hi)].sum(-1) for lo,hi in BANDS.values()], -1)  # [N,C,5]
    ts = np.stack([X.mean(-1), X.std(-1), np.sqrt((X**2).mean(-1))], -1)                # [N,C,3]
    return np.concatenate([bp, ts], -1).reshape(X.shape[0], -1)

def tsne2(Z, perp):
    perp = max(5, min(perp, (len(Z)-1)//3))
    return TSNE(n_components=2, perplexity=perp, init="pca", random_state=0).fit_transform(Z)

try:
    # ---- load features per (subject, session) ----
    F, ysub, yses = [], [], []
    for fpath in sorted(glob.glob(os.path.join(DATA, "*.npz"))):
        d = np.load(fpath, allow_pickle=True)
        if not all(k in d for k in ("X","y_subject","y_session")): continue
        X = np.asarray(d["X"], np.float32); ys = np.asarray(d["y_subject"], int)
        sid = int(np.unique(d["y_session"])[0])
        keep = []
        for s in np.unique(ys):
            idx = np.where(ys == s)[0]
            keep.append(RNG.choice(idx, min(MAX_PER_SUBJECT_SESSION, len(idx)), replace=False))
        keep = np.concatenate(keep)
        F.append(features(X[keep])); ysub.append(ys[keep]); yses.append(np.full(len(keep), sid))
    Zf = np.concatenate(F); ysub = np.concatenate(ysub); yses = np.concatenate(yses)
    Zf = StandardScaler().fit_transform(Zf)
    subjects = sorted(np.unique(ysub).tolist()); sessions = sorted(np.unique(yses).tolist())
    print(f"[load] points={len(Zf)} subjects={len(subjects)} sessions={sessions}")

    # ---- auto-pick most-stable & most-drift-prone subject ----
    # drift score = mean pairwise session-centroid distance / mean within-session spread
    def drift_score(s):
        m = ysub == s; sc = []
        cents = []
        for sid in sessions:
            sel = m & (yses == sid)
            if sel.sum() < 5: return np.nan
            cents.append(Zf[sel].mean(0)); sc.append(np.linalg.norm(Zf[sel]-Zf[sel].mean(0), axis=1).mean())
        cents = np.array(cents)
        between = np.mean([np.linalg.norm(cents[i]-cents[j]) for i in range(len(cents)) for j in range(i+1,len(cents))])
        return between / (np.mean(sc)+1e-9)
    ds = {s: drift_score(s) for s in subjects}
    ds = {s:v for s,v in ds.items() if np.isfinite(v)}
    stable = min(ds, key=ds.get); drifty = max(ds, key=ds.get)
    print(f"[pick] stable subject={stable} (score={ds[stable]:.3f}) | drift-prone={drifty} (score={ds[drifty]:.3f})")

    # ---- global t-SNE (subsample) for identity panel ----
    gi = RNG.choice(len(Zf), min(MAX_GLOBAL_POINTS, len(Zf)), replace=False)
    g = tsne2(Zf[gi], 30)

    # ---- panels ----
    fig = plt.figure(figsize=(13, 4.4), dpi=600)
    ax0 = fig.add_subplot(1,3,1)
    ax0.scatter(g[:,0], g[:,1], c=ysub[gi], cmap="tab20", s=6, alpha=0.75)
    ax0.set_title("(a) All subjects: identity clusters"); ax0.set_xticks([]); ax0.set_yticks([])

    for k,(subj,tag) in enumerate([(stable,"stable"),(drifty,"drift-prone")]):
        ax = fig.add_subplot(1,3,2+k)
        m = ysub == subj
        e = tsne2(Zf[m], 30); ss = yses[m]
        sc = ax.scatter(e[:,0], e[:,1], c=ss, cmap="viridis", s=10, alpha=0.8,
                        vmin=min(sessions), vmax=max(sessions))
        ax.set_title(f"({'bc'[k]}) Subject {subj} by session ({tag})"); ax.set_xticks([]); ax.set_yticks([])
        if k==1:
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, ticks=sessions)
            cb.set_label("Session")
    fig.suptitle("Identity is separable across subjects; session drift is uneven across individuals",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT, "Fig_embedding_tsne.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    print("SAVED:", out, f"| stable={stable} drift-prone={drifty} | global points={len(gi)}")

except Exception:
    print("ERROR — traceback follows:"); traceback.print_exc(); sys.exit(1)
