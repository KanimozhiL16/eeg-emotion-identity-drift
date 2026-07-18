#!/usr/bin/env python3
"""
step3_baselines.py  -- Step 3: published baselines under the P4 cross-session protocol.

Puts P4's EER in context by evaluating two standard EEG-biometric baselines on the
SAME data and SAME verification protocol (enroll session 1 -> verify session 2 / 3,
genuine vs impostor, EER):

  Baseline A  PSD band-power + cosine prototype   (classic handcrafted biometric)
  Baseline B  EEGNet embedding + cosine prototype (deep baseline, trained on S1)

Protocol: per-subject prototype from session 1; test windows from sessions 2 and 3
scored (cosine) against all 16 claimed prototypes -> genuine/impostor -> EER, AUC.
A within-session reference (S1 split) is also reported.

Outputs -> outputs/run_12_baselines/baseline_comparison.csv

USAGE (project root, env active; GPU used automatically if available):
    cd /home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
    source p4_seedv_env/bin/activate
    python -u scripts/step3_baselines.py 2>&1 | tee outputs/run_12_baselines/step3_log.txt
"""
import os, glob, numpy as np, pandas as pd
from scipy.signal import welch
from sklearn.metrics import roc_curve, roc_auc_score

RNG = np.random.default_rng(0)
MAX_PER_SUBJ = 400          # cap test windows/subject for speed (EER is stable)
def has(p): return os.path.isdir(os.path.join(p, "outputs"))
_hp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT = next((c for c in [os.getcwd(), _hp,
            "/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
            "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if has(c)), os.getcwd())
OUT = os.path.join(ROOT, "outputs", "run_12_baselines"); os.makedirs(OUT, exist_ok=True)
print("="*78); print("STEP 3 BASELINES | root:", ROOT); print("="*78)

# ----------------------------------------------------- load sessions
def load_session(n):
    f = glob.glob(os.path.join(ROOT, "**", f"*SESSION{n}_16sub.npz"), recursive=True)
    z = np.load(sorted(f, key=len)[0], allow_pickle=True)
    return z["X"].astype(np.float32), z["y_subject"].astype(int)
X1, y1 = load_session(1); X2, y2 = load_session(2); X3, y3 = load_session(3)
SUBS = sorted(np.unique(y1).tolist()); print("subjects:", len(SUBS), "| S1/S2/S3:", X1.shape, X2.shape, X3.shape)

def subsample(X, y, cap):
    keep = []
    for s in np.unique(y):
        idx = np.where(y == s)[0]
        if len(idx) > cap: idx = RNG.choice(idx, cap, replace=False)
        keep += list(idx)
    keep = np.array(sorted(keep)); return X[keep], y[keep]

# ----------------------------------------------------- metric helpers
def l2(F): return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
def prototypes(F, y):
    P = np.stack([l2(F[y == s]).mean(0) for s in SUBS]); return l2(P)
def verify_eer(F_test, y_test, P):
    F = l2(F_test); S = F @ P.T               # (n,16) cosine
    gen = S[np.arange(len(F)), [SUBS.index(t) for t in y_test]]
    mask = np.ones_like(S, bool); mask[np.arange(len(F)), [SUBS.index(t) for t in y_test]] = False
    imp = S[mask]
    yv = np.r_[np.ones(len(gen)), np.zeros(len(imp))]; sv = np.r_[gen, imp]
    fpr, tpr, _ = roc_curve(yv, sv); fnr = 1 - tpr
    eer = (fpr[np.nanargmin(np.abs(fpr-fnr))] + fnr[np.nanargmin(np.abs(fpr-fnr))]) / 2
    return float(eer), float(roc_auc_score(yv, sv))

rows = []
def evaluate(name, feat_fn):
    # features
    F1 = feat_fn(X1)
    # S1 within-session: split each subject 50/50 (enroll/test)
    en, te = [], []
    for s in SUBS:
        idx = np.where(y1 == s)[0]; RNG.shuffle(idx); h = len(idx)//2
        en += list(idx[:h]); te += list(idx[h:])
    en, te = np.array(en), np.array(te)
    P_half = prototypes(F1[en], y1[en])
    Xs, ys = subsample(F1[te], y1[te], MAX_PER_SUBJ)
    e, a = verify_eer(Xs, ys, P_half); rows.append((name, "S1 (within)", e, a)); print(f"  {name} S1: EER={e:.4f} AUC={a:.4f}")
    # cross-session: full S1 prototypes -> S2, S3
    P_full = prototypes(F1, y1)
    for sn, X, y in [("S1->S2", X2, y2), ("S1->S3", X3, y3)]:
        F = feat_fn(X); Xs, ys = subsample(F, y, MAX_PER_SUBJ)
        e, a = verify_eer(Xs, ys, P_full); rows.append((name, sn, e, a)); print(f"  {name} {sn}: EER={e:.4f} AUC={a:.4f}")

# ===================================================== Baseline A: PSD + cosine
print("\n--- Baseline A: PSD band-power + cosine ---")
BANDS = [(0.5,4),(4,8),(8,13),(13,30),(30,45)]
def psd_feats(X):
    f, P = welch(X, fs=200, nperseg=200, axis=-1)          # (N,62,Fbins)
    out = []
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi); out.append(P[:, :, m].sum(-1))
    F = np.log(np.concatenate(out, axis=1) + 1e-12)         # (N, 62*5)
    return F.astype(np.float32)
evaluate("PSD+cosine", psd_feats)

# ===================================================== Baseline B: EEGNet
print("\n--- Baseline B: EEGNet embedding + cosine ---")
try:
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("  device:", dev)
    class EEGNet(nn.Module):
        def __init__(self, C=62, T=400, ncl=16, F1=8, D=2, F2=16, k=64):
            super().__init__()
            self.f1 = nn.Sequential(nn.Conv2d(1,F1,(1,k),padding=(0,k//2),bias=False), nn.BatchNorm2d(F1))
            self.dw = nn.Sequential(nn.Conv2d(F1,F1*D,(C,1),groups=F1,bias=False), nn.BatchNorm2d(F1*D),
                                    nn.ELU(), nn.AvgPool2d((1,4)), nn.Dropout(0.5))
            self.sep = nn.Sequential(nn.Conv2d(F1*D,F1*D,(1,16),padding=(0,8),groups=F1*D,bias=False),
                                     nn.Conv2d(F1*D,F2,(1,1),bias=False), nn.BatchNorm2d(F2),
                                     nn.ELU(), nn.AvgPool2d((1,8)), nn.Dropout(0.5))
            self.flat = nn.Flatten(); self.head = nn.LazyLinear(ncl)
        def embed(self, x): return self.flat(self.sep(self.dw(self.f1(x))))
        def forward(self, x): return self.head(self.embed(x))
    def to_t(X): return torch.tensor(X[:,None,:,:], dtype=torch.float32)
    # normalize per-channel using S1 stats
    mu = X1.mean((0,2), keepdims=True); sd = X1.std((0,2), keepdims=True)+1e-6
    nrm = lambda X: ((X-mu)/sd).astype(np.float32)
    Xtr, ytr = subsample(nrm(X1), y1, 1200)
    net = EEGNet().to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = nn.CrossEntropyLoss(); lab = np.array([SUBS.index(s) for s in ytr])
    net(to_t(Xtr[:8]).to(dev))  # init Lazy
    print("  training EEGNet (20 epochs)...")
    for ep in range(20):
        net.train(); perm = RNG.permutation(len(Xtr))
        for i in range(0, len(perm), 256):
            b = perm[i:i+256]; xb = to_t(Xtr[b]).to(dev); yb = torch.tensor(lab[b]).to(dev)
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
    net.eval()
    @torch.no_grad()
    def emb(X):
        X = nrm(X); o = []
        for i in range(0, len(X), 512): o.append(net.embed(to_t(X[i:i+512]).to(dev)).cpu().numpy())
        return np.concatenate(o).astype(np.float32)
    evaluate("EEGNet+cosine", emb)
except Exception as e:
    print("  EEGNet skipped:", repr(e))

# ===================================================== assemble + add P4 context
df = pd.DataFrame(rows, columns=["method","split","EER","AUC"])
ctx = pd.DataFrame([
    ("P4 Mahalanobis (RUN03)","pooled cross-session",0.3182,0.7495),
    ("P4 lightweight+adaptive (RUN07)","S1->S2",0.1818,0.8980),
    ("P4 lightweight+adaptive (RUN07)","S1->S3",0.2508,0.8193),
], columns=["method","split","EER","AUC"])
full = pd.concat([df, ctx], ignore_index=True)
csv = os.path.join(OUT, "baseline_comparison.csv"); full.to_csv(csv, index=False)
print("\n"+"="*78); print("SAVED:", os.path.relpath(csv, ROOT)); print(full.to_string(index=False)); print("="*78)
