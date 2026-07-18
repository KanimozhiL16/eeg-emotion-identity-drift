#!/usr/bin/env python3
"""
33_aep_within_vs_cross.py  --  REVIEWER FIX M3 (auditory dataset: demonstrate DEGRADATION, not one EER).

The manuscript reported a single AEP cross-condition EER (0.337) with no within-condition reference,
so "generalises" was not shown as a degradation. The PhysioNet AEP dataset (Abo Alzahab et al. 2021,
DOI 10.13026/ps31-fc50; 20 subjects, 4 ch T7/F8/Cz/P4) records resting-state experiment ex01 over
THREE sessions -> a clean within-session vs cross-session comparison, directly analogous to SEED-V.

The per-window provenance is in npz key 'source_file' = 'sXX_exYY[_sZZ].csv'. This script parses it,
then under the identical enrol-then-verify cosine protocol reports, per resting experiment (ex01, ex02):
  EER_within  = enrol session 1 (50% split), verify the session-1 held-out half
  EER_cross   = enrol session 1, verify sessions 2 and 3
  dEER        = EER_cross - EER_within   (the degradation; per-subject Wilcoxon + bootstrap 95% CI)
Also a cross-CONDITION check: enrol resting ex01-s1, verify the auditory experiments (ex05-ex10).

Outputs -> outputs/run_20_aep_within_vs_cross/{aep_within_vs_cross.csv, persubject.csv, log.txt}
USAGE:
  cd /home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
  source p4_seedv_env/bin/activate
  mkdir -p outputs/run_20_aep_within_vs_cross
  python -u scripts/33_aep_within_vs_cross.py 2>&1 | tee outputs/run_20_aep_within_vs_cross/log.txt
"""
import os, re, numpy as np, pandas as pd
from scipy import signal, stats
from sklearn.metrics import roc_curve

def _has(p): return os.path.isdir(os.path.join(p, "outputs"))
_hp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT = next((c for c in [os.getcwd(), _hp,
      "/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)), os.getcwd())
NPZ = os.path.join(ROOT, "data", "cross_dataset_aep", "AEP_win2s_step1s_fs256_4ch.npz")
OUT = os.path.join(ROOT, "outputs", "run_20_aep_within_vs_cross"); os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)
print("=" * 80); print("M3  AEP within-session vs cross-session degradation"); print("=" * 80)

d = np.load(NPZ, allow_pickle=True)
X = np.asarray(d["X"], np.float32)                 # (N, 4, 512)
subj = np.asarray(d["y_subject"], int)
src = np.asarray([str(s) for s in d["source_file"]])
fs = int(d["fs"]) if "fs" in d and np.ndim(d["fs"]) == 0 else 256
print(f"  X={X.shape}  subjects={len(np.unique(subj))}  fs={fs}")

# ---- parse provenance: sXX_exYY[_sZZ] ----
def parse(s):
    m = re.match(r"s(\d+)_ex(\d+)(?:_s(\d+))?", s)
    if not m: return (None, None, None)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else 0)
pid = np.array([parse(s) for s in src])            # columns: subj, exp, sess
exp = pid[:, 1]; sess = pid[:, 2]
print(f"  experiments present: {sorted(np.unique(exp))}  (ex01/ex02 carry sessions {sorted(np.unique(sess[np.isin(exp,[1,2])]))})")

# ---- features: per-channel time stats + 5 band powers (fs) ----
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
def feats(win):                                    # win (4, T)
    row = []
    for ch in win:
        row += [ch.mean(), ch.std(), np.sqrt((ch ** 2).mean())]
        f, p = signal.welch(ch, fs=fs, nperseg=min(256, ch.shape[-1]))
        for lo, hi in BANDS: row.append(float(p[(f >= lo) & (f < hi)].sum()))
    return row
F = np.array([feats(w) for w in X], float)
print(f"  features: {F.shape}")

def eer_of(y, s):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr))); return float((fpr[i] + fnr[i]) / 2)

def l2(v): 
    n = np.linalg.norm(v, axis=-1, keepdims=True); return v / np.clip(n, 1e-12, None)

def verify(enrol_idx, probe_idx):
    """enrol prototypes per subject on enrol_idx; score probe_idx by cosine. Returns pooled EER + per-subject dict."""
    mu = F[enrol_idx].mean(0); sd = F[enrol_idx].std(0) + 1e-8   # enrolment-only standardisation
    Fe = (F[enrol_idx] - mu) / sd; Fp = (F[probe_idx] - mu) / sd
    se, sp = subj[enrol_idx], subj[probe_idx]
    subs = np.unique(se)
    P = np.array([l2(Fe[se == c].mean(0)) for c in subs]); own = subs
    Q = l2(Fp)
    S = Q @ P.T                                   # cosine (both L2)
    y_true, score, prb = [], [], []
    for j, c in enumerate(own):
        y_true.append((sp == c).astype(int)); score.append(S[:, j]); prb.append(sp)
    y_true = np.concatenate(y_true); score = np.concatenate(score); prb = np.concatenate(prb)
    pooled = eer_of(y_true, score)
    # per-subject EER: genuine = probes of c vs template c; impostor = probes of others vs template c
    per = {}
    for j, c in enumerate(own):
        gen = S[sp == c, j]; imp = S[sp != c, j]
        yy = np.r_[np.ones(len(gen)), np.zeros(len(imp))]; ss = np.r_[gen, imp]
        if len(gen) and len(imp): per[int(c)] = eer_of(yy, ss)
    return pooled, per

rows, ps_rows = [], []
def split_enrol(mask_s1):
    """50/50 per-subject split of the session-1 mask -> (enrol_idx, heldout_idx)."""
    en = np.zeros(len(subj), bool)
    for c in np.unique(subj[mask_s1]):
        idx = np.where(mask_s1 & (subj == c))[0]
        if len(idx) < 2: en[idx] = True; continue
        RNG.shuffle(idx); en[idx[:len(idx)//2]] = True
    enrol_idx = np.where(en)[0]; held_idx = np.where(mask_s1 & ~en)[0]
    return enrol_idx, held_idx

for ex, name in [(1, "ex01_resting_eyes_open"), (2, "ex02_resting_eyes_closed")]:
    m_s1 = (exp == ex) & (sess == 1)
    m_cross = (exp == ex) & np.isin(sess, [2, 3])
    if m_s1.sum() < 10 or m_cross.sum() < 10:
        print(f"  [{name}] insufficient sessions, skip"); continue
    enrol_idx, held_idx = split_enrol(m_s1)
    eer_w, per_w = verify(enrol_idx, held_idx)
    eer_c, per_c = verify(enrol_idx, np.where(m_cross)[0])
    print(f"\n[{name}]  EER_within={eer_w:.4f}  EER_cross(s2+s3)={eer_c:.4f}  dEER={eer_c-eer_w:+.4f}")
    rows.append({"experiment": name, "EER_within": eer_w, "EER_cross": eer_c, "dEER": eer_c - eer_w})
    for c in set(per_w) & set(per_c):
        ps_rows.append({"experiment": name, "subject": c, "EER_within": per_w[c],
                        "EER_cross": per_c[c], "dEER": per_c[c] - per_w[c]})

# cross-CONDITION: enrol resting ex01 s1, verify auditory ex05-ex10
m_s1 = (exp == 1) & (sess == 1)
enrol_idx, _ = split_enrol(m_s1)
m_aud = np.isin(exp, [5, 6, 7, 8, 9, 10])
if m_aud.sum() > 10:
    eer_aud, _ = verify(enrol_idx, np.where(m_aud)[0])
    eer_w1, _ = verify(*split_enrol(m_s1))
    print(f"\n[cross-condition] enrol resting ex01-s1 -> verify auditory ex05-10  EER={eer_aud:.4f}")
    rows.append({"experiment": "resting->auditory(cross-condition)", "EER_within": np.nan,
                 "EER_cross": eer_aud, "dEER": np.nan})

df = pd.DataFrame(rows); ps = pd.DataFrame(ps_rows)
df.to_csv(os.path.join(OUT, "aep_within_vs_cross.csv"), index=False)
ps.to_csv(os.path.join(OUT, "persubject.csv"), index=False)

# per-subject significance + bootstrap CI on dEER, pooled over the two resting experiments
if len(ps):
    dd = ps.groupby("subject")["dEER"].mean()
    try: W, pv = stats.wilcoxon(dd.values)
    except Exception: pv = float("nan")
    bs = [np.mean(dd.values[RNG.integers(0, len(dd), len(dd))]) for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print("\n" + "=" * 80)
    print(f"RESTING within->cross degradation: mean dEER={dd.mean():+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  Wilcoxon p={pv:.3g}  (n={len(dd)})")
    print("READ-OFF: dEER>0 and p<0.05 => a genuine cross-session DEGRADATION on AEP -> 'the phenomenon")
    print("recurs on an independent auditory dataset' is now demonstrated (not a single EER).")
    print("=" * 80)
