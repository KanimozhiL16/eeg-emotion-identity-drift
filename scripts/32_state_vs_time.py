# ============================================================================
# DEPRECATED — DO NOT USE.
# This script uses a random-window split (can place 50%-overlapping neighbours
# in both enrolment and test) and reports the older +0.047 / +0.077 effects.
# It is SUPERSEDED by the leakage-free, whole-trial-disjoint analysis in:
#     scripts/r4rev_reanalysis.py   (2x2, theta-transition, manip check)
#     scripts/r4rev_finalize.py     (participant-aware mixed model + FDR)
# The canonical trial-disjoint results are +0.031 (emotion) / +0.062 (session).
# Retained only for provenance. See docs/REPRODUCE.md.
# ============================================================================#!/usr/bin/env python3
"""
32_state_vs_time.py  --  REVIEWER FIX M2 (state vs time decomposition).

The manuscript's headline degradation is CROSS-SESSION (confounds elapsed time + montage +
cognitive state). The title claims "cognitive-state-induced". This script separates the two by
crossing SESSION change with EMOTION (cognitive-state) change, all from a fixed session-1 enrolment,
reusing run_07's OWN feature/scoring functions so numbers are consistent with the manuscript.

2x2 design (enrol on session 1):
  baseline    = same session, same emotion      (session_changed=0, emotion_changed=0)
  STATE-only  = same session, different emotion  (0,1)  <- pure cognitive-state drift
  TIME-only   = different session, same emotion  (1,0)  <- pure elapsed-time drift
  BOTH        = different session, different emotion (1,1)

Reports mean EER per cell, dEER_state and dEER_time vs baseline (per-subject Wilcoxon + bootstrap
95% CIs), and a two-way OLS  EER ~ session_changed + emotion_changed  to weigh the two effects.

Outputs -> outputs/run_19_state_vs_time/{cells.csv, conditions.csv, persubject.csv, deltas.csv, summary.txt, log.txt}
USAGE:
  cd /home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
  source p4_seedv_env/bin/activate
  pip install -q statsmodels --break-system-packages
  mkdir -p outputs/run_19_state_vs_time
  python -u scripts/32_state_vs_time.py 2>&1 | tee outputs/run_19_state_vs_time/log.txt
"""
import os, glob, importlib.util, numpy as np, pandas as pd
from scipy import stats

def _has(p): return os.path.isdir(os.path.join(p, "outputs"))
_hp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT = next((c for c in [os.getcwd(), _hp,
      "/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)), os.getcwd())
SRC  = os.path.join(ROOT, "scripts", "07_seedv_session_drift_subject_adaptation.py")
DATA = os.path.join(ROOT, "data", "processed", "sessionwise")
OUT  = os.path.join(ROOT, "outputs", "run_19_state_vs_time"); os.makedirs(OUT, exist_ok=True)
RNG  = np.random.default_rng(42)
EMO_NAME = {0: "Disgust", 1: "Fear", 2: "Sad", 3: "Neutral", 4: "Happy"}
print("=" * 80); print("M2  state (emotion) vs time (session) decomposition  |  run_07 functions"); print("=" * 80)

spec = importlib.util.spec_from_file_location("run07", SRC)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

# ---- load sessions with emotion labels (self-contained downsample keeps F/sub/emo aligned) ----
MAXPS = 900
def cap_indices(ys, cap, seed=42):
    rng = np.random.default_rng(seed); keep = []
    for su in np.unique(ys):
        idx = np.where(ys == su)[0]
        if len(idx) > cap: idx = rng.choice(idx, cap, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))

sess = {}
for p in sorted(set(glob.glob(os.path.join(DATA, "*.npz")))):
    d = np.load(p, allow_pickle=True)
    if not all(k in d for k in ("X", "y_subject", "y_session", "y_emotion")): continue
    sid = int(np.unique(d["y_session"])[0])
    if sid in sess: continue
    X = np.asarray(d["X"], np.float32)
    ys = np.asarray(d["y_subject"], int); ye = np.asarray(d["y_emotion"], int)
    keep = cap_indices(ys, MAXPS)
    X, ys, ye = X[keep], ys[keep], ye[keep]
    F = np.asarray(m.simple_features(X), np.float64)
    sess[sid] = {"F": F, "sub": ys, "emo": ye}
    print(f"  session {sid}: F={F.shape}  emotions={sorted(np.unique(ye))}")

SUBS = sorted(np.unique(sess[1]["sub"]).tolist())
EMOS = sorted(np.unique(sess[1]["emo"]).tolist())

# ---- 50/50 split of session 1 per (subject, emotion): enrol half vs held-out half ----
F1, s1, e1 = sess[1]["F"], sess[1]["sub"], sess[1]["emo"]
enrol_mask = np.zeros(len(s1), bool)
for su in SUBS:
    for em in EMOS:
        idx = np.where((s1 == su) & (e1 == em))[0]
        if len(idx) < 2:
            enrol_mask[idx] = True; continue
        RNG.shuffle(idx); enrol_mask[idx[:len(idx)//2]] = True
EN = {"F": F1[enrol_mask], "sub": s1[enrol_mask], "emo": e1[enrol_mask]}
HO = {"F": F1[~enrol_mask], "sub": s1[~enrol_mask], "emo": e1[~enrol_mask]}

def probe_pool(St):
    return HO if St == 1 else {"F": sess[St]["F"], "sub": sess[St]["sub"], "emo": sess[St]["emo"]}

rows, ps_rows = [], []
for e in EMOS:
    em = EN["emo"] == e
    if em.sum() < len(SUBS): continue
    P, owners = m.make_prototypes(EN["F"][em], EN["sub"][em]); owners = np.asarray(owners)
    for St in [1, 2, 3]:
        pool = probe_pool(St)
        for et in EMOS:
            pm = pool["emo"] == et
            if pm.sum() < len(SUBS): continue
            sdf = m.verification_scores(pool["F"][pm], pool["sub"][pm], P, owners)
            _r = m.eer_auc(sdf["y_true"].values, sdf["score"].values); eer, auc = _r[0], _r[1]
            sc, ec = int(St != 1), int(et != e)
            rows.append({"enrol_emotion": EMO_NAME.get(e, e), "test_session": St,
                         "test_emotion": EMO_NAME.get(et, et), "session_changed": sc,
                         "emotion_changed": ec, "EER": eer, "AUC": auc, "n_probe": int(pm.sum())})
            for c, g in sdf.groupby("probe_subject"):
                pe = m.eer_auc(g["y_true"].values, g["score"].values)[0]
                ps_rows.append({"subject": int(c), "session_changed": sc, "emotion_changed": ec, "EER": pe})

df = pd.DataFrame(rows); ps = pd.DataFrame(ps_rows)
df.to_csv(os.path.join(OUT, "conditions.csv"), index=False)
ps.to_csv(os.path.join(OUT, "persubject.csv"), index=False)

cell = df.groupby(["session_changed", "emotion_changed"])["EER"].agg(["mean", "std", "count"]).reset_index()
cell.to_csv(os.path.join(OUT, "cells.csv"), index=False)
def cm(sc, ec):
    r = cell[(cell.session_changed == sc) & (cell.emotion_changed == ec)]
    return float(r["mean"].iloc[0]) if len(r) else float("nan")
base, state, time_, both = cm(0, 0), cm(0, 1), cm(1, 0), cm(1, 1)

def per_subject_cell(sc, ec):
    return ps[(ps.session_changed == sc) & (ps.emotion_changed == ec)].groupby("subject")["EER"].mean()
b0 = per_subject_cell(0, 0)
def delta_test(sc, ec, label):
    x = per_subject_cell(sc, ec); j = b0.index.intersection(x.index)
    d = (x.loc[j] - b0.loc[j]).values
    try: W, pv = stats.wilcoxon(x.loc[j].values, b0.loc[j].values)
    except Exception: W, pv = float("nan"), float("nan")
    bs = [np.mean(d[RNG.integers(0, len(d), len(d))]) for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  dEER_{label:5s} = {np.mean(d):+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  Wilcoxon p={pv:.3g}  (n={len(d)})")
    return {"effect": label, "dEER": float(np.mean(d)), "ci_lo": float(lo), "ci_hi": float(hi),
            "wilcoxon_p": float(pv), "n": int(len(d))}

print("\n-- 2x2 cell mean EER (pooled) --")
print(f"  baseline (same sess, same emo) = {base:.4f}")
print(f"  STATE-only (same sess, diff emo) = {state:.4f}")
print(f"  TIME-only  (diff sess, same emo) = {time_:.4f}")
print(f"  BOTH       (diff sess, diff emo) = {both:.4f}")
print("\n-- per-subject deltas vs baseline --")
d_state = delta_test(0, 1, "state"); d_time = delta_test(1, 0, "time"); d_both = delta_test(1, 1, "both")

try:
    import statsmodels.formula.api as smf
    mod = smf.ols("EER ~ session_changed + emotion_changed", df).fit()
    print("\n-- two-way OLS  EER ~ session_changed + emotion_changed --")
    for t in ["session_changed", "emotion_changed"]:
        print(f"     {t}: beta={mod.params[t]:+.4f}  p={mod.pvalues[t]:.3g}")
    open(os.path.join(OUT, "summary.txt"), "w").write(str(mod.summary()))
except Exception as ex:
    print("  statsmodels unavailable:", ex)

pd.DataFrame([d_state, d_time, d_both]).to_csv(os.path.join(OUT, "deltas.csv"), index=False)
ratio = (d_state["dEER"] / d_time["dEER"]) if d_time["dEER"] else float("nan")
print("\n" + "=" * 80)
print(f"READ-OFF: cognitive-STATE dEER={d_state['dEER']:+.4f} vs TIME dEER={d_time['dEER']:+.4f}  (state/time ratio={ratio:.2f}).")
print("  Both significant & comparable -> keep 'cognitive-state AND cross-session' framing.")
print("  Time dominates, state small/ns  -> retitle to 'cross-session' (state = correlational only).")
print("=" * 80)
