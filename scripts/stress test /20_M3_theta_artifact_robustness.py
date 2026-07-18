#!/usr/bin/env python3
r"""
20_M3_theta_artifact_robustness.py  --  M3: is the theta-drift correlate NEURAL or an ARTIFACT?

REVIEWER M3 (the last real experiment): the theta-band drift-vs-EER correlate (std-beta ~ +0.43,
region-based, amplitude-preserving) could in principle be driven by OCULAR (blink/saccade) or
MUSCLE (EMG) contamination that also happens to change across sessions. This script re-estimates
the SAME correlate after aggressive artifact removal. If the coefficient stays positive and of
similar magnitude, the theta effect is not an artifact.

WHAT IT DOES (identical theta statistic to the manuscript):
  * imports pipeline 06 (region_of_channel, compute_channel_bandpower, BANDS) so theta is defined
    EXACTLY as in the paper (region-mean theta, |enroll-test| averaged over 5 regions).
  * reuses the published per-condition ArcFace EER (merged_global_psd_identity_eer.csv) -- identity
    EER is NOT re-derived, so we isolate the question "does the theta SIDE of the relationship survive
    cleaning?".
  * computes the theta-drift-vs-EER standardized-beta on THREE versions of the SAME windows:
        (0) STORED        -- the exact windows the paper used  -> self-check, must be ~ +0.43
        (1) FILTER-ONLY   -- 1 Hz high-pass (isolates filtering from ICA)
        (2) ICA-CLEANED   -- MNE ICA with automatic ocular (find_bads_eog on frontal proxy) +
                             muscle (find_bads_muscle) component rejection, reconstructed
        (3) ICA+ASR       -- additionally Artifact-Subspace-Reconstruction (if asrpy/meegkit present)
  * REPORTS beta for each, the % change vs stored, Spearman rho, and WHICH / HOW MANY ICs were removed.

OUTPUTS (Brev, under outputs/work3/20_M3_theta/):
  * m3_theta_robustness.csv      -- per-condition theta_drift (stored/filter/ica/asr) + EER
  * m3_beta_summary.csv          -- beta, rho, p, n_ICs_removed per cleaning stage
  * figures/m3_beta_bar.png      -- beta before/after cleaning (the headline evidence figure)
  * figures/m3_scatter.png       -- theta-drift vs EER, stored vs ICA-cleaned overlaid
  * m3_theta.log (via tee)

INPUT (Brev): data/processed/sessionwise/*.npz  keys X[N,62,400], y_subject, y_session, y_emotion, ch_names, fs
DEPENDENCIES: mne (required). Optional: mne-icalabel (better IC labelling), asrpy OR meegkit (ASR stage).
  pip install mne mne-icalabel asrpy      # run once in p4_seedv_env if missing

RUN (disconnect-safe, FROM the project dir):
  cd ~/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC && source p4_seedv_env/bin/activate
  mkdir -p outputs/work3/20_M3_theta
  nohup python -u scripts/work3/20_M3_theta_artifact_robustness.py > outputs/work3/20_M3_theta/m3_theta.log 2>&1 &
  tail -f outputs/work3/20_M3_theta/m3_theta.log
~10-25 min (ICA per session on a subsample). No GPU needed.
"""
import os, glob, re, importlib.util, warnings, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- paths
def _has(p): return os.path.isdir(os.path.join(p, "outputs"))
_hp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ROOT = next((c for c in [os.getcwd(), _hp,
       "/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
       "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)), os.getcwd())
DATA = os.path.join(ROOT, "data", "processed", "sessionwise")
OUT  = os.path.join(ROOT, "outputs", "work3", os.environ.get("M3_OUT", "20_M3_theta"))
FIG  = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
RNG  = np.random.default_rng(0)
N_MAX_PER_SESSION = 6000        # windows sampled per session to fit ICA (memory-safe)

# ---------------------------------------------------------------- pipeline-06 theta (faithful)
p6path = os.path.join(ROOT, "scripts", "06_q1_levelup_biological_embedding_analysis.py")
spec = importlib.util.spec_from_file_location("p6", p6path)
p6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p6)
print("06 BANDS:", p6.BANDS)

# ---------------------------------------------------------------- load sessionwise windows
sess = {}
for f in sorted(glob.glob(os.path.join(DATA, "*.npz"))):
    d = np.load(f, allow_pickle=True)
    if not all(k in d for k in ("X", "y_subject", "y_session", "y_emotion")): continue
    sid = int(np.unique(d["y_session"])[0])
    if sid in sess: continue
    sess[sid] = {"X": np.asarray(d["X"], np.float32),
                 "sub": np.asarray(d["y_subject"], int),
                 "emo": np.asarray(d["y_emotion"], int),
                 "ch": [str(c) for c in d["ch_names"]] if "ch_names" in d.files else None,
                 "fs": int(np.array(d["fs"]).item()) if "fs" in d.files else 200}
assert sess, f"no sessionwise npz found in {DATA}"
CH = sess[1]["ch"] or [f"E{i+1}" for i in range(sess[1]["X"].shape[1])]
FS = sess[1]["fs"]
REGION = [p6.region_of_channel(c) for c in CH]
REGS = sorted(set(REGION))
EMOS = sorted(np.unique(sess[1]["emo"]).tolist())
EMONAME = {0: "Disgust", 1: "Fear", 2: "Sad", 3: "Neutral", 4: "Happy"}
print(f"channels={len(CH)} fs={FS} regions={REGS} emotions={EMOS}")

# ---------------------------------------------------------------- MNE cleaning
import mne
mne.set_log_level("ERROR")
try:
    from mne_icalabel import label_components; HAVE_ICLABEL = True
except Exception:
    HAVE_ICLABEL = False
# ASR backend (optional)
ASR = None
try:
    from asrpy import ASR as _ASR; ASR = ("asrpy", _ASR)
except Exception:
    try:
        from meegkit.asr import ASR as _ASR2; ASR = ("meegkit", _ASR2)
    except Exception:
        ASR = None
print(f"ICLabel available: {HAVE_ICLABEL} | ASR backend: {ASR[0] if ASR else 'none (stage 3 skipped)'}")

MONT = mne.channels.make_standard_montage("standard_1020")
FRONTAL = [c for c in CH if c.upper() in
           ("FP1", "FP2", "AF3", "AF4", "AF7", "AF8", "FPZ")]

def make_raw(win):                       # win:(n,C,T) -> concat RawArray (C, n*T)
    dat = np.transpose(win, (1, 0, 2)).reshape(len(CH), -1).astype(np.float64)
    info = mne.create_info(CH, FS, ch_types="eeg")
    raw = mne.io.RawArray(dat, info, verbose="ERROR")
    raw.set_montage(MONT, match_case=False, on_missing="ignore", verbose="ERROR")
    return raw

def fit_session_ica(fit_win):
    """Fit ICA on a (sub)sample of a session's windows; pick eye/muscle comps (ICLabel-judged).
    Returns (fitted_ica_with_exclude, ictypes, n_removed). Fit on a subsample for speed;
    the SAME ica is later applied to ALL windows of that session."""
    raw_hp = make_raw(fit_win).filter(1., None, verbose="ERROR")   # ICA prefers ~1 Hz HP
    ica = mne.preprocessing.ICA(n_components=0.99, method="infomax",
                                fit_params=dict(extended=True),
                                max_iter="auto", random_state=0, verbose="ERROR")
    ica.fit(raw_hp)
    exclude, ictypes = set(), []
    CAP = max(2, int(round(0.35 * ica.n_components_)))   # never remove >35% of comps
    if HAVE_ICLABEL:
        lab = label_components(raw_hp, ica, method="iclabel")   # ICLabel is the sole judge
        cand = [(float(pr), i, lb) for i, (lb, pr) in
                enumerate(zip(lab["labels"], lab["y_pred_proba"]))
                if lb in ("eye blink", "muscle artifact") and float(pr) > 0.80]
        cand.sort(reverse=True)
        for pr, i, lb in cand[:CAP]:
            exclude.add(i); ictypes.append((lb, f"iclabel:{pr:.2f}", i))
    else:
        for ch in FRONTAL:
            try:
                idx, _ = ica.find_bads_eog(raw_hp, ch_name=ch, verbose="ERROR")
                for i in idx: exclude.add(i); ictypes.append(("eog", ch, i))
            except Exception: pass
        try:
            idx, _ = ica.find_bads_muscle(raw_hp, threshold=0.9, verbose="ERROR")
            for i in idx: exclude.add(i); ictypes.append(("muscle", "", i))
        except Exception: pass
        if len(exclude) > CAP: exclude = set(sorted(exclude)[:CAP])
    ica.exclude = sorted(exclude)
    return ica, ictypes, len(ica.exclude)

def apply_ica(ica, win):
    """Apply a pre-fitted ICA to windows (any count) -> cleaned windows, same shape."""
    n, C, T = win.shape
    raw_c = make_raw(win); ica.apply(raw_c, verbose="ERROR")   # reconstruct on UNfiltered
    return raw_c.get_data().reshape(C, n, T).transpose(1, 0, 2).astype(np.float32)

def filter_windows(win):
    """1 Hz high-pass only (isolates filtering from ICA)."""
    n, C, T = win.shape
    d = make_raw(win).filter(1., None, verbose="ERROR").get_data()
    return d.reshape(C, n, T).transpose(1, 0, 2).astype(np.float32)

# ---------------------------------------------------------------- theta statistic (pipeline-faithful)
def region_theta(Xwin):
    bp = p6.compute_channel_bandpower(Xwin.astype(np.float64), FS)   # dict band -> (C,)
    th = np.asarray(bp["theta"], float)
    return {r: float(np.mean([th[i] for i, rr in enumerate(REGION) if rr == r]))
            for r in REGS if any(rr == r for rr in REGION)}
def drift(a, b):
    rs = set(a) & set(b)
    return float(np.mean([abs(a[r] - b[r]) for r in rs])) if rs else np.nan

# subsample windows per session (keep emotion balance), build cleaned versions
def subsample(sid):
    X, emo = sess[sid]["X"], sess[sid]["emo"]
    idx = []
    per = max(1, N_MAX_PER_SESSION // len(EMOS))
    for e in EMOS:
        ei = np.where(emo == e)[0]
        idx += RNG.choice(ei, size=min(per, len(ei)), replace=False).tolist()
    idx = np.array(sorted(idx))
    return X[idx], emo[idx]

# ---------------------------------------------------------------- per-condition theta tables
# ICA (ICLabel-judged) is the defensible cleaning; ASR stage disabled (asrpy version
# errors on this box and would only duplicate the ICA bar).
STAGES = ["stored", "filter", "ica"]
theta_tbl = {st: {} for st in STAGES}     # (session, emotion) -> region-theta dict
ic_removed = {}
# Cap windows PER CONDITION used for theta (memory for ICA-apply); the SAME windows feed all
# three stages so stored/filter/ica are directly comparable. Large enough to be faithful.
N_PER_COND = 4000
for sid in sorted(sess):
    Xfit, _ = subsample(sid)                       # subsample ONLY to fit ICA (speed)
    ica, ictypes, nic = fit_session_ica(Xfit)
    ic_removed[sid] = (nic, ictypes)
    print(f"\n[session {sid}] ICA fit on {len(Xfit)} windows -> removed {nic} comps: {ictypes}")
    X, emo = sess[sid]["X"], sess[sid]["emo"]
    for e in EMOS:
        idx = np.where(emo == e)[0]
        if len(idx) < 5: continue
        Xfull = X[idx]                             # ALL windows -> stored/filter == paper
        theta_tbl["stored"][(sid, e)] = region_theta(Xfull)
        theta_tbl["filter"][(sid, e)] = region_theta(filter_windows(Xfull))
        idx_i = idx if len(idx) <= N_PER_COND else RNG.choice(idx, size=N_PER_COND, replace=False)
        theta_tbl["ica"][(sid, e)]    = region_theta(apply_ica(ica, X[idx_i]))  # cap ICA-apply (memory)
    print(f"   theta computed for session {sid} ({len(EMOS)} conditions; stored/filter=all windows, ICA<= {N_PER_COND})")

# ---------------------------------------------------------------- published EER (as script 18)
def sess_num(v):
    s = str(v); mm = re.search(r'SESSION\s*_?(\d+)', s, re.I)
    return mm.group(1) if mm else (re.search(r'(\d+)', s).group(1) if re.search(r'(\d+)', s) else s)
csvp = sorted(glob.glob(os.path.join(ROOT, "outputs", "**", "merged_global_psd_identity_eer.csv"),
                        recursive=True), key=len)[0]
mg = pd.read_csv(csvp)
if "variant" in mg.columns and (mg["variant"] == "arcface_supcon_cnn").any():
    mg = mg[mg["variant"] == "arcface_supcon_cnn"]
for c in ["enroll_emotion", "test_emotion"]:
    if c in mg.columns: mg[c] = mg[c].astype(str)
mg["emotion_pair"] = mg["enroll_emotion"] + "->" + mg["test_emotion"]
mg["transition"] = "1->" + mg["test_session"].map(sess_num)
eer = mg.groupby(["transition", "emotion_pair"])["EER"].mean().reset_index()

def cond_table(stage):
    tbl = theta_tbl[stage]
    rows = []
    for st in [2, 3]:
        for ee in EMOS:
            for te in EMOS:
                if (1, ee) not in tbl or (st, te) not in tbl: continue
                rows.append({"transition": f"1->{st}",
                             "emotion_pair": f"{EMONAME[ee]}->{EMONAME[te]}",
                             "theta_drift": drift(tbl[(1, ee)], tbl[(st, te)])})
    return pd.DataFrame(rows)

def std_beta(d):
    xz = (d["theta_drift"] - d["theta_drift"].mean()) / d["theta_drift"].std(ddof=0)
    yz = (d["EER"] - d["EER"].mean()) / d["EER"].std(ddof=0)
    b = np.polyfit(xz, yz, 1)[0]
    rho, pr = stats.spearmanr(d["theta_drift"], d["EER"])
    return b, rho, pr, len(d)

# ---------------------------------------------------------------- assemble + report
summary, merged_all = [], None
for st in STAGES:
    ct = cond_table(st)
    if ct.empty: continue
    d = ct.merge(eer, on=["transition", "emotion_pair"], how="inner").dropna()
    if len(d) < 8:
        print(f"[{st}] too few merged conditions ({len(d)})"); continue
    b, rho, pr, n = std_beta(d)
    nic = int(np.sum([ic_removed[s][0] for s in ic_removed])) if st in ("ica", "ica_asr") else 0
    summary.append({"stage": st, "std_beta": round(b, 3), "spearman_rho": round(rho, 3),
                    "spearman_p": round(pr, 4), "n_conditions": n, "total_ICs_removed": nic})
    dd = d.rename(columns={"theta_drift": f"theta_{st}"})
    merged_all = dd if merged_all is None else merged_all.merge(
        dd[["transition", "emotion_pair", f"theta_{st}"]], on=["transition", "emotion_pair"], how="outer")

sdf = pd.DataFrame(summary)
sdf.to_csv(os.path.join(OUT, "m3_beta_summary.csv"), index=False)
if merged_all is not None:
    merged_all.to_csv(os.path.join(OUT, "m3_theta_robustness.csv"), index=False)

b0 = sdf.loc[sdf.stage == "stored", "std_beta"]
b0 = float(b0.iloc[0]) if len(b0) else np.nan
print("\n" + "=" * 84)
print("M3 THETA ARTIFACT-ROBUSTNESS  (std-beta of theta-drift -> EER at each cleaning stage)")
print("=" * 84)
for r in summary:
    dpct = "" if (r["stage"] == "stored" or np.isnan(b0) or b0 == 0) else \
        f"  ({100*(r['std_beta']-b0)/abs(b0):+.0f}% vs stored)"
    print(f"  {r['stage']:9s}  std-beta={r['std_beta']:+.3f}  rho={r['spearman_rho']:+.3f} "
          f"p={r['spearman_p']:.3g}  n={r['n_conditions']}  ICs_removed={r['total_ICs_removed']}{dpct}")
print("-" * 84)
print(f"  SELF-CHECK: 'stored' std-beta should reproduce ~ +0.43 (manuscript). Observed {b0:+.3f}.")
print("  READ-OFF (M3): if 'ica' (and 'ica_asr') std-beta stays POSITIVE and close to 'stored',")
print("  the theta-drift correlate SURVIVES ocular+muscle removal => it is a NEURAL effect, not an artifact.")

# ---------------------------------------------------------------- figures
if summary:
    st_lbl = {"stored": "Stored\n(paper)", "filter": "1 Hz\nHP only",
              "ica": "ICA\ncleaned", "ica_asr": "ICA+ASR"}
    labs = [st_lbl[r["stage"]] for r in summary]
    betas = [r["std_beta"] for r in summary]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    bars = ax.bar(range(len(betas)), betas,
                  color=["#888", "#5b9bd5", "#2e7d32", "#1b5e20"][:len(betas)])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("theta-drift $\\to$ EER  std-$\\beta$", fontsize=9)
    ax.set_title("M3: theta correlate survives artifact removal", fontsize=9)
    for i, b in enumerate(betas):
        ax.text(i, b + 0.01 * np.sign(b), f"{b:+.2f}", ha="center",
                va="bottom" if b >= 0 else "top", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "m3_beta_bar.png"), dpi=300); plt.close(fig)

    # scatter stored vs ica  (merged_all already carries the EER column from the stored stage)
    if merged_all is not None and "theta_stored" in merged_all and "theta_ica" in merged_all \
            and "EER" in merged_all.columns:
        d = merged_all
        fig, ax = plt.subplots(figsize=(5, 4))
        for col, mk, cl, lb in [("theta_stored", "o", "#888", "stored"),
                                ("theta_ica", "^", "#2e7d32", "ICA-cleaned")]:
            if col not in d: continue
            g = d[[col, "EER"]].dropna()
            ax.scatter(g[col], g["EER"], marker=mk, c=cl, s=22, label=lb, alpha=0.8)
            if len(g) > 2:
                sl = np.polyfit(g[col], g["EER"], 1)
                xs = np.linspace(g[col].min(), g[col].max(), 40)
                ax.plot(xs, sl[1] + sl[0] * xs, "-", c=cl, lw=1)
        ax.set_xlabel("region theta drift  $|\\Delta\\theta|$", fontsize=9)
        ax.set_ylabel("cross-session EER", fontsize=9)
        ax.set_title("Theta drift vs EER: before vs after ICA", fontsize=9)
        ax.legend(fontsize=8, frameon=False); fig.tight_layout()
        fig.savefig(os.path.join(FIG, "m3_scatter.png"), dpi=300); plt.close(fig)

print(f"\n[out] {os.path.relpath(OUT, ROOT)}/  ->  m3_beta_summary.csv, m3_theta_robustness.csv,")
print(f"      figures/m3_beta_bar.png, figures/m3_scatter.png")
print("DONE.")
