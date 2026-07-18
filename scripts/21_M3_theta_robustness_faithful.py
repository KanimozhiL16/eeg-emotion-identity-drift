#!/usr/bin/env python3
r"""
21_M3_theta_robustness_faithful.py -- M3 (correct): is the theta correlate NEURAL or ARTIFACT?

FAITHFUL rebuild. The paper's theta correlate uses pipeline-06's `theta_power_drift`:
  per (session,emotion): compute_channel_bandpower (welch, MEAN psd in 4-8 Hz, 2500-window
     subsample, seed = 100 + session_index + emotion_id) -> per-channel theta;
  region power = mean over channels in region;
  region drift = | theta_power(session1, enroll_emotion, region) - theta_power(session_t, test_emotion, region) |;
  theta_power_drift = mean over regions.
Regressing that column vs ArcFace EER gives the paper's POSITIVE std-beta (~ +0.32 simple / +0.43 Table-3).

M3 recomputes THAT EXACT quantity on (a) STORED windows [must reproduce the paper's +] and
(b) the SAME windows after ICA ocular+muscle removal. If the std-beta stays positive and close
to stored, the theta correlate SURVIVES artifact removal => NEURAL, not artifact.

Key faithfulness choices:
  * imports pipeline-06 (compute_channel_bandpower, region_of_channel, REGION_ORDER, EMO_MAP,
    BANDS, get_session_files, read_npz_file) so theta is byte-for-byte the manuscript's.
  * uses the pipeline's per-condition seed (100+sess+emo) so STORED == paper.
  * cleans the SAME 2500 windows that feed the statistic (stored vs ICA differ ONLY by cleaning).
  * reuses the published ArcFace EER (merged_global_psd_identity_eer.csv).

OUTPUT (Brev): outputs/work3/<M3_OUT or 21_M3_faithful>/
  m3f_beta_summary.csv   -- std-beta, Spearman, p, ICs removed per stage (stored/filter/ica)
  m3f_theta_conditions.csv -- per-condition theta_drift(stored/filter/ica) + paper theta + EER
  figures/m3f_beta_bar.png, figures/m3f_scatter.png
RUN (from project dir):
  cd ~/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
  export M3_OUT=21_M3_faithful; mkdir -p outputs/work3/$M3_OUT
  nohup python -u "scripts/21_M3_theta_robustness_faithful.py" > outputs/work3/$M3_OUT/m3f.log 2>&1 &
  tail -f outputs/work3/$M3_OUT/m3f.log
First check in log: 'stored' std-beta must be POSITIVE and ~ the paper value, AND
'my stored theta_drift vs paper theta_power_drift' correlation must be ~ 0.99 (faithfulness proof).
"""
import os, glob, re, importlib.util, warnings, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def _has(p): return os.path.isdir(os.path.join(p, "outputs"))
_hp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ROOT = next((c for c in [os.getcwd(), _hp,
       "/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
       "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)), os.getcwd())
OUT = os.path.join(ROOT, "outputs", "work3", os.environ.get("M3_OUT", "21_M3_faithful"))
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)

# ---- import pipeline 06 (exact manuscript functions) ----
p6path = os.path.join(ROOT, "scripts", "06_q1_levelup_biological_embedding_analysis.py")
spec = importlib.util.spec_from_file_location("p6", p6path)
p6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p6)
BANDS = p6.BANDS; EMO_MAP = p6.EMO_MAP; REGION_ORDER = p6.REGION_ORDER
print("06 BANDS:", BANDS, "| REGION_ORDER:", REGION_ORDER)

# ---- load the 3 sessions with the pipeline's OWN loaders ----
files = p6.get_session_files()
print("session files:", [os.path.basename(str(f)) for f in files])
SESS = {}   # sess_idx(1..) -> dict(X, y_emo, ch, fs, regions)
for sidx, f in enumerate(files, start=1):
    X, y_sub, y_emo, fs, ch_names, keys = p6.read_npz_file(f)
    SESS[sidx] = dict(X=np.asarray(X, np.float32), y_emo=np.asarray(y_emo, int),
                      ch=[str(c) for c in ch_names], fs=int(fs),
                      regions=[p6.region_of_channel(c) for c in ch_names])
CH = SESS[1]["ch"]; FS = SESS[1]["fs"]; REGIONS = SESS[1]["regions"]
REG_PRESENT = [r for r in REGION_ORDER if r in set(REGIONS)]
EMOS = sorted(np.unique(SESS[1]["y_emo"]).tolist())
print(f"channels={len(CH)} fs={FS} regions={REG_PRESENT} emotions={EMOS}")

# ---- MNE cleaning (fit per session on subsample, apply to the exact stat windows) ----
import mne; mne.set_log_level("ERROR")
try:
    from mne_icalabel import label_components; HAVE_ICLABEL = True
except Exception:
    HAVE_ICLABEL = False
MONT = mne.channels.make_standard_montage("standard_1020")
FRONTAL = [c for c in CH if c.upper() in ("FP1", "FP2", "AF3", "AF4", "AF7", "AF8", "FPZ")]
print(f"ICLabel available: {HAVE_ICLABEL} | frontal proxies: {FRONTAL}")

def make_raw(win):
    dat = np.transpose(win, (1, 0, 2)).reshape(len(CH), -1).astype(np.float64)
    raw = mne.io.RawArray(dat, mne.create_info(CH, FS, ch_types="eeg"), verbose="ERROR")
    raw.set_montage(MONT, match_case=False, on_missing="ignore", verbose="ERROR")
    return raw

def fit_session_ica(fit_win):
    raw_hp = make_raw(fit_win).filter(1., None, verbose="ERROR")
    ica = mne.preprocessing.ICA(n_components=0.99, method="infomax",
                                fit_params=dict(extended=True), max_iter="auto",
                                random_state=0, verbose="ERROR")
    ica.fit(raw_hp)
    excl, ictypes = set(), []
    CAP = max(2, int(round(0.35 * ica.n_components_)))
    if HAVE_ICLABEL:
        lab = label_components(raw_hp, ica, method="iclabel")
        cand = sorted([(float(pr), i, lb) for i, (lb, pr) in
                       enumerate(zip(lab["labels"], lab["y_pred_proba"]))
                       if lb in ("eye blink", "muscle artifact") and float(pr) > 0.80], reverse=True)
        for pr, i, lb in cand[:CAP]: excl.add(i); ictypes.append((lb, round(pr, 2), i))
    else:
        for ch in FRONTAL:
            try:
                idx, _ = ica.find_bads_eog(raw_hp, ch_name=ch, verbose="ERROR")
                for i in idx: excl.add(i); ictypes.append(("eog", ch, i))
            except Exception: pass
        try:
            idx, _ = ica.find_bads_muscle(raw_hp, threshold=0.9, verbose="ERROR")
            for i in idx: excl.add(i); ictypes.append(("muscle", "", i))
        except Exception: pass
        if len(excl) > CAP: excl = set(sorted(excl)[:CAP])
    ica.exclude = sorted(excl)
    return ica, ictypes, len(ica.exclude)

def apply_ica(ica, win):
    n, C, T = win.shape
    raw_c = make_raw(win); ica.apply(raw_c, verbose="ERROR")
    return raw_c.get_data().reshape(C, n, T).transpose(1, 0, 2).astype(np.float32)

def hp_filter(win):
    n, C, T = win.shape
    return make_raw(win).filter(1., None, verbose="ERROR").get_data() \
        .reshape(C, n, T).transpose(1, 0, 2).astype(np.float32)

# ---- theta region-power for one (session,emotion), pipeline-faithful ----
def theta_regions(Xemo, sidx, emo_id, mode="stored", ica=None):
    """Replicates p6.compute_channel_bandpower's 2500-window subsample (seed=100+sidx+emo),
    optionally cleans those SAME windows, returns {region: theta_power}."""
    n = len(Xemo)
    if n > 2500:
        idx = np.random.default_rng(100 + sidx + int(emo_id)).choice(n, size=2500, replace=False)
        Xs = Xemo[idx]
    else:
        Xs = Xemo
    if mode == "filter":
        Xs = hp_filter(Xs)
    elif mode == "ica":
        Xs = apply_ica(ica, Xs)
    bp = p6.compute_channel_bandpower(Xs, FS, max_samples=10**9)   # no further subsample
    th = np.asarray(bp["theta"], float)
    return {r: float(np.mean([th[i] for i, rr in enumerate(REGIONS) if rr == r]))
            for r in REG_PRESENT}

def drift(a, b):
    rs = [r for r in REG_PRESENT if r in a and r in b]
    return float(np.mean([abs(a[r] - b[r]) for r in rs])) if rs else np.nan

# ---- fit ICA once per session (subsample across emotions) ----
def session_fit_windows(sidx, per=1200):
    X, emo = SESS[sidx]["X"], SESS[sidx]["y_emo"]; idx = []
    for e in EMOS:
        ei = np.where(emo == e)[0]
        idx += np.random.default_rng(7 + sidx).choice(ei, size=min(per, len(ei)), replace=False).tolist()
    return X[np.array(sorted(idx))]

ica_by_sess, ic_removed = {}, {}
for sidx in sorted(SESS):
    ica, ictypes, nrm = fit_session_ica(session_fit_windows(sidx))
    ica_by_sess[sidx] = ica; ic_removed[sidx] = nrm
    print(f"[session {sidx}] ICA removed {nrm} comps: {ictypes}")

# ---- theta-power tables per (session,emotion) for each stage ----
STAGES = ["stored", "filter", "ica"]
theta = {st: {} for st in STAGES}   # (sidx, emo) -> {region: power}
for sidx in sorted(SESS):
    X, emo = SESS[sidx]["X"], SESS[sidx]["y_emo"]
    for e in EMOS:
        m = emo == e
        if int(m.sum()) < 20: continue      # pipeline's skip rule
        Xe = X[m]
        theta["stored"][(sidx, e)] = theta_regions(Xe, sidx, e, "stored")
        theta["filter"][(sidx, e)] = theta_regions(Xe, sidx, e, "filter")
        theta["ica"][(sidx, e)]    = theta_regions(Xe, sidx, e, "ica", ica_by_sess[sidx])
    print(f"[session {sidx}] theta computed ({len(EMOS)} emotions)")

# ---- per-condition drift (enroll=session1 -> test session 2,3, cross-emotion) ----
def cond_table(stage):
    tb = theta[stage]; rows = []
    for st in [2, 3]:
        for ee in EMOS:
            for te in EMOS:
                if (1, ee) not in tb or (st, te) not in tb: continue
                rows.append({"test_session": st,
                             "enroll_emotion": EMO_MAP[ee], "test_emotion": EMO_MAP[te],
                             "theta_drift": drift(tb[(1, ee)], tb[(st, te)])})
    return pd.DataFrame(rows)

# ---- published ArcFace EER + paper theta_power_drift (for the faithfulness cross-check) ----
csvp = sorted(glob.glob(os.path.join(ROOT, "outputs", "**", "merged_global_psd_identity_eer.csv"),
                        recursive=True), key=len)[0]
mg = pd.read_csv(csvp)
if "variant" in mg.columns and (mg["variant"] == "arcface_supcon_cnn").any():
    mg = mg[mg["variant"] == "arcface_supcon_cnn"]
def snum(v):
    s = str(v)
    m = re.search(r'SESSION[_ ]?(\d+)', s, re.I)     # target the SESSION number, not "Q1"
    if m: return int(m.group(1))
    m = re.search(r'(\d+)', s); return int(m.group(1)) if m else -1
mg["test_session"] = mg["test_session"].map(snum)
for c in ["enroll_emotion", "test_emotion"]:
    mg[c] = mg[c].astype(str)
agg = {"EER": "mean"}
if "theta_power_drift" in mg.columns: agg["theta_power_drift"] = "mean"
eer = mg.groupby(["test_session", "enroll_emotion", "test_emotion"], as_index=False).agg(agg)

def std_beta(x, y):
    xz = (x - x.mean()) / x.std(ddof=0); yz = (y - y.mean()) / y.std(ddof=0)
    return float(np.polyfit(xz, yz, 1)[0])

# ---- assemble, self-check, report ----
base = cond_table("stored").rename(columns={"theta_drift": "theta_stored"}) \
        .merge(eer, on=["test_session", "enroll_emotion", "test_emotion"], how="inner")
for st in ["filter", "ica"]:
    base = base.merge(cond_table(st).rename(columns={"theta_drift": f"theta_{st}"}),
                      on=["test_session", "enroll_emotion", "test_emotion"], how="inner")
base = base.dropna(subset=["theta_stored", "EER"])
base.to_csv(os.path.join(OUT, "m3f_theta_conditions.csv"), index=False)
if len(base) == 0:
    st = cond_table("stored")
    print("[FATAL] merge produced 0 rows. Key mismatch between theta table and EER table.")
    print("  theta keys (sample):", st[["test_session","enroll_emotion","test_emotion"]].head(6).to_dict("records"))
    print("  EER   keys (sample):", eer[["test_session","enroll_emotion","test_emotion"]].head(6).to_dict("records"))
    print("  theta test_session uniq:", sorted(st.test_session.unique().tolist()),
          "| EER test_session uniq:", sorted(eer.test_session.unique().tolist()))
    raise SystemExit(1)
print(f"[merge] base rows = {len(base)}  (expect 50: 2 sessions x 25 emotion pairs)")

# faithfulness proof: my stored drift vs the paper's own theta_power_drift column
faith = np.nan
if "theta_power_drift" in base.columns:
    g = base[["theta_stored", "theta_power_drift"]].dropna()
    if len(g) > 3: faith = float(np.corrcoef(g["theta_stored"], g["theta_power_drift"])[0, 1])

rows = []
for st in STAGES:
    col = f"theta_{st}"
    d = base[[col, "EER"]].dropna()
    b = std_beta(d[col].values, d["EER"].values)
    rho, p = stats.spearmanr(d[col], d["EER"])
    nic = sum(ic_removed.values()) if st == "ica" else 0
    rows.append(dict(stage=st, std_beta=round(b, 3), spearman_rho=round(rho, 3),
                     spearman_p=round(float(p), 4), n=len(d), ICs_removed=nic))
sdf = pd.DataFrame(rows); sdf.to_csv(os.path.join(OUT, "m3f_beta_summary.csv"), index=False)
b0 = sdf.loc[sdf.stage == "stored", "std_beta"].iloc[0]

print("\n" + "=" * 82)
print("M3 (FAITHFUL) THETA ARTIFACT-ROBUSTNESS  -- pipeline-06 theta_power_drift -> ArcFace EER")
print("=" * 82)
for r in rows:
    dpc = "" if r["stage"] == "stored" or b0 == 0 else f"  ({100*(r['std_beta']-b0)/abs(b0):+.0f}% vs stored)"
    print(f"  {r['stage']:7s} std-beta={r['std_beta']:+.3f}  rho={r['spearman_rho']:+.3f} "
          f"p={r['spearman_p']:.3g}  n={r['n']}  ICs_removed={r['ICs_removed']}{dpc}")
print("-" * 82)
print(f"  FAITHFULNESS: corr(my stored theta_drift, paper theta_power_drift) = {faith:.3f}  (want ~0.99)")
print(f"  SELF-CHECK  : stored std-beta = {b0:+.3f}  (must be POSITIVE, ~ the paper's theta effect)")
print("  READ-OFF    : if 'ica' std-beta stays POSITIVE and close to stored => theta SURVIVES")
print("                ocular+muscle removal => NEURAL, not artifact.")

# ---- figures ----
lbl = {"stored": "Stored\n(paper)", "filter": "1 Hz HP", "ica": "ICA\ncleaned"}
betas = [r["std_beta"] for r in rows]
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.bar(range(len(betas)), betas, color=["#888", "#5b9bd5", "#2e7d32"])
ax.axhline(0, color="k", lw=.8); ax.set_xticks(range(len(rows)))
ax.set_xticklabels([lbl[r["stage"]] for r in rows], fontsize=8)
ax.set_ylabel("theta_power_drift $\\to$ EER  std-$\\beta$", fontsize=9)
ax.set_title("M3: theta correlate vs artifact removal", fontsize=9)
for i, b in enumerate(betas):
    ax.text(i, b + 0.01*np.sign(b), f"{b:+.2f}", ha="center",
            va="bottom" if b >= 0 else "top", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m3f_beta_bar.png"), dpi=300); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4))
for col, mk, cl, lb2 in [("theta_stored", "o", "#888", "stored"), ("theta_ica", "^", "#2e7d32", "ICA-cleaned")]:
    g = base[[col, "EER"]].dropna()
    ax.scatter(g[col], g["EER"], marker=mk, c=cl, s=22, alpha=.8, label=lb2)
    if len(g) > 2:
        sl = np.polyfit(g[col], g["EER"], 1); xs = np.linspace(g[col].min(), g[col].max(), 40)
        ax.plot(xs, sl[1] + sl[0]*xs, "-", c=cl, lw=1)
ax.set_xlabel("theta_power_drift  $|\\Delta\\theta|$", fontsize=9)
ax.set_ylabel("cross-session EER", fontsize=9)
ax.set_title("Theta drift vs EER: before vs after ICA", fontsize=9)
ax.legend(fontsize=8, frameon=False); fig.tight_layout()
fig.savefig(os.path.join(FIG, "m3f_scatter.png"), dpi=300); plt.close(fig)

print(f"\n[out] {os.path.relpath(OUT, ROOT)}/  ->  m3f_beta_summary.csv, m3f_theta_conditions.csv, figures/*")
print("DONE.")
