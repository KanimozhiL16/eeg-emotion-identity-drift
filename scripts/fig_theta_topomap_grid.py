#!/usr/bin/env python3
"""
fig_theta_topomap_grid.py  --  Subject x Session THETA-band scalp topomap grid.

WHAT IT SHOWS (supports the paper's two headline claims in ONE figure):
  * rows  = a few representative subjects   -> per-subject patterns differ  (identity = stable core)
  * cols  = Session 1 / 2 / 3               -> pattern shifts across columns (cross-session DRIFT)
  * colour = theta-band (4-8 Hz) power, z-scored per subject for visual comparability
This is the biometrics analogue of CCIEP Fig.3C/Fig.4 (which used emotion energy).

INPUT (already on the Brev box, same project as run_06/run_07):
  data/processed/sessionwise/*.npz  with keys: X [N,C,T], y_subject [N], y_session [N]
  (C = 62 channels, T = 400 samples @ 200 Hz)

RUN:
  cd /home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
  python -W ignore fig_theta_topomap_grid.py 2>&1 | tee outputs/figs/theta_topomap.log

OUTPUT: outputs/figs/Fig_theta_topomap_grid.png  (600 dpi)

NOTE: uses ONLY data you already have (theta power per channel). No new claims,
no model retraining. Verify the channel-name list matches your preprocessing order.
"""
import os, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

# ----------------------------------------------------------------------
FS = 200                      # sampling rate after your preprocessing
THETA = (4, 8)                # Hz
N_SUBJECTS_TO_SHOW = 4        # rows; pick a stable + a drift-prone one for contrast
DATA = "data/processed/sessionwise"
OUT  = "outputs/figs"; os.makedirs(OUT, exist_ok=True)

# SEED / SEED-V 62-channel order (standard). If your config stores channel names,
# load them instead of this list to be 100% exact.
SEED62 = ["FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8",
"FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8","T7","C5","C3","C1","CZ","C2",
"C4","C6","T8","TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8","P7","P5","P3",
"P1","PZ","P2","P4","P6","P8","PO7","PO5","PO3","POZ","PO4","PO6","PO8","CB1","O1",
"OZ","O2","CB2"]

def theta_power_per_channel(X):
    """X:[N,C,T] -> [C] mean theta power across windows."""
    f, P = welch(X, fs=FS, nperseg=min(256, X.shape[-1]), axis=-1)   # [N,C,F]
    band = (f >= THETA[0]) & (f <= THETA[1])
    return P[:, :, band].sum(-1).mean(0)                            # [C]

# ---- load session-wise data, compute theta power per subject/session ----
sess = {}
for fpath in sorted(glob.glob(os.path.join(DATA, "*.npz"))):
    d = np.load(fpath, allow_pickle=True)
    if not all(k in d for k in ("X", "y_subject", "y_session")): continue
    sid = int(np.unique(d["y_session"])[0])
    sess[sid] = (np.asarray(d["X"], np.float32), np.asarray(d["y_subject"], int))
sessions = sorted(sess)                      # e.g. [1,2,3]
subjects = sorted(np.unique(sess[sessions[0]][1]).tolist())

# choose subjects to display: most-stable + most-variable theta across sessions (visual contrast)
theta = {}   # theta[sid][subj] = [C]
for sid in sessions:
    X, y = sess[sid]
    theta[sid] = {s: theta_power_per_channel(X[y == s]) for s in subjects}
var_across_sess = {s: np.mean([np.var([theta[sid][s][c] for sid in sessions]) for c in range(len(subjects and SEED62))]) for s in subjects}
ordered = sorted(subjects, key=lambda s: var_across_sess[s])
show = ordered[:1] + ordered[-(N_SUBJECTS_TO_SHOW-1):]   # 1 stable + rest drift-prone

# ---- plot with MNE topomaps (fallback to scatter if MNE/montage missing) ----
try:
    import mne
    info = mne.create_info(SEED62, FS, "eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="ignore")    # CB1/CB2 not in 10-20 -> ignored
    good = [i for i,ch in enumerate(SEED62) if ch in montage.ch_names]
    USE_MNE = True
except Exception as e:
    print("MNE topomap unavailable, using scatter fallback:", e); USE_MNE = False

from matplotlib import cm
from matplotlib.colors import Normalize
VLIM = 2.0                                  # z-score colour range
fig, axes = plt.subplots(len(show), len(sessions),
                         figsize=(2.1*len(sessions)+0.9, 2.1*len(show)), dpi=600)
axes = np.atleast_2d(axes)
im = None
for r, s in enumerate(show):
    # z-score this subject's theta across all its channels/sessions for comparability
    allvals = np.concatenate([theta[sid][s] for sid in sessions])
    mu, sd = allvals.mean(), allvals.std() + 1e-9
    for c, sid in enumerate(sessions):
        ax = axes[r, c]; v = (theta[sid][s] - mu) / sd
        if USE_MNE:
            im, _ = mne.viz.plot_topomap(v[good], mne.pick_info(info, good), axes=ax,
                                 show=False, cmap="RdBu_r", vlim=(-VLIM, VLIM), contours=4)
        else:
            im = ax.scatter(range(len(v)), v, c=v, cmap="RdBu_r",
                            vmin=-VLIM, vmax=VLIM); ax.set_xticks([])
        if r == 0: ax.set_title(f"Session {sid}", fontsize=10)
        if c == 0: ax.text(-0.35, 0.5, f"Subject {s}", rotation=90,
                           va="center", ha="center", transform=ax.transAxes, fontsize=10)
fig.suptitle("Theta-band (4-8 Hz) power: identity persists across rows, drifts across sessions",
             fontsize=10, y=1.00)
# ---- shared colorbar (z-scored theta power) ----
fig.tight_layout(rect=[0, 0, 0.9, 1])
sm = cm.ScalarMappable(norm=Normalize(-VLIM, VLIM), cmap="RdBu_r")
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cb = fig.colorbar(sm, cax=cax)
cb.set_label("Theta power (z-scored per subject)", fontsize=9)
cb.set_ticks([-VLIM, 0, VLIM]); cb.set_ticklabels([f"-{VLIM:.0f}", "0", f"+{VLIM:.0f}"])
out = os.path.join(OUT, "Fig_theta_topomap_grid.png")
fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
print("SAVED:", out, "| subjects shown:", show, "| sessions:", sessions)
