# Data

The EEG corpora used in this study are **public, third-party datasets** and are **not
redistributed** in this repository. Obtain each from its official source under its own
license / data-use agreement, then preprocess into the layout the code expects.

## Datasets and official sources

| Dataset | Description | Official access |
|---|---|---|
| **SEED-V** | SJTU emotion EEG, 16 subjects, 3 separate-day sessions, 62 ch, 5 emotions | https://bcmi.sjtu.edu.cn/home/seed/ (request access) |
| **SEED-IV** (*EmotionMeter*) | SJTU emotion EEG, 15 subjects, 3 sessions, 62 ch | https://bcmi.sjtu.edu.cn/home/seed/ (request access) |
| **AEP** (Auditory-Evoked-Potential) | Same-day auditory EEG biometric set, 20 subjects | See the AEP dataset paper (cited in the manuscript) |

> SEED-V / SEED-IV require signing SJTU BCMI's data-use agreement. Do not commit raw EEG
> or derived per-trial arrays to any public repository.

## Expected layout after preprocessing

The analysis scripts read session-wise arrays from:

```
data/processed/sessionwise/
  SEEDV_Q1_SAFE_SESSION1_16sub.npz
  SEEDV_Q1_SAFE_SESSION2_16sub.npz
  SEEDV_Q1_SAFE_SESSION3_16sub.npz
```

Each `.npz` contains:

| key | shape / type | meaning |
|---|---|---|
| `X` | `(N, 62, 400)` float32 | 2 s windows @ 200 Hz, 62 scalp channels |
| `y_subject` | `(N,)` int | subject id |
| `y_session` | `(N,)` int | session id (1–3) |
| `y_emotion` | `(N,)` int | 0=Disgust,1=Fear,2=Sad,3=Neutral,4=Happy |
| `y_trial` | `(N,)` int | stimulus/trial id (for leakage-safe splits) |
| `ch_names` | `(62,)` str | channel labels |
| `fs` | int | 200 |

## Preprocessing summary (to regenerate the arrays)

Drop non-scalp channels (M1, M2, VEO, HEO) → keep 62; common-average reference;
band-pass 0.5–45 Hz; 50 Hz notch; resample 1000→200 Hz; segment into 2 s windows with
1 s hop (2 s edge-trim per trial); per-window z-score. Implemented in `scripts/` (see
`docs/REPRODUCE.md`).

Window counts (SEED-V): S1 = 37,600 · S2 = 33,856 · S3 = 42,688 · total = 114,144.
