# BREV work3 experiment pack — all runs to lift the SEED-V drift paper to Q1

These scripts address the 9 supervisor comments (work3, 14 Jul 2026) that need an A100 re-run.
They **reuse the existing pipeline** (`data/processed/sessionwise/*.npz`, `outputs/**/merged_global_psd_identity_eer.csv`, `scripts/07_seedv_session_drift_subject_adaptation.py`) — no result is fabricated; every number is recomputed.

## 0. One-time setup on Brev
```bash
cd /home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC   # or /lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
source p4_seedv_env/bin/activate
pip install -q statsmodels scikit-learn --break-system-packages
mkdir -p scripts/work3 outputs/work3
# copy the 7 scripts below into scripts/work3/  (paste-in heredoc if upload is flaky)
```

## Run order (safe → heavier)
| Step | Script | Comment(s) | Input | Runtime |
|------|--------|-----------|-------|---------|
| 1 | `10_csv_mechanism_checks.py` | C9, C10, C18, C19, C8(if multi-variant) | merged CSV only | seconds |
| 2 | `11_band_recompute_from_windows.py` | **C12**, C9(delta), C10 | sessionwise windows + merged CSV | ~10 min |
| 3 | `12_subject_theta_vs_eer.py` | **C20** | sessionwise windows | ~10 min |
| 4 | `13_matcher_theta_consistency.py` | **C8** | sessionwise windows + merged CSV | ~15 min |
| 5 | `14_theta_aware_mitigation.py` | **C15** | sessionwise windows | ~15 min |
| 6 | `15_reve_linear_probe.py` | C7 | sessionwise windows + REVE | ~20 min (GPU) |
| 7 | `16_aep_spectral_theta.py` | C14 | auditory dataset | ~5 min |

Steps 1 is zero-risk (CSV only). Steps 2–7 recompute from raw windows; **run step 1 first** — if the merged CSV already has a `delta_power_drift`/`total_psd_drift` column and multiple `variant` values, several sub-points are answered without any window recompute.

`bash run_all.sh` runs 1–7 in order and tees each log to `outputs/work3/`.

## What each proves (the READ-OFF)
- **C12** (step 2): if theta stays the top positive band predictor on **amplitude-preserving** PSD (no per-window z-score), the spectral finding is real, not a z-score artefact → keep "band-power drift". If it flips, we rename to "spectral redistribution".
- **C8** (step 4): if theta β>0 with p<0.05 for PSD-cosine, lightweight and ArcFace matchers, theta generalises across matchers → the headline claim holds.
- **C20** (step 3): if Spearman(Δθ_i, EER_i) is positive/significant across subjects, theta links to identity at the person level → strongest new evidence.
- **C18** (step 1): cross-validated MAE/RMSE/predictive-R² justify the word "predictable"; if predictive-R²≤0, downgrade to "systematically associated".
- **C19** (step 1): Spearman + isotonic confirm monotonic (not just "no change-point").
- **C9/C10** (steps 1–2): theta survives adding delta + a broadband/total-drift control, and survives σ-normalising the drift index.
- **C15** (step 5): theta-aware normalisation should beat, or match, generic re-enrolment → connects mechanism to mitigation.
- **C7** (step 6): a linear probe on frozen REVE is the fair test; report frozen-cosine vs linear-probe vs (optional) fine-tune.
- **C14** (step 7): does theta (T7) drift predict EER on the auditory set → "generalisation" vs "partial replication".

## After the runs
Send me the `outputs/work3/*.csv` + console logs. I will paste the verified numbers into `SR_FINAL/main.tex` (new Table/paragraph per comment), recompile, and update the second brain. **No number goes into the paper until it is in one of these output CSVs.**
