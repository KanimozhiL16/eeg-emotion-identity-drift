# Reproducing the results

**Paper:** *Emotion-Associated and Cross-Session Identity Drift in EEG Biometrics: Measurement, an Exploratory Spectral Correlate, and Lightweight Mitigation* (SEED-V; IEEE TAFFC submission).

This document maps every reported table/figure to the exact script and output file that
produces it, gives the run order, and lists the environment. It replaces the previous
`REPRODUCE.md`, which referenced scripts that no longer exist and described the
**superseded** (pre-correction) analyses.

> **Legend.** ✓ = script/output confirmed. ⚠ = **confirm the exact filename/output path on
> the analysis box before tagging `v1.0`** (the analysis exists; the path is to be verified
> against your box, not guessed here).

---

## 0. Canonical vs superseded scripts (read first)

The **leakage-free, whole-trial-disjoint** analyses are the canonical results. Some older
scripts in earlier releases used *random-window* splits (which can place 50%-overlapping
neighbours in both enrolment and test) and reported older effect sizes. Those are
**deprecated** and must be removed or clearly marked in the repository:

| Deprecated script | Problem | Replaced by (canonical) |
|---|---|---|
| `32_state_vs_time.py` | ignores `y_trial`; random split; reports old +0.047 / +0.077; 75-row OLS | `r4rev_reanalysis.py` (R2) + `r4rev_finalize.py` ✓ |
| old `07_seedv_session_drift_subject_adaptation.py` feature block | 434-D = mean/SD/**RMS**/**4 bands**/FFT (≠ paper) | canonical descriptor = 62×(mean, SD, **5 bands**) — see **Known Issue 1** |
| random-window capacity ablation | one run, random split | `r4rev_R1_capacity.py` (whole-trial-disjoint, 3 seeds) ✓ |
| random-window adaptation | random split | trial-disjoint adaptation run — see **Known Issue 3** ⚠ |

---

## 1. Environment

```bash
conda env create -f environment.yml
conda activate p4_seedv
# exact pins: conda env export --no-builds > environment.lock.yml   (run on the box)
```

Deep-encoder / EEGNet / REVE baselines need PyTorch (+GPU recommended). Every other
analysis (PSD+cosine, spectral regression, mixed model, mitigation) runs CPU-only.

## 2. Data

SEED-V is obtained from the BCMI Laboratory, SJTU (not redistributed here). The auditory
cross-dataset cohort is the PhysioNet Auditory-evoked-potential EEG-Biometric dataset
v1.0.0 (`https://physionet.org/content/auditory-eeg/1.0.0/`). Place the preprocessed
session arrays at:

```
data/processed/sessionwise/SEEDV_Q1_SAFE_SESSION{1,2,3}_16sub.npz
```

Each `.npz` provides: `X` (N×62×400, float32), `y_subject`, `y_session`, `y_trial`,
`y_emotion`, `fs` (200), `ch_names` (62). Windows are **amplitude-preserving** (mean-centred,
non-unit variance; per-window SD ≈ 0.88). Variance normalisation is applied **only** at the
434-D feature level using **enrolment-set** statistics — *not* per-window z-scoring. (Update
`README.md` and `data/README.md` to say this; the old text saying "per-window z-scored" is
incorrect and is one of the reviewer's reproducibility findings.)

## 3. Paper → code → output map

### Main text

| Item | Produces | Script | Output |
|---|---|---|---|
| Table I (positioning) | literature values only | — (no code) | — |
| Table (`tab:base`) — PSD+cosine row (0.183 / 0.229 / 0.272) | ✓ | `step3_baselines.py` (Welch `nperseg=200`, log, 5 bands, ≤400 win/subj) | `outputs/**/step3_baselines*.csv` ⚠path |
| Table (`tab:base`) — EEGNet / lightweight-prototype / REVE rows | ⚠ | EEGNet: `R_STAT3_eegnet_multiseed.py`; REVE frozen + linear probe: `M4_foundation_finetune_baseline.py` | ⚠ confirm CSVs |
| Fig. 2 (session drift, per-participant) | ✓ | `S1_reverse_enrolment.py` / `step3_baselines.py` | per-subject EER CSV ⚠path |
| Leakage-free 2×2 (Fig. S-M11; cells 0.170/0.201/0.232/0.252; Δ +0.062 sess, +0.031 emo) | ✓ | `r4rev_reanalysis.py` (**R2**) | `outputs/run_reanalysis/R2_2x2_persubject.csv`, `R2_2x2_trialdisjoint_percell.csv` |
| 2×2 participant-aware MixedLM (β_sess +0.062, β_emo +0.031 p=0.086, ICC 0.80) + Wilcoxon/FDR | ✓ | `r4rev_finalize.py` | prints model + FDR (from R2 CSV) |
| Table (`tab:band`) — spectral bands, theta β=+0.43, p=0.004 | ✓ | `06_q1_levelup_biological_embedding_analysis.py` → seed-collapse `align_repo_spectral_tables.py` | `outputs/**/merged_global_psd_identity_eer.csv`; `manuscript_assets/tables/T5a_regression_band_contributions.csv` |
| Change-point (no abrupt break; n=250 vs n=50) | ✓ | `05_q1_psd_biological_drift_analysis.py` ⚠ | change-point CSV ⚠path |
| Mitigation — Table V (S2 0.216→0.197 −9.0% p=0.006; S3 0.237→0.211 −10.9% p=2e-4; **f=0.2**) | ✓ | trial-disjoint adaptation run — see Known Issue 3 | `outputs/run_16_adaptation_significance/adaptation_significance.csv` |
| Cross-dataset replication (SEED-IV 0.233→0.362; AEP 0.19→0.32) | ✓ | AEP: `10_aep_cross_dataset_validation.py`; SEED-IV: same protocol on SEED-IV arrays ⚠ | `outputs/**` ⚠path |

### Supplementary

| Item | Script | Output |
|---|---|---|
| Table S2 — per-subject balanced accuracy + trial-permutation p (mean 0.325; 12/16 sig) | `r4rev_reanalysis.py` (**R4**) ✓ | `outputs/run_reanalysis/R4_manipcheck_balanced.csv` |
| Table S-C — capacity ablation, leakage-free (5 encoders, 3 seeds) | `r4rev_R1_capacity.py` ✓ | `outputs/**/capacity_leakagefree*.csv` ⚠path |
| Table S-TT — theta transition-adjusted (β+95% CI; S1→S3 p=6.6e-5; S1→S2 NS) | `r4rev_finalize.py` ✓ | prints from `R3_theta_conditions.csv` |
| Table S-M3 / Fig. S-M3 — ICA artifact-robustness (β +0.36→+0.40; 3 session-level fits) | `21_M3_theta_robustness_faithful.py` ✓ | `outputs/work3/21_M3_faithful/m3f_beta_summary.csv`, `m3f_theta_conditions.csv` |
| Table S-CLIP — clip-controlled affect (B 0.174 vs C 0.195; W=18, p=0.008) | `r4_clip_control.py` ✓ | `outputs/run_clipctrl/` |
| Table S-ADAPT — trial-disjoint adaptation-fraction sweep (knee f=0.1) | trial-disjoint adaptation run ⚠ | `tables/subject_adaptation_TRIALDISJOINT.csv` ⚠ |
| Table S-STAT1 — participant-clustered bootstrap (CI spans zero) | `R_STAT2_participant_cluster_bootstrap_EER.py` ✓ | ⚠path |
| Table S-STAT2 — condition-level OLS (secondary) | `R_STAT1_participant_mixedeffects.py` ⚠ | ⚠path |
| Table S-REP — SEED-IV + AEP replication | see main cross-dataset row | ⚠path |

## 4. Run order (from repository root, env active)

```bash
# 1. Baselines + session-drift (Table tab:base, Fig. 2)
python step3_baselines.py
python S1_reverse_enrolment.py

# 2. Leakage-free reanalyses (canonical 2x2, theta-transition, manip check)
python r4rev_reanalysis.py            # -> outputs/run_reanalysis/{R2_*,R3_*,R4_*}.csv
python r4rev_finalize.py              # -> MixedLM + FDR + theta transition table
python r4rev_R1_capacity.py           # -> leakage-free capacity ablation (Table S-C)
python r4_clip_control.py             # -> clip-controlled affect (Table S-CLIP)

# 3. Spectral correlate + ICA robustness (Table tab:band, S-M3)
python 06_q1_levelup_biological_embedding_analysis.py
python align_repo_spectral_tables.py  # seed-collapsed OLS -> paper-aligned band table
python 21_M3_theta_robustness_faithful.py

# 4. Mitigation + cross-dataset replication (Table V, S-ADAPT, S-REP)
#    <trial-disjoint adaptation script>   # Known Issue 3 - confirm exact name
python 10_aep_cross_dataset_validation.py
```

## 5. Known issues to close before `git tag v1.0`

1. **Feature descriptor (reviewer concern 1).** The paper's canonical 434-D descriptor is
   `62 × (mean, SD, 5 band powers)` with Welch band power. Confirm that the released
   verification/adaptation feature builder computes exactly this, and **replace** any code
   path that instead uses `mean, SD, RMS, 4 bands, FFT/log` (the current
   `07_...adaptation.py` block). If both variants were used, standardise on one and re-run
   the affected rows.
2. **Trial-disjoint everywhere (concern 2).** Ensure every released 2×2 / capacity /
   adaptation script splits by `y_trial` (whole trials), not random windows. The canonical
   scripts above already do; delete or clearly deprecate `32_state_vs_time.py` and the
   random-window variants.
3. **Adaptation fraction (concern 3).** `f=0.2` was the fraction that maximised the
   per-participant reduction (`adaptation_significance.csv`). State that it was pre-specified,
   or re-run with nested/validation-set selection; otherwise label the mitigation
   *exploratory*. Confirm the exact trial-disjoint adaptation script name/path.
4. **Docs & release.** Update `README.md` / `data/README.md` (amplitude-preserving, not
   per-window z-scored); regenerate the Fig. S-BR1 image (its picture still shows
   per-window z-scoring even though the caption was fixed); commit `environment.yml` +
   `environment.lock.yml`; then `git tag v1.0 && git push --tags`.
5. **Audit script.** Replace any string-search audit (which can match a number inside a
   README, source comment, or the audit script itself) with numeric checks that read the
   named CSV **result** fields listed in §3.

## 6. Verified anchor numbers (sanity checks after a fresh run)

| Quantity | Expected |
|---|---|
| Window counts S1/S2/S3 | 37,600 / 33,856 / 42,688 (= 114,144) |
| PSD+cosine EER (within / 1→2 / 1→3) | 0.183 / 0.229 / 0.272 |
| Trial-disjoint 2×2 cells | 0.170 / 0.201 / 0.232 / 0.252 |
| MixedLM session β / emotion β | +0.062 (p=5e-4) / +0.031 (p=0.086) |
| Theta band β (seed-collapsed) | +0.43 (p=0.004) |
| Manip-check mean balanced acc / significant | 0.325 / 12 of 16 |
| Mitigation S2 / S3 (f=0.2) | 0.216→0.197 (−9.0%) / 0.237→0.211 (−10.9%) |
| SEED-IV PSD (within/1→2/1→3) | 0.233 / 0.324 / 0.362 |

If any of these differ after a clean run, the code and paper are out of sync — fix before release.
