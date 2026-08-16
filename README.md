<!-- README for the IEEE TAC submission code/evidence repository -->
<div align="center">

# Emotion-Associated and Cross-Session Identity Drift in EEG Biometrics
### Measurement, an Exploratory Spectral Correlate, and Lightweight Mitigation

Reproducibility and evidence repository for the manuscript submitted to
**IEEE Transactions on Affective Computing (TAC)**.

**Kanimozhi L.** and **S. Shridevi** · Centre for Neuroinformatics, VIT Chennai

</div>

---

## Overview

This repository contains the complete analysis code, per-experiment result tables, logs, and
figures behind the paper. The study measures how the **elicited-emotion condition** and
**recording-session mismatch** degrade the stability of EEG-based identity representations,
attaches an **exploratory theta-band spectral correlate** (condition-level, not
participant-general), tests whether a foundation-scale EEG encoder removes the degradation, and
evaluates a **lightweight labelled target-session re-enrolment** mitigation — all under a strict,
leakage-controlled *enrol-then-verify* protocol.

Every numerical claim in the manuscript is traceable to a result file in [`results/`](results/);
the mapping is given in [`docs/PAPER_TABLE_MAP.md`](docs/PAPER_TABLE_MAP.md) and
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Key findings (all reproducible from this repo)

| Finding | Result |
|---|---|
| Per-participant EER rises across sessions | 0.131 (within) → 0.169 (S1→S2) → 0.246 (S1→S3); Friedman p=0.003 |
| Global pooled EER (same ordering) | 0.138 / 0.182 / 0.251 |
| Leakage-free trial-disjoint 2×2 (session vs emotion) | session ΔEER **+0.062** (β=+0.062, p=5×10⁻⁴); emotion ΔEER **+0.031** — significant in paired/FDR tests but **marginal in the participant-aware mixed model** (p=0.086); cells 0.170/0.201/0.232/0.252 |
| Emotion effect not explained by film clip | same-emotion clip-change baseline 0.174 vs different-emotion 0.195 (paired W=18, p=0.008) |
| Foundation model does not close the gap | frozen-cosine REVE degrades (0.251/0.313/0.308); a supervised linear probe recovers most (0.31 → 0.10–0.11), with read-out choice substantially affecting performance |
| Exploratory spectral correlate | theta-band drift predicts EER (standardised β ≈ +0.43, p=0.004); concentrated in the larger S1→S3 mismatch (not a transition-general law); survives a **session-level** ICA ocular+muscle control (+0.36 → +0.40); **not** robust to participant clustering (CI spans zero) |
| Manipulation check (label validity) | raw 5-way emotion decoding 34.5% vs 20% chance (t(15)=5.96, p=2.6×10⁻⁵); imbalance-aware **balanced accuracy 0.325**, significant in **12/16** participants (200-fold trial-level permutation) |
| Lightweight mitigation | labelled **target-session** re-enrolment (fraction f=0.2) lowers **remaining target-session** EER by ≈9–11% (S2 0.216→0.197, −9.0%; S3 0.237→0.211, −10.9%; p<0.01) |
| External replications | separate-day SEED-IV (PSD+cosine 0.233→0.324→0.362, n=15) and same-day auditory-EEG/AEP (0.19→0.32, n=20) reproduce the within-to-cross collapse |

## Repository structure

```
.
├── README.md                     # this file
├── LICENSE                       # MIT (code); datasets are third-party — see data/README.md
├── CITATION.cff                  # how to cite this work
├── requirements.txt              # exact pinned Python environment (from `pip freeze` on the box)
├── environment.yml               # conda environment
├── scripts/                      # all analysis code
├── results/                      # per-experiment result tables, logs, and metrics (the evidence)
│   └── manuscript_tables/        # CSVs that map 1:1 to the paper's tables
├── figures/                      # main-text and supplementary figures
├── supplementary/                # supplementary material (S2 manip-check, S-M3 ICA, S-M11 2×2, S-C, S-TT, S-CLIP)
├── data/
│   └── README.md                 # how to obtain SEED-V, SEED-IV, AEP (not redistributed here)
└── docs/
    ├── REPRODUCE.md              # step-by-step reproduction + paper→code→output map
    └── PAPER_TABLE_MAP.md        # paper table/figure → script → result file
```

## Datasets

This work uses **public, third-party EEG corpora**; raw data are **not redistributed** here. See
[`data/README.md`](data/README.md) for official download links and the preprocessing this repo expects.

| Dataset | Role | Subjects | Sessions |
|---|---|---|---|
| SEED-V (SJTU) | primary | 16 | 3 (separate days) |
| SEED-IV (SJTU, *EmotionMeter*) | separate-day replication | 15 | 3 |
| Auditory-Evoked-Potential (AEP) | same-day replication | 20 | 1 |

**Preprocessing (SEED-V/IV):** 62 scalp channels, 50 Hz notch then band-pass 0.5–45 Hz,
common-average reference, resample 1000→200 Hz, 2 s windows with 1 s hop (50% overlap) → arrays of
shape `(N, 62, 400)`.

**Normalisation (important — reviewer-relevant):** the stored windows are
**amplitude-preserving** — mean-centred but **not** variance-normalised (mean per-window,
per-channel SD ≈ 0.88; CV ≈ 0.06). Variance normalisation is applied **only** downstream, at the
**feature** level: the 434-D matcher descriptor is standardised using **enrolment-set** statistics
— *not* by per-window z-scoring — so the per-window standard-deviation feature survives as a
genuine (non-constant) descriptor feature. The interpretive/spectral branch uses raw (unnormalised)
band powers.

## Reproducing the results

Full instructions and the paper→code→output map are in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).
In brief:

```bash
# 1. environment
conda env create -f environment.yml && conda activate p4_seedv
#    (or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt)

# 2. obtain + preprocess the datasets (see data/README.md) into data/processed/sessionwise/

# 3. baselines + session drift
python scripts/step3_baselines.py                     # matcher baselines (cross-session table)

# 4. canonical leakage-free, whole-trial-disjoint reanalyses
python scripts/r4rev_reanalysis.py                    # 2×2 (R2), theta-transition (R3), manip check (R4)
python scripts/r4rev_finalize.py                      # participant-aware MixedLM + FDR + theta transition table
python scripts/r4rev_R1_capacity.py                   # leakage-free capacity ablation (Table S-C)
python scripts/r4_clip_control.py                     # clip-controlled affect (Table S-CLIP)

# 5. spectral correlate + ICA robustness
python scripts/06_q1_levelup_biological_embedding_analysis.py
python scripts/align_repo_spectral_tables.py          # seed-collapsed OLS → paper-aligned band table
python scripts/21_M3_theta_robustness_faithful.py     # theta correlate + session-level ICA control

# 6. mitigation + external replication
python scripts/10_aep_cross_dataset_validation.py     # AEP same-day replication
```

> **Deprecated:** `scripts/32_state_vs_time.py` (random-window split; older +0.047/+0.077 effects)
> is superseded by `r4rev_reanalysis.py` + `r4rev_finalize.py` and should not be used.

A one-shot integrity audit reconciles headline numbers against the result files:

```bash
python scripts/audit_project.py            # prints PASS/FLAG per manuscript value
```

> Note: the audit is being upgraded to check named CSV **result** fields directly (rather than
> string-matching README/source), so a value cannot pass unless it appears in a genuine result file.

## Environment

Analyses were run on an NVIDIA A100 (conda env `p4_seedv`). The **authoritative** pinned versions
live in [`requirements.txt`](requirements.txt) / [`environment.yml`](environment.yml); regenerate
them on the analysis machine with `pip freeze` / `conda env export --no-builds` so the README, the
lockfiles, and the actual runtime always agree.

## Citation

If you use this code or the analysis, please cite the paper (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{kanimozhi_eeg_drift_2026,
  title   = {Emotion-Associated and Cross-Session Identity Drift in EEG Biometrics:
             Measurement, an Exploratory Spectral Correlate, and Lightweight Mitigation},
  author  = {Kanimozhi, L. and Shridevi, S.},
  journal = {IEEE Transactions on Affective Computing},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code is released under the **MIT License** ([`LICENSE`](LICENSE)). The EEG datasets are governed by
their respective owners' licenses/agreements — see [`data/README.md`](data/README.md).

## Contact

Kanimozhi L. — kanimozhi.l2024@vitstudent.ac.in — Centre for Neuroinformatics, VIT Chennai.
