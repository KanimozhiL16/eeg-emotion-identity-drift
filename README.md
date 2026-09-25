<!-- README for the IEEE Access submission code/evidence repository -->
<div align="center">

# Leakage-Controlled Evaluation of Session- and Elicitation-Condition-Related Identity Drift in EEG Biometrics
### A Two-Dataset Dissociation with Clip-Level Control

Reproducibility and evidence repository for the manuscript under review at
**IEEE Access**.

**Kanimozhi L.** and **S. Shridevi** · Centre for Neuroinformatics, VIT Chennai

</div>

---

## Overview

This repository contains the complete analysis code, per-experiment result tables, logs, and
figures behind the paper. Under a strict, leakage-controlled *enrol-then-verify* protocol, the
study separates the **session-mismatch** and **elicited-emotion (elicitation-condition)**
components of EEG identity drift, replicates the dissociation on **two multi-session
emotional-EEG datasets** (SEED-V and SEED-IV), adds a **same-emotion clip-change control**,
tests whether added encoder capacity closes the gap, evaluates a representative
**unsupervised domain-adaptation baseline (CORAL)**, and tests a **lightweight labelled
target-session re-enrolment** mitigation. An exploratory **theta-band spectral correlate**
(condition-level, session-confounded, not participant-general) is reported as exploratory only.

Every numerical claim in the manuscript is traceable to a result file in [`results/`](results/);
the mapping is given in [`docs/PAPER_TABLE_MAP.md`](docs/PAPER_TABLE_MAP.md) and
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Key findings (all reproducible from this repo)

| Finding | Result |
|---|---|
| Per-participant EER rises across sessions (SEED-V) | 0.131 (within) → 0.169 (S1→S2) → 0.246 (S1→S3); Friedman p=0.003 |
| Global pooled EER (same ordering) | 0.145 / 0.185 / 0.253 |
| SEED-V leakage-free trial-disjoint 2×2 | session ΔEER **+0.062** (p=5×10⁻⁴); emotion ΔEER **+0.031** — significant in paired/FDR tests but **marginal in the participant-aware mixed model** (p=0.086) |
| SEED-IV replication of the 2×2 (n=15) | session ΔEER **+0.102** (p=6.1×10⁻⁵, 15/15 worse); emotion ΔEER **+0.026** (p=6.1×10⁻⁴, 13/15 worse); both +0.112 |
| Emotion effect exceeds a same-emotion clip-change baseline | SEED-IV clip control 0.245 → 0.271 (ΔEER +0.026, paired p=0.015, 11/15 worse); does not claim to isolate affect from all stimulus properties |
| Added capacity does not close the cross-session gap | within the tested convolutional-encoder (CNN) family and fixed cosine read-out (EEGNet within-session ≈0.01–0.03, cross-session ≈0.19–0.25 across a ~38× parameter range) |
| Unsupervised domain adaptation does not close it either | CORAL (SEED-IV): S1→S2 0.320→0.358 (+0.038); S1→S3 0.364→0.467 (+0.103) — worse, an honest negative baseline |
| Exploratory spectral correlate | theta-band drift predicts EER (standardised β ≈ +0.43, p=0.004); survives a **session-level** ICA ocular+muscle control (+0.36 → +0.40); **not** robust to participant clustering (CI spans zero) — exploratory only |
| Manipulation check (label validity) | balanced accuracy 0.325 vs 0.25 chance; raw 5-way decoding 34.5% vs 20% (t(15)=5.96, p=2.6×10⁻⁵); significant in 12/16 participants |
| Lightweight mitigation | labelled **target-session** re-enrolment recovers part of the lost accuracy |
| External replication | separate-day SEED-IV (PSD+cosine 0.233→0.324→0.362, n=15) and same-day auditory-EEG/AEP (0.19→0.32, n=20) reproduce the within-to-cross collapse |

## Repository structure

```
.
├── README.md                     # this file
├── LICENSE                       # MIT (code); datasets are third-party — see data/README.md
├── CITATION.cff                  # how to cite this work
├── requirements.txt              # exact pinned Python environment
├── environment.yml               # conda environment
├── scripts/                      # all analysis code (incl. seediv_repro/, seediv_coral_baseline.py)
├── results/                      # per-experiment result tables, logs, and metrics (the evidence)
├── figures/                      # main-text and supplementary figures
├── supplementary/                # supplementary material
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
| SEED-IV (SJTU, *EmotionMeter*) | second same-laboratory dataset | 15 | 3 (separate days) |
| Auditory-Evoked-Potential (AEP) | independent cross-source check | 20 | 1 (same day) |

**Preprocessing (SEED-V/IV):** 62 scalp channels, 50 Hz notch then band-pass 0.5–45 Hz,
common-average reference, resample 1000→200 Hz, 2 s windows with 1 s hop (50% overlap).

## Reproducing the results

Full instructions and the paper→code→output map are in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

```bash
# environment
conda env create -f environment.yml && conda activate p4_seedv
#   (or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt)

# baselines + session drift, canonical leakage-free reanalyses, spectral correlate,
# SEED-IV 2×2 + clip control, CORAL baseline, and external replication:
python scripts/step3_baselines.py
python scripts/r4rev_reanalysis.py
python scripts/r4rev_finalize.py
python scripts/r4rev_R1_capacity.py
python scripts/r4_clip_control.py
python scripts/seediv_2x2.py
python scripts/seediv_coral_baseline.py
python scripts/10_aep_cross_dataset_validation.py
```

## Citation

If you use this code or the analysis, please cite the paper (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{kanimozhi_eeg_drift_2026,
  title   = {Leakage-Controlled Evaluation of Session- and Elicitation-Condition-Related
             Identity Drift in EEG Biometrics: A Two-Dataset Dissociation with Clip-Level Control},
  author  = {Kanimozhi, L. and Shridevi, S.},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code is released under the **MIT License** ([`LICENSE`](LICENSE)). The EEG datasets are governed by
their respective owners' licenses/agreements — see [`data/README.md`](data/README.md).

## Contact

Kanimozhi L. — kanimozhi.l2024@vitstudent.ac.in — Centre for Neuroinformatics, VIT Chennai.
