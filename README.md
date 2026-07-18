<!-- README for the IEEE TAC submission code/evidence repository -->
<div align="center">

# Emotion-Associated and Cross-Session Identity Drift in EEG Biometrics
### Measurement, a Theta-Band Correlate, and Lightweight Mitigation

Reproducibility and evidence repository for the manuscript submitted to
**IEEE Transactions on Affective Computing (TAC)**.

**Kanimozhi L.** and **S. Shridevi** · Centre for Neuroinformatics, VIT Chennai

</div>

---

## Overview

This repository contains the complete analysis code, per-experiment result tables, logs, and
figures behind the paper. The study measures how **affective state (elicited-emotion condition)**
and **elapsed time (recording session)** degrade the stability of EEG-based identity representations,
localises the degradation to a **theta-band spectral correlate**, tests whether a foundation-scale
EEG encoder removes it, and evaluates a **lightweight single-sample re-enrolment** mitigation — all
under a strict, leakage-controlled *enrol-then-verify* protocol.

Every numerical claim in the manuscript is traceable to a result file in [`results/`](results/); the
mapping is given in [`docs/PAPER_TABLE_MAP.md`](docs/PAPER_TABLE_MAP.md).

## Key findings (all reproducible from this repo)

| Finding | Result |
|---|---|
| Per-participant EER rises across sessions | 0.131 (within) → 0.169 (S1→S2) → 0.246 (S1→S3) |
| Global pooled EER (same ordering) | 0.138 / 0.182 / 0.251 |
| Emotion-condition change vs elapsed-time change | each independently ↑ EER (ΔEER +0.047 / +0.077; BH-FDR significant) |
| Foundation model does not close the gap | frozen-cosine degrades; a supervised linear probe recovers most (0.31 → 0.10–0.11) |
| Spectral correlate | theta-band drift predicts EER (standardised β ≈ +0.43); survives ICA ocular+muscle removal (+0.36 → +0.40) |
| Manipulation check | 5-way emotion decoding 34.5% vs 20% chance (t(15)=5.96, p=2.6×10⁻⁵) |
| Lightweight mitigation | single-sample re-enrolment lowers future-session EER by ≈9–11% (p<0.01) |
| External replications | within-day auditory-EEG (AEP) and separate-day SEED-IV cohorts reproduce the collapse |

## Repository structure

```
.
├── README.md                     # this file
├── LICENSE                       # MIT (code); datasets are third-party — see data/README.md
├── CITATION.cff                  # how to cite this work
├── requirements.txt              # exact pinned Python environment
├── environment.yml               # conda environment
├── scripts/                      # all analysis code (numbered by pipeline stage)
├── results/                      # per-experiment result tables, logs, and metrics (the evidence)
│   └── manuscript_tables/        # T1–T7 CSVs that map 1:1 to the paper's tables
├── figures/                      # main-text and supplementary figures
├── supplementary/                # supplementary material (S-M2 manip-check, S-M3 ICA, S-M11 2×2)
├── data/
│   └── README.md                 # how to obtain SEED-V, SEED-IV, AEP (not redistributed here)
└── docs/
    ├── REPRODUCE.md              # step-by-step reproduction
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

Preprocessing (SEED-V/IV): 62 scalp channels, band-pass 0.5–45 Hz, 50 Hz notch, common-average
reference, resample 1000→200 Hz, 2 s windows with 1 s hop, per-window z-score → arrays of shape
`(N, 62, 400)`.

## Reproducing the results

Full instructions are in [`docs/REPRODUCE.md`](docs/REPRODUCE.md). In brief:

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. obtain + preprocess the datasets (see data/README.md) into data/processed/sessionwise/

# 3. run the pipeline (each stage writes to results/)
python scripts/step3_baselines.py                 # matcher baselines (Table II/III)
python scripts/reve_benchmark.py                  # frozen foundation encoder + linear probe
python scripts/32_state_vs_time.py                # emotion vs time decomposition (Table, 2×2)
python scripts/40_m11_wilcoxon_fdr.py             # W + BH-FDR for the 2×2 contrasts
python scripts/21_M3_theta_robustness_faithful.py # theta correlate + ICA artifact robustness
python scripts/manip_fig_only.py                  # neural label-validity manipulation check
```

A one-shot integrity audit that reconciles every headline number against the result files:

```bash
python scripts/audit_project.py            # prints PASS/FLAG per manuscript value
```

## Environment

Analyses were run with **Python 3.13.12**, NumPy 2.5.1, SciPy 1.18.0, scikit-learn 1.9.0,
PyTorch 2.11.0 (CUDA 13.0), and MNE-Python 1.12.1 on an NVIDIA A100. The exact pinned set is in
[`requirements.txt`](requirements.txt).

## Citation

If you use this code or the analysis, please cite the paper (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{kanimozhi_eeg_drift_2026,
  title   = {Emotion-Associated and Cross-Session Identity Drift in EEG Biometrics:
             Measurement, a Theta-Band Correlate, and Lightweight Mitigation},
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
