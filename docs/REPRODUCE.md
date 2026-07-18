# Reproduction guide

This guide reproduces every result in the paper from raw data to final tables/figures.

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.13.x
pip install -r requirements.txt
```

Reference environment: Python 3.13.12, NumPy 2.5.1, SciPy 1.18.0, scikit-learn 1.9.0,
PyTorch 2.11.0 (CUDA 13.0), MNE-Python 1.12.1. A CUDA GPU is used for the deep encoders;
the spectral/statistics analyses run on CPU.

## 1. Obtain and preprocess data

Follow [`../data/README.md`](../data/README.md) to download SEED-V, SEED-IV, and AEP, and to
produce the session-wise arrays in `data/processed/sessionwise/`. Verify integrity:

```bash
python scripts/00_verify_uploaded_npz.py
```

## 2. Run the pipeline

Each stage writes to `results/`. Order:

```bash
# baselines & protocol (Table II/III, session EER)
python scripts/step3_baselines.py
python scripts/07_seedv_session_drift_subject_adaptation.py

# foundation encoder (frozen cosine vs supervised linear probe)
python scripts/reve_benchmark.py

# spectral / biological analysis (theta correlate, Table V)
python scripts/06_q1_levelup_biological_embedding_analysis.py

# statistics (Friedman, FDR, change-point audit)
python scripts/step2_statistics.py

# capacity ablation
python scripts/14_capacity.py

# mitigation (single-sample re-enrolment)
python scripts/16_adaptation_significance.py

# emotion-vs-time decomposition + W/FDR  (Supp. S-M11)
python scripts/32_state_vs_time.py
python scripts/40_m11_wilcoxon_fdr.py
python scripts/41_state_vs_time_figure.py

# theta artifact robustness — ICA ocular+muscle  (Supp. S-M3)
python scripts/21_M3_theta_robustness_faithful.py

# neural label-validity manipulation check  (Supp. S-M2)
python scripts/manip_fig_only.py

# external replications
python scripts/10_aep_cross_dataset_validation.py    # AEP (same-day)
python scripts/09E_seediv_step3_baselines.py         # SEED-IV (separate-day)
```

## 3. Verify against the manuscript

```bash
python scripts/audit_project.py | tee results/AUDIT_REPORT.txt
```

This reconciles every headline number in the paper against the generated result files and prints
`OK` (found) or `**FLAG` (missing) for each. All 35 values should report `OK`.

## Notes on determinism / long runs

- Random seeds are fixed in each script; the permutation null in the manipulation check is the only
  slow step (~20 min) and is **not** required for the reported significance (which comes from a
  one-sample t-test) — use `scripts/manip_fig_only.py` to skip it.
- Long jobs on a shared server: run inside `tmux` or with `nohup ... &` from the project root so a
  closed terminal does not kill them.
