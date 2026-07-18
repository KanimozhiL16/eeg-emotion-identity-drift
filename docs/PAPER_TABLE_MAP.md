# Paper → code → result map

Every table, figure, and headline number in the manuscript maps to a script and a result file in
this repository. Values were byte-verified against these sources.

## Main-text tables

| Paper item | Script | Result file | Key values |
|---|---|---|---|
| Session verification (per-participant & pooled EER) | `scripts/07_seedv_session_drift_subject_adaptation.py` | `results/manuscript_tables/T1a_session_verification.csv`; per-participant recomputed from `results/run_07_.../score_level_session_verification_scores.csv` | per-participant 0.131 / 0.169 / 0.246; pooled 0.138 / 0.182 / 0.251 |
| Table II/III — matcher baselines | `scripts/step3_baselines.py`, `scripts/reve_benchmark.py` | `results/manuscript_tables/T3_baselines.csv` | PSD 0.183/0.229/0.272; EEGNet 0.011/0.190/0.245; Mahalanobis 0.318 |
| Statistics (H1–H5) | `scripts/step2_statistics.py` | `results/manuscript_tables/T2_statistics_H1_H5.csv` | Friedman 11.38, p=0.003, Kendall W=0.355; theta–EER Pearson 0.317, FDR-q 1.53e-6 |
| Capacity ablation | `scripts/14_capacity.py` | `results/manuscript_tables/T4_capacity_ablation.csv` | params 1.4k→53.6k; within→cross gap |
| Regression / biological | `scripts/06_q1_levelup_biological_embedding_analysis.py` | `results/manuscript_tables/T5a_regression_band_contributions.csv`, `T5b_biological_stats.csv` | theta standardised β ≈ +0.43 |
| Adaptation / mitigation | `scripts/16_adaptation_significance.py` | `results/manuscript_tables/T6a_adaptation_significance.csv` | rel. reduction 9.04% / 10.92%; Wilcoxon W=16/3, p=0.005/0.0002 |
| AEP cross-dataset | `scripts/10_aep_cross_dataset_validation.py` | `results/manuscript_tables/T7a_aep_summary.csv` | 20 subj, AUC 0.731, EER 0.337 |
| Table VI — positioning | (literature; not computed) | manuscript | values as reported in cited works |

## Supplementary / reviewer experiments

| Supplementary item | Script | Result file | Key values |
|---|---|---|---|
| S-M2 neural manipulation check | `scripts/manip_fig_only.py` (fast) / `scripts/manip_check_emotion_decoding.py` | `results/manip_check.log`, `results/csv/manip_check_*.csv`, `figures/manip_check_*.png` | mean 34.5%, t(15)=5.96, p=2.6e-5, d=1.49, 14/16>chance |
| S-M3 theta artifact robustness | `scripts/21_M3_theta_robustness_faithful.py` | `results/work3/21_M3_faithful/m3f_beta_summary.csv` | β +0.362 (stored) → +0.404 (ICA), faithfulness r=1.000 |
| S-M11 emotion vs time (2×2) | `scripts/32_state_vs_time.py`, `scripts/40_m11_wilcoxon_fdr.py` | `results/run_19_state_vs_time/{deltas.csv,m11_wilcoxon_fdr.csv}` | ΔEER state +0.047 / time +0.077 / both +0.100; all BH-FDR significant |
| SEED-IV replication | `scripts/09E_seediv_step3_baselines.py` | `results/work3/21_M3_faithful/...`, SEED-IV logs | PSD 0.233 → 0.324 / 0.362 |

## Integrity audit

`scripts/audit_project.py` walks the whole project and reports, for each manuscript value,
whether it is found in a result file (PASS) or missing (FLAG). All 35 headline values PASS.
