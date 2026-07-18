# RUN06 Q1 Level-Up Biological and Embedding Analysis Report

## What this run adds

This run strengthens the paper by adding region-level neurophysiological validation, FDR-corrected statistics, regression-based explanation of verification error, and spectral trajectory visualization. It does not overwrite previous results.

## Data and outputs

- Project root: `/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC`
- Session files: `/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC/data/processed/sessionwise`
- Identity/EER source table: `/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC/outputs/run_02_q1_validation/tables/cross_emotion_results_all_seeds.csv`
- Figures: `/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC/outputs/run_06_q1_levelup_biological_embedding_analysis/figures`
- Tables: `/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC/outputs/run_06_q1_levelup_biological_embedding_analysis/tables`

## Methodological logic

1. EEG is decomposed into theta, alpha, beta, and gamma band powers using Welch spectral estimation.
2. Band power is summarized globally and by brain region: frontal, central/motor, temporal, parietal, and occipital.
3. Cognitive-state drift is measured as the absolute spectral change between enrollment emotion and probe emotion.
4. Spectral drift is merged with the previously generated identity-drift/EER table.
5. Pearson and Spearman correlations are corrected using Benjamini-Hochberg FDR.
6. A ridge-regression model estimates how much band-specific spectral drift explains EER and identity drift.
7. PCA visualizes cognitive-state trajectories in spectral feature space.

## Scientific interpretation

The analysis tests whether EEG identity degradation is accompanied by measurable changes in neural spectral dynamics. If verification error increases together with spectral-band or region-level drift, the result supports the claim that EEG biometric identity is not a fixed point, but a cognitive-state-dependent distribution.

## Top FDR-corrected associations with EER

| level   | region        | predictor         | target   |   n |   pearson_r |   pearson_p |   spearman_r |   spearman_p |   pearson_q_fdr |   spearman_q_fdr |
|:--------|:--------------|:------------------|:---------|----:|------------:|------------:|-------------:|-------------:|----------------:|-----------------:|
| region  | Temporal      | theta_power_drift | EER      | 250 |     0.36574 |     0       |      0.30759 |      0       |         0       |          5e-05   |
| global  | ALL           | theta_power_drift | EER      | 250 |     0.31703 |     0       |      0.26749 |      2e-05   |         1e-05   |          0.00064 |
| region  | Central/Motor | theta_power_drift | EER      | 250 |     0.24838 |     7e-05   |      0.17779 |      0.00481 |         0.00126 |          0.05612 |
| region  | Occipital     | theta_power_drift | EER      | 250 |     0.23756 |     0.00015 |      0.22458 |      0.00034 |         0.00209 |          0.00805 |
| region  | Frontal       | alpha_power_drift | EER      | 250 |    -0.17832 |     0.00468 |     -0.15274 |      0.01564 |         0.04099 |          0.1243  |
| region  | Parietal      | theta_power_drift | EER      | 250 |     0.17254 |     0.00624 |      0.15225 |      0.01598 |         0.04851 |          0.1243  |
| region  | Central/Motor | gamma_power_drift | EER      | 250 |     0.16808 |     0.00774 |      0.12024 |      0.05764 |         0.05418 |          0.23733 |
| region  | Temporal      | total_psd_drift   | EER      | 250 |     0.16229 |     0.01016 |      0.10024 |      0.11389 |         0.06467 |          0.35669 |
| region  | Parietal      | alpha_power_drift | EER      | 250 |    -0.14851 |     0.01881 |     -0.13002 |      0.03995 |         0.10677 |          0.23241 |
| global  | ALL           | alpha_power_drift | EER      | 250 |    -0.14727 |     0.01983 |     -0.13807 |      0.02906 |         0.10677 |          0.18494 |
| region  | Central/Motor | beta_power_drift  | EER      | 250 |     0.13157 |     0.03763 |      0.10812 |      0.08801 |         0.16463 |          0.29337 |
| region  | Occipital     | beta_power_drift  | EER      | 250 |    -0.12438 |     0.04949 |     -0.14592 |      0.021   |         0.20379 |          0.14702 |

## Regression model summary

| level   | region        | target   |   n | model   |     alpha |   R2_in_sample |    RMSE |
|:--------|:--------------|:---------|----:|:--------|----------:|---------------:|--------:|
| global  | ALL           | EER      | 250 | RidgeCV |   46.4159 |        0.14488 | 0.008   |
| global  | ALL           | AUC      | 250 | RidgeCV |   46.4159 |        0.06194 | 0.01394 |
| region  | Frontal       | EER      | 250 | RidgeCV | 1000      |        0.01359 | 0.00859 |
| region  | Frontal       | AUC      | 250 | RidgeCV | 4641.59   |        0.00118 | 0.01438 |
| region  | Central/Motor | EER      | 250 | RidgeCV |  215.443  |        0.06389 | 0.00837 |
| region  | Central/Motor | AUC      | 250 | RidgeCV |  215.443  |        0.03994 | 0.0141  |
| region  | Temporal      | EER      | 250 | RidgeCV |   21.5444 |        0.2009  | 0.00773 |
| region  | Temporal      | AUC      | 250 | RidgeCV |   10      |        0.10007 | 0.01365 |
| region  | Parietal      | EER      | 250 | RidgeCV |  215.443  |        0.04808 | 0.00844 |
| region  | Parietal      | AUC      | 250 | RidgeCV |  215.443  |        0.02084 | 0.01424 |
| region  | Occipital     | EER      | 250 | RidgeCV |  100      |        0.10692 | 0.00817 |
| region  | Occipital     | AUC      | 250 | RidgeCV | 1000      |        0.00856 | 0.01433 |

## Standardized band-contribution coefficients

| level   | region        | target   | feature           |   standardized_beta |
|:--------|:--------------|:---------|:------------------|--------------------:|
| global  | ALL           | EER      | theta_power_drift |            0.00262  |
| global  | ALL           | EER      | alpha_power_drift |           -0.001513 |
| global  | ALL           | EER      | beta_power_drift  |           -0.000666 |
| global  | ALL           | EER      | gamma_power_drift |            0.000227 |
| global  | ALL           | AUC      | theta_power_drift |           -0.002815 |
| global  | ALL           | AUC      | alpha_power_drift |            0.001634 |
| global  | ALL           | AUC      | beta_power_drift  |            0.000399 |
| global  | ALL           | AUC      | gamma_power_drift |           -0.000253 |
| region  | Frontal       | EER      | theta_power_drift |           -7.7e-05  |
| region  | Frontal       | EER      | alpha_power_drift |           -0.000299 |
| region  | Frontal       | EER      | beta_power_drift  |            1.4e-05  |
| region  | Frontal       | EER      | gamma_power_drift |            0.000117 |
| region  | Frontal       | AUC      | theta_power_drift |            8e-06    |
| region  | Frontal       | AUC      | alpha_power_drift |            7.9e-05  |
| region  | Frontal       | AUC      | beta_power_drift  |           -4e-06    |
| region  | Frontal       | AUC      | gamma_power_drift |           -1e-05    |
| region  | Central/Motor | EER      | theta_power_drift |            0.001051 |
| region  | Central/Motor | EER      | alpha_power_drift |           -0.000337 |
| region  | Central/Motor | EER      | beta_power_drift  |            0.000171 |
| region  | Central/Motor | EER      | gamma_power_drift |            0.000541 |
| region  | Central/Motor | AUC      | theta_power_drift |           -0.001243 |
| region  | Central/Motor | AUC      | alpha_power_drift |            0.000556 |
| region  | Central/Motor | AUC      | beta_power_drift  |           -0.000363 |
| region  | Central/Motor | AUC      | gamma_power_drift |           -0.000761 |
| region  | Temporal      | EER      | theta_power_drift |            0.004013 |
| region  | Temporal      | EER      | alpha_power_drift |           -0.001129 |
| region  | Temporal      | EER      | beta_power_drift  |            0.000875 |
| region  | Temporal      | EER      | gamma_power_drift |           -0.002692 |
| region  | Temporal      | AUC      | theta_power_drift |           -0.00492  |
| region  | Temporal      | AUC      | alpha_power_drift |            0.001595 |
| region  | Temporal      | AUC      | beta_power_drift  |           -0.002114 |
| region  | Temporal      | AUC      | gamma_power_drift |            0.004263 |
| region  | Parietal      | EER      | theta_power_drift |            0.000829 |
| region  | Parietal      | EER      | alpha_power_drift |           -0.000701 |
| region  | Parietal      | EER      | beta_power_drift  |           -0.000395 |
| region  | Parietal      | EER      | gamma_power_drift |            0.000108 |
| region  | Parietal      | AUC      | theta_power_drift |           -0.000951 |
| region  | Parietal      | AUC      | alpha_power_drift |            0.000698 |
| region  | Parietal      | AUC      | beta_power_drift  |            0.00037  |
| region  | Parietal      | AUC      | gamma_power_drift |           -0.000304 |
| region  | Occipital     | EER      | theta_power_drift |            0.001736 |
| region  | Occipital     | EER      | alpha_power_drift |           -0.001157 |
| region  | Occipital     | EER      | beta_power_drift  |           -0.001114 |
| region  | Occipital     | EER      | gamma_power_drift |            0.000711 |
| region  | Occipital     | AUC      | theta_power_drift |           -0.000392 |
| region  | Occipital     | AUC      | alpha_power_drift |            0.000109 |
| region  | Occipital     | AUC      | beta_power_drift  |            0.000159 |
| region  | Occipital     | AUC      | gamma_power_drift |           -0.000113 |

## Paper-ready novelty statement

Existing EEG biometric studies primarily report aggregate verification performance. In contrast, this work explicitly models identity as a dynamic distribution and validates cognitive-state-induced biometric drift using embedding-level, decision-level, spectral-band, and region-level evidence.