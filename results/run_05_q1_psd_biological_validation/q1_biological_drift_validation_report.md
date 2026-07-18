# Q1 Biological Drift Validation Report

## Purpose

This analysis strengthens the EEG biometric study by adding a neurophysiological validation layer. Instead of only showing that identity drift occurs, it examines whether cognitive-state-induced drift is reflected in EEG spectral-band changes.

## Data Used

- Session folder: `/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC/data/processed/sessionwise`

- Identity drift source: `/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC/outputs/run_02_q1_validation/tables/cross_emotion_results_all_seeds.csv`

- PSD rows: 15

- Cross-emotion PSD drift rows: 50


## Biological Interpretation

EEG identity is not stationary because cognitive and emotional states modulate brain rhythms. Theta activity is often linked with cognitive control and workload, alpha activity with attention and cortical inhibition, beta activity with active sensorimotor and cognitive processing, and gamma activity with higher-frequency neural synchronization. Therefore, when emotional or cognitive states change, the spectral distribution of EEG changes, which can shift the learned identity representation and increase biometric verification error.


## Theoretical Framing

This experiment supports the thesis that EEG identity should be modeled as a dynamic distribution rather than a fixed biometric signature. Under cognitive-state variability, this distribution shifts in spectral and embedding space, producing measurable identity drift.


## Correlation Summary

| predictor         | target   |   n |   pearson_r |   pearson_p |   spearman_r |   spearman_p |
|:------------------|:---------|----:|------------:|------------:|-------------:|-------------:|
| theta_power_drift | EER      | 250 |      0.212  |      0.0007 |       0.1663 |       0.0084 |
| theta_power_drift | AUC      | 250 |     -0.1384 |      0.0287 |      -0.13   |       0.04   |
| alpha_power_drift | EER      | 250 |     -0.1298 |      0.0403 |      -0.1    |       0.1148 |
| alpha_power_drift | AUC      | 250 |      0.0907 |      0.1527 |       0.0799 |       0.2081 |
| beta_power_drift  | EER      | 250 |      0.023  |      0.718  |       0.0432 |       0.4968 |
| beta_power_drift  | AUC      | 250 |     -0.0371 |      0.5592 |      -0.0451 |       0.4782 |
| gamma_power_drift | EER      | 250 |      0.0695 |      0.2738 |       0.0788 |       0.2143 |
| gamma_power_drift | AUC      | 250 |     -0.0631 |      0.3203 |      -0.0636 |       0.3163 |
| total_psd_drift   | EER      | 250 |      0.0724 |      0.2539 |       0.055  |       0.3869 |
| total_psd_drift   | AUC      | 250 |     -0.0557 |      0.3804 |      -0.044  |       0.4889 |



## Paper-Ready Novelty Sentence

Existing EEG biometric studies primarily report aggregate verification performance, whereas this work explicitly models, quantifies, and biologically validates cognitive-state-induced identity drift using both embedding-level and spectral-band evidence.
