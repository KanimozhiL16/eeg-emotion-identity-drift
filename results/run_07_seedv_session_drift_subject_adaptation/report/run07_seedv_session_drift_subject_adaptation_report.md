# RUN07: SEED-V Session-Drift and Subject-Adaptation Analysis

## Purpose

This run addresses longitudinal realism inside SEED-V by evaluating Session-1 enrollment against later sessions and by testing subject-adaptive mitigation. It does not overwrite earlier SEED-V results.

## Protocol

- Enrollment: SEED-V Session 1.
- Verification: held-out Session 1, Session 2, and Session 3.
- Scoring: subject prototype cosine verification using reproducible EEG time/frequency features.
- Adaptation: a small fraction of probe-session data updates subject prototypes; adaptive thresholds are estimated per claimed subject.

## Session Verification Results

| comparison                      |   probe_session |      EER |      AUC |   threshold |   adaptive_balanced_error |   n_scores |
|:--------------------------------|----------------:|---------:|---------:|------------:|--------------------------:|-----------:|
| S1 split → S1 heldout           |               1 | 0.137977 | 0.932519 |    0.335189 |                  0.019076 |     115200 |
| S1 enrollment → S2 verification |               2 | 0.181822 | 0.898031 |    0.276326 |                  0.023472 |     230400 |
| S1 enrollment → S3 verification |               3 | 0.250755 | 0.819263 |    0.207743 |                  0.02597  |     230400 |


## Session Drift Summary

|   probe_session |     mean |      std |   median |   count |      EER |      AUC |   adaptive_balanced_error |
|----------------:|---------:|---------:|---------:|--------:|---------:|---------:|--------------------------:|
|               2 | 0.154909 | 0.093399 | 0.141556 |      16 | 0.181822 | 0.898031 |                  0.023472 |
|               3 | 0.304934 | 0.312134 | 0.176337 |      16 | 0.250755 | 0.819263 |                  0.02597  |


## Subject Adaptation Summary

|   probe_session |   baseline_EER_no_adaptation |   best_EER_with_adaptation |   best_adaptation_fraction |   absolute_EER_reduction |   relative_EER_reduction_percent |
|----------------:|-----------------------------:|---------------------------:|---------------------------:|-------------------------:|---------------------------------:|
|               2 |                     0.181822 |                   0.161024 |                        0.2 |                 0.020797 |                          11.4384 |
|               3 |                     0.250755 |                   0.196752 |                        0.1 |                 0.054003 |                          21.5362 |


## Paper-ready interpretation

The SEED-V analysis indicates that EEG identity should not be treated as a fixed template: enrollment-to-future-session verification introduces measurable session-level representation drift. Subject-adaptive thresholding and lightweight prototype adaptation provide an explicit mitigation mechanism for session-induced biometric degradation.

## Generated figures

- fig01_sessionwise_eer_degradation.png
- fig02_subject_session_drift_boxplot.png
- fig03_session_drift_vs_eer.png
- fig04_adaptive_threshold_comparison.png
- fig05_subject_adaptation_curve.png
- fig06_score_level_session_roc.png
- fig07_session_score_distribution.png