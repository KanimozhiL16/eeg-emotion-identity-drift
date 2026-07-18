
# RUN10 AEP Cross-Dataset EEG Biometric Validation

## Dataset

- Dataset: Auditory Evoked Potential EEG Biometric Dataset
- Subjects: 20
- Input shape: (22201, 4, 512)
- Channels: 4
- Sampling rate: 256 Hz

## Protocol

- Subject-disjoint biometric verification
- Enrollment: first 50%
- Verification: remaining 50%
- Scoring: cosine similarity to subject prototype

## Results

- AUC: 0.731410
- EER: 0.336805

## Scientific Interpretation

The experiment validates that EEG biometric identity remains measurable within auditory-evoked neural paradigms using lightweight statistical EEG representations and prototype-based verification.

## Generated Outputs

### Figures
- fig01_aep_roc.png
- fig02_score_distribution.png

### Tables
- verification_scores.csv
- summary.csv
