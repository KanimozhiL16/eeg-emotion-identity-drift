# WORK3 experiment RESULTS LOG

Paste the console output / CSV contents from each Brev run under its heading. This file is the
authoritative record of what was run and what it produced. Numbers only enter the manuscript from here.

**Env:** host `brev-x1v47hbeh` · ROOT `/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC` · venv `p4_seedv_env`
**Scripts placed in:** `scripts/work3/` · **outputs to:** `outputs/work3/`

---

## 00_verify_setup.py  (paste SUMMARY block)
```
(paste here)
```

## 10_csv_mechanism_checks.py — C9 / C10 / C18 / C19 / C8(if multi-variant)
```
(paste here)
```

## 11_band_recompute_from_windows.py — C12 (amplitude-preserving PSD) + delta/normalised
```
(paste here)
```

## 12_subject_theta_vs_eer.py — C20 (subject-level theta → EER)
```
(paste here)
```

## 13_matcher_theta_consistency.py — C8 (theta across matchers)
```
(paste here)
```

## 14_theta_aware_mitigation.py — C15 (theta-aware mitigation)
```
(paste here)
```

## 15_reve_linear_probe.py — C7 (REVE frozen vs linear probe)
```
(paste here)
```

## 16_aep_spectral_theta.py — C14 (auditory theta / T7)
```
(paste here)
```

---
## Integration decisions (filled after results)
- C12: keep "band-power drift" vs rename "spectral redistribution" → _pending run_
- C18: keep "predictable" vs "systematically associated" → _pending run_
- C8/C20/C15/C7/C14/C9/C10/C19 → _pending run_
