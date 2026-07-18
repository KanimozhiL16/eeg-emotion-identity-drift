#!/usr/bin/env bash
# Run the full work3 experiment pack on Brev, in order, teeing each log.
# Prereqs: cd P4 root ; source p4_seedv_env/bin/activate ; pip install -q statsmodels scikit-learn transformers --break-system-packages
set -u
ROOT="$(pwd)"
mkdir -p outputs/work3
SD="scripts/work3"   # adjust if you placed the scripts elsewhere
run () { echo "======== $1 ========"; python -u "$SD/$1" 2>&1 | tee "outputs/work3/${1%.py}.log"; echo; }

run 00_verify_setup.py               # VERIFY paths/keys/columns FIRST -- read the SUMMARY before continuing
run 10_csv_mechanism_checks.py       # C9 C10 C18 C19 C8(if multi-variant) -- zero risk, run first
run 11_band_recompute_from_windows.py# C12 (amplitude-preserving PSD) + delta/normalised
run 12_subject_theta_vs_eer.py       # C20 subject-level theta -> EER
run 13_matcher_theta_consistency.py  # C8 theta across matchers
run 14_theta_aware_mitigation.py     # C15 theta-aware mitigation
run 15_reve_linear_probe.py          # C7 REVE frozen vs linear probe (needs GPU + transformers)
run 16_aep_spectral_theta.py         # C14 auditory theta / T7
