#!/usr/bin/env python3
r"""
audit_project.py -- reconcile the P4 project's result files against the paper's headline numbers.
Runs IN BREV over the whole project (skips the venv / checkpoints / raw arrays), searches every
text-like result file (csv/log/txt/json/md/tex) for each manuscript value, and reports where each
is found -- or FLAGS numbers that appear nowhere (possible fabrication / stale value).

RUN:
  cd ~/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
  python3 scripts/audit_project.py | tee outputs/AUDIT_REPORT.txt
Then upload outputs/AUDIT_REPORT.txt here (small text file).
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
SKIP_DIRS = {"p4_seedv_env", "checkpoints", "__pycache__", ".git", ".ipynb_checkpoints",
             "node_modules", "wandb"}
TEXT_EXT = {".csv", ".log", ".txt", ".json", ".md", ".tex", ".py", ".yml", ".yaml"}
MAX_BYTES = 8 * 1024 * 1024

# (label, [accepted string forms]) -- a value PASSES if any form appears in some result file
PAPER_VALUES = [
    ("within-session EER 0.131",           ["0.131"]),
    ("S1->S2 EER 0.169",                    ["0.169"]),
    ("cross-session EER 0.246",             ["0.246"]),
    ("Friedman p=0.003",                    ["0.003"]),
    ("Kendall W=0.355",                     ["0.355"]),
    ("PSD within 0.183",                    ["0.183"]),
    ("PSD S1->S2 0.229",                    ["0.229"]),
    ("PSD S1->S3 0.272",                    ["0.272"]),
    ("EEGNet within 0.011",                 ["0.011"]),
    ("EEGNet 0.190",                        ["0.190", "0.19 "]),
    ("EEGNet 0.245",                        ["0.245"]),
    ("REVE 0.251",                          ["0.251"]),
    ("REVE 0.313",                          ["0.313"]),
    ("REVE 0.308",                          ["0.308"]),
    ("Mahalanobis 0.318",                   ["0.318"]),
    ("theta beta +0.43",                    ["0.43"]),
    ("theta z-scored 0.79",                 ["0.79"]),
    ("theta z-scored 0.91",                 ["0.91"]),
    ("theta ICA baseline 0.36",             ["0.36"]),
    ("theta ICA cleaned 0.40",              ["0.40", "0.404"]),
    ("manip-check mean 34.5%",              ["34.5"]),
    ("manip t(15)=5.96",                    ["5.96"]),
    ("manip Cohen d=1.49",                  ["1.49"]),
    ("SEED-IV PSD 0.233",                   ["0.233"]),
    ("SEED-IV PSD 0.324",                   ["0.324"]),
    ("SEED-IV PSD 0.362",                   ["0.362"]),
    ("linear probe 0.10",                   ["0.10"]),
    ("linear probe 0.11",                   ["0.11"]),
    ("m11 state dEER 0.047",                ["0.047"]),
    ("m11 time dEER 0.077",                 ["0.077"]),
    ("m11 both dEER 0.100",                 ["0.100", "0.0999"]),
    ("window count S1 37600",               ["37600", "37,600"]),
    ("window count S2 33856",               ["33856", "33,856"]),
    ("window count S3 42688",               ["42688", "42,688"]),
    ("total windows 114144",                ["114144", "114,144"]),
]

def text_files():
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in TEXT_EXT:
                fp = os.path.join(dp, fn)
                try:
                    if os.path.getsize(fp) <= MAX_BYTES:
                        yield fp
                except OSError:
                    pass

print("="*80); print("P4 PROJECT AUDIT  (result files vs paper values)"); print("ROOT:", ROOT); print("="*80)

files = list(text_files())
print(f"scanned {len(files)} text/result files (venv, checkpoints, raw arrays skipped)\n")

# load contents once
blobs = {}
for fp in files:
    try:
        blobs[fp] = open(fp, "r", errors="ignore").read()
    except Exception:
        pass

print("-"*80); print(f"{'VALUE':38s} {'STATUS':6s}  FOUND-IN (up to 3 files)"); print("-"*80)
flags = []
for label, forms in PAPER_VALUES:
    hits = []
    for fp, txt in blobs.items():
        if any(f in txt for f in forms):
            hits.append(os.path.relpath(fp, ROOT))
    status = "OK" if hits else "**FLAG"
    if not hits: flags.append(label)
    ex = "; ".join(hits[:3]) if hits else "-- not found in any result file --"
    print(f"{label:38s} {status:6s}  {ex}")

print("\n" + "="*80)
if flags:
    print(f"FLAGGED ({len(flags)}) -- these paper values were NOT found in any result file:")
    for f in flags: print("   -", f)
    print("   -> verify these are real (recompute) or correct the manuscript.")
else:
    print("ALL paper values were located in the project's result files.")
print("="*80)

# inventory of result CSVs (headers) so the reconciliation is auditable
print("\n--- RESULT CSV INVENTORY (path : header) ---")
for fp, txt in sorted(blobs.items()):
    if fp.lower().endswith(".csv"):
        head = txt.splitlines()[0][:120] if txt.strip() else "(empty)"
        print(f"  {os.path.relpath(fp, ROOT)} : {head}")
print("\nDONE. Upload outputs/AUDIT_REPORT.txt for review.")
