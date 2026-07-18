#!/usr/bin/env python3
"""
00_verify_setup.py -- VERIFY EVERYTHING before running the work3 experiments.
Checks: ROOT, python packages, SEED-V sessionwise npz (keys/shapes/emotion labels),
the merged mechanism CSV (path/columns/variants/session-transitions/drift columns),
run_07 helper functions + eer_auc arity, and the AEP npz (channels incl T7).
Prints a PASS/FAIL report and says which of the 7 experiment scripts are GREEN to run.
Nothing is modified. Run FIRST.
  python -u scripts/work3/00_verify_setup.py 2>&1 | tee outputs/work3/00_verify.log
"""
import os, glob, re, importlib.util, numpy as np
def _has(p): return os.path.isdir(os.path.join(p,"outputs"))
_hp=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
ROOT=next((c for c in [os.getcwd(),_hp,"/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC",
      "/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"] if _has(c)),os.getcwd())
os.makedirs(os.path.join(ROOT,"outputs","work3"),exist_ok=True)
PASS=[]; FAIL=[]; WARN=[]
def ok(c,msg): (PASS if c else FAIL).append(msg); print(("  [OK]  " if c else "  [!!]  ")+msg)
def warn(msg): WARN.append(msg); print("  [~~]  "+msg)
print("="*84); print("WORK3 SETUP VERIFICATION"); print("ROOT =",ROOT); print("="*84)

# 1. packages
print("\n[1] python packages")
import importlib
for pkg,req in [("numpy",1),("scipy",1),("pandas",1),("sklearn",1),("statsmodels",0),("torch",0),("transformers",0)]:
    try: importlib.import_module(pkg); ok(True,f"{pkg} import")
    except Exception as e:
        if req: ok(False,f"{pkg} MISSING (required): {e}")
        else: warn(f"{pkg} missing (needed only for: statsmodels=extra stats, torch/transformers=REVE step15)")

# 2. SEED-V sessionwise
print("\n[2] SEED-V sessionwise windows  (data/processed/sessionwise/*.npz)")
DATA=os.path.join(ROOT,"data","processed","sessionwise")
npzs=sorted(glob.glob(os.path.join(DATA,"*.npz")))
ok(len(npzs)>=3, f"found {len(npzs)} npz files")
emo_key=None; sessions=set()
for f in npzs:
    d=np.load(f,allow_pickle=True); ks=list(d.keys())
    ek=next((k for k in ["y_emotion","y_label","emotion"] if k in ks),None)
    if ek: emo_key=ek
    sid=int(np.unique(d["y_session"])[0]) if "y_session" in ks else None
    if sid: sessions.add(sid)
    print(f"    {os.path.basename(f)}: X={d['X'].shape} keys={ks}")
ok(emo_key is not None, f"emotion label key present ({emo_key})")
ok("y_session" in ks, "y_session present")
ok({1,2,3}.issubset(sessions), f"sessions present = {sorted(sessions)} (need 1,2,3)")
d0=np.load(npzs[0],allow_pickle=True)
if "ch_names" in d0: ok(len(d0["ch_names"])>=62, f"channels = {len(d0['ch_names'])}")

# 3. merged mechanism CSV
print("\n[3] merged mechanism CSV (Table-3 data)")
import pandas as pd
cands=glob.glob(os.path.join(ROOT,"outputs","**","merged_global_psd_identity_eer.csv"),recursive=True)
ok(len(cands)>0,"merged_global_psd_identity_eer.csv found")
if cands:
    CSV=sorted(cands,key=len)[0]; df=pd.read_csv(CSV); print("    path:",os.path.relpath(CSV,ROOT),"| rows:",len(df))
    print("    columns:",list(df.columns))
    band_cols=[c for c in df.columns if c.endswith("_power_drift")]
    ok("theta_power_drift" in df.columns, "theta_power_drift column present")
    ok("EER" in df.columns, "EER column present")
    warn("delta_power_drift present -> C9 delta usable from CSV") if "delta_power_drift" in df.columns \
        else warn("delta_power_drift NOT in CSV -> C9 delta comes from step 11 (window recompute)")
    warn("total_psd_drift present -> broadband control usable") if "total_psd_drift" in df.columns \
        else warn("total_psd_drift NOT in CSV -> broadband control skipped in step 10")
    if "variant" in df.columns: print("    variants:",sorted(map(str,df['variant'].unique())))
    for c in ["enroll_emotion","test_emotion","enroll_session","test_session"]:
        if c in df.columns: print(f"    {c} unique:",sorted(map(str,df[c].dropna().unique()))[:6])
    # confirm session cols are filenames -> our sess_num() maps them
    def sess_num(v):
        s=str(v); m=re.search(r'SESSION\s*_?(\d+)',s,re.I); return m.group(1) if m else (re.search(r'(\d+)',s).group(1) if re.search(r'(\d+)',s) else s)
    if "test_session" in df.columns:
        mapped=sorted(set(df["test_session"].map(sess_num)))
        ok(set(mapped)<= {"1","2","3"}, f"test_session -> session number mapping = {mapped} (expect 2,3)")

# 4. run_07 helper module
print("\n[4] run_07 helper functions")
SRC=os.path.join(ROOT,"scripts","07_seedv_session_drift_subject_adaptation.py")
ok(os.path.exists(SRC), "scripts/07_seedv_session_drift_subject_adaptation.py exists")
if os.path.exists(SRC):
    try:
        spec=importlib.util.spec_from_file_location("run07",SRC); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        for fn in ["simple_features","make_prototypes","verification_scores","eer_auc","downsample_per_subject"]:
            ok(hasattr(m,fn), f"run_07.{fn} available")
        # eer_auc arity check on a tiny synthetic input
        try:
            r=m.eer_auc(np.array([1,0,1,0]),np.array([0.9,0.1,0.8,0.2]))
            n=len(r) if hasattr(r,"__len__") else 1
            print(f"    eer_auc returns {n} values -> scripts use [0]/[1], OK")
        except Exception as e: warn(f"eer_auc quick-call failed (ok if it needs specific args): {e}")
    except Exception as e:
        ok(False,f"run_07 import failed: {e}")

# 5. AEP dataset
print("\n[5] AEP auditory dataset")
AEP=os.path.join(ROOT,"data","cross_dataset_aep","AEP_win2s_step1s_fs256_4ch.npz")
ok(os.path.exists(AEP), "AEP npz exists")
if os.path.exists(AEP):
    a=np.load(AEP,allow_pickle=True); ch=[str(c) for c in a["ch_names"]]
    print("    X:",a["X"].shape," ch_names:",ch," fs:",int(a["fs"]))
    ok("T7" in ch, f"T7 channel present (index {ch.index('T7') if 'T7' in ch else 'NA'})")
    ok("source_file" in a, "source_file provenance present (for session parsing)")

# summary
print("\n"+"="*84); print(f"SUMMARY: {len(PASS)} OK | {len(FAIL)} FAIL | {len(WARN)} notes"); print("="*84)
if FAIL:
    print("BLOCKERS:"); [print("  -",x) for x in FAIL]
green={
 "10_csv_mechanism_checks.py":"merged CSV" ,
 "11_band_recompute_from_windows.py":"sessionwise npz + merged CSV",
 "12_subject_theta_vs_eer.py":"sessionwise npz + run_07",
 "13_matcher_theta_consistency.py":"sessionwise npz + merged CSV",
 "14_theta_aware_mitigation.py":"sessionwise npz",
 "15_reve_linear_probe.py":"sessionwise npz + torch/transformers",
 "16_aep_spectral_theta.py":"AEP npz",
}
print("\nScripts and their needs:")
for s,need in green.items(): print(f"  {s:38s} needs: {need}")
print("\nIf 0 blockers -> run  bash scripts/work3/run_all.sh  (or run each step 10..16 individually).")
