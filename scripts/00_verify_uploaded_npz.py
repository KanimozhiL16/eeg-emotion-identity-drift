from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC")
DATA_DIR = BASE / "data" / "raw_npz"
OUT_DIR = BASE / "data" / "verified"
OUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(DATA_DIR.glob("SEEDV_Q1_SAFE_SESSION*_16sub.npz"))

print("Files found:", len(files))
assert len(files) == 3, "Expected exactly 3 session NPZ files"

total_windows = 0
rows = []

for f in files:
    print("\nChecking:", f.name)
    d = np.load(f, allow_pickle=True)

    X = d["X"]
    y_subject = d["y_subject"]
    y_session = d["y_session"]
    y_trial = d["y_trial"]
    y_emotion = d["y_emotion"]

    print("X:", X.shape, X.dtype)
    print("Subjects:", sorted(np.unique(y_subject).tolist()))
    print("Sessions:", sorted(np.unique(y_session).tolist()))
    print("Trials:", sorted(np.unique(y_trial).tolist()))
    print("Emotions:", sorted(np.unique(y_emotion).tolist()))
    print("Emotion counts:", np.bincount(y_emotion, minlength=5))

    assert X.ndim == 3
    assert X.shape[1:] == (62, 400)
    assert set(np.unique(y_subject)) == set(range(1, 17))
    assert set(np.unique(y_trial)) == set(range(1, 16))
    assert set(np.unique(y_emotion)) == set(range(5))
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()

    total_windows += X.shape[0]

    rows.append({
        "file": f.name,
        "shape": str(X.shape),
        "windows": int(X.shape[0]),
        "session": int(np.unique(y_session)[0]),
        "subjects": int(len(np.unique(y_subject))),
        "emotion_counts": np.bincount(y_emotion, minlength=5).tolist()
    })

print("\nTotal windows:", total_windows)
assert total_windows == 114144, "Total window count mismatch"

out_csv = OUT_DIR / "brev_uploaded_npz_verification.csv"
pd.DataFrame(rows).to_csv(out_csv, index=False)

print("\n✅ BREV UPLOADED NPZ FILES VERIFIED SUCCESSFULLY")
print("Saved:", out_csv)
