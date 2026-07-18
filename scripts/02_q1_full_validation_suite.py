import os, json, time, random, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import ttest_ind
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path("/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC")
DATA_DIR = ROOT / "data/processed/sessionwise"

OUT_ROOT = ROOT / "outputs/run_02_q1_validation"
FIG_DIR = OUT_ROOT / "figures"
TAB_DIR = OUT_ROOT / "tables"
MET_DIR = OUT_ROOT / "metrics"
LOG_DIR = OUT_ROOT / "logs"
CKPT_DIR = ROOT / "checkpoints/run_02_q1_validation"

for p in [FIG_DIR, TAB_DIR, MET_DIR, LOG_DIR, CKPT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
EPOCHS = int(os.environ.get("P4_EPOCHS", 20))
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTIONS = ["Disgust", "Fear", "Sad", "Neutral", "Happy"]


# ============================================================
# UTILITIES
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def eer_auc(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)

    valid = np.isfinite(scores) & np.isfinite(y_true)
    y_true = y_true[valid]
    scores = scores[valid]

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float(fpr[idx]), float(auc(fpr, tpr))

def l2norm(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# LOAD DATA
# ============================================================

def load_session_npz(sess):
    path = DATA_DIR / f"SEEDV_Q1_SAFE_SESSION{sess}_16sub.npz"
    d = np.load(path)
    return {
        "X": d["X"].astype(np.float32),
        "subject": d["y_subject"].astype(np.int64),
        "emotion": d["y_emotion"].astype(np.int64),
        "session": d["y_session"].astype(np.int64),
    }

print("Loading sessionwise NPZ files...")
s1, s2, s3 = load_session_npz(1), load_session_npz(2), load_session_npz(3)

X_train = np.concatenate([s1["X"], s2["X"]], axis=0)
y_train_sub = np.concatenate([s1["subject"], s2["subject"]], axis=0)
y_train_emo = np.concatenate([s1["emotion"], s2["emotion"]], axis=0)

X_test = s3["X"]
y_test_sub = s3["subject"]
y_test_emo = s3["emotion"]

# subject label 1–16 → 0–15
y_train = y_train_sub - 1
y_test = y_test_sub - 1

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Device:", DEVICE)

file_hashes = {
    f.name: sha256_file(f) for f in sorted(DATA_DIR.glob("*.npz"))
}
with open(MET_DIR / "input_file_hashes.json", "w") as f:
    json.dump(file_hashes, f, indent=2)


# ============================================================
# DATASET
# ============================================================

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


# ============================================================
# MODEL
# ============================================================

class EncoderCNN(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(62, 128, 7, padding=3),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.proj(z)
        return F.normalize(z, dim=1)


class ArcFaceHead(nn.Module):
    def __init__(self, emb_dim=128, n_classes=16, s=16.0, m=0.20):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, emb_dim))
        nn.init.xavier_uniform_(self.W)
        self.s = s
        self.m = m

    def forward(self, emb, labels=None):
        W = F.normalize(self.W, dim=1)
        cosine_logits = F.linear(emb, W)

        if labels is None:
            return cosine_logits * self.s

        theta = torch.acos(torch.clamp(cosine_logits, -1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta + self.m)

        one_hot = F.one_hot(labels, num_classes=W.shape[0]).float()
        logits = cosine_logits * (1 - one_hot) + target_logits * one_hot
        return logits * self.s


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        sim = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return -mean_log_prob_pos.mean()


# ============================================================
# TRAINING
# ============================================================

def train_model(seed, variant):
    set_seed(seed)

    run_dir = CKPT_DIR / f"{variant}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"

    encoder = EncoderCNN().to(DEVICE)
    head = ArcFaceHead().to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=3e-4,
        weight_decay=1e-4
    )

    supcon = SupConLoss()

    start_epoch = 1
    best_acc = -1

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        encoder.load_state_dict(ckpt["encoder"])
        head.load_state_dict(ckpt["head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_acc"]
        print(f"Resuming {variant} seed {seed} from epoch {start_epoch}")

    train_loader = DataLoader(
        EEGDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        EEGDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    log_rows = []

    for epoch in range(start_epoch, EPOCHS + 1):
        encoder.train()
        head.train()

        total_loss, correct, total = 0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            emb = encoder(xb)

            if variant == "softmax_cnn":
                logits = head(emb, None)
                loss = F.cross_entropy(logits, yb)

            elif variant == "arcface_cnn":
                logits = head(emb, yb)
                loss = F.cross_entropy(logits, yb)

            elif variant == "arcface_supcon_cnn":
                logits = head(emb, yb)
                loss = F.cross_entropy(logits, yb) + 0.5 * supcon(emb, yb)

            else:
                raise ValueError("Unknown variant")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += len(xb)

        train_acc = correct / total

        encoder.eval()
        head.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                emb = encoder(xb)
                logits = head(emb, None)
                pred = logits.argmax(1)
                correct += (pred == yb).sum().item()
                total += len(xb)

        test_acc = correct / total

        row = {
            "seed": seed,
            "variant": variant,
            "epoch": epoch,
            "train_loss": total_loss / total,
            "train_acc": train_acc,
            "test_acc": test_acc
        }
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(LOG_DIR / f"trainlog_{variant}_seed{seed}.csv", index=False)

        torch.save({
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc
        }, ckpt_path)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "head": head.state_dict(),
                "best_acc": best_acc
            }, best_path)

        print(f"{variant} seed={seed} epoch={epoch}/{EPOCHS} test_acc={test_acc:.4f}")

    ckpt = torch.load(best_path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    return encoder.eval()


def extract_embeddings(encoder):
    def extract(X, y):
        loader = DataLoader(
            EEGDataset(X, y),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )
        embs, labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                emb = encoder(xb).cpu().numpy()
                embs.append(emb)
                labels.append(yb.numpy())
        return np.concatenate(embs), np.concatenate(labels)

    return extract(X_train, y_train), extract(X_test, y_test)


# ============================================================
# BIOMETRIC EVALUATION
# ============================================================

def biometric_scores(E_train, y_train, E_test, y_test):
    subjects = np.unique(y_train)
    templates = {
        s: l2norm(E_train[y_train == s]).mean(axis=0)
        for s in subjects
    }
    templates = {s: t / (np.linalg.norm(t) + 1e-8) for s, t in templates.items()}

    y_true, scores = [], []

    for i in range(len(E_test)):
        probe = E_test[i]
        true = y_test[i]
        for s, temp in templates.items():
            y_true.append(1 if s == true else 0)
            scores.append(cosine(probe, temp))

    return eer_auc(np.array(y_true), np.array(scores))


def cross_emotion_analysis(E_train, y_train, emo_train, E_test, y_test, emo_test, seed, variant):
    rows = []
    drift_rows = []

    subjects = sorted(np.unique(y_train))

    for e_enroll in range(5):
        for e_probe in range(5):

            templates = {}
            train_centroids = {}
            test_centroids = {}

            for s in subjects:
                idx_tr = np.where((y_train == s) & (emo_train == e_enroll))[0]
                idx_te = np.where((y_test == s) & (emo_test == e_probe))[0]

                if len(idx_tr) == 0 or len(idx_te) == 0:
                    continue

                tr_cent = l2norm(E_train[idx_tr]).mean(axis=0)
                te_cent = l2norm(E_test[idx_te]).mean(axis=0)

                tr_cent = tr_cent / (np.linalg.norm(tr_cent) + 1e-8)
                te_cent = te_cent / (np.linalg.norm(te_cent) + 1e-8)

                templates[s] = tr_cent
                train_centroids[s] = tr_cent
                test_centroids[s] = te_cent

            y_true, scores = [], []

            idx_probe = np.where(emo_test == e_probe)[0]

            for i in idx_probe:
                probe = E_test[i]
                true = y_test[i]

                for s, temp in templates.items():
                    y_true.append(1 if s == true else 0)
                    scores.append(cosine(probe, temp))

            eer, roc_auc = eer_auc(np.array(y_true), np.array(scores))

            subject_drifts = []
            for s in templates:
                cross_sim = cosine(train_centroids[s], test_centroids[s])
                drift = 1.0 - cross_sim
                subject_drifts.append(drift)

                drift_rows.append({
                    "seed": seed,
                    "variant": variant,
                    "subject": int(s),
                    "enroll_emotion": EMOTIONS[e_enroll],
                    "test_emotion": EMOTIONS[e_probe],
                    "cross_similarity": cross_sim,
                    "drift_index": drift
                })

            rows.append({
                "seed": seed,
                "variant": variant,
                "enroll_emotion": EMOTIONS[e_enroll],
                "test_emotion": EMOTIONS[e_probe],
                "EER": eer,
                "AUC": roc_auc,
                "mean_drift": float(np.mean(subject_drifts))
            })

    return pd.DataFrame(rows), pd.DataFrame(drift_rows)


# ============================================================
# PSD BASELINE
# ============================================================

def bandpower_features(X, fs=200):
    bands = [(0.5,4), (4,8), (8,13), (13,30), (30,45)]
    freqs = np.fft.rfftfreq(X.shape[-1], d=1/fs)
    fft = np.abs(np.fft.rfft(X, axis=-1)) ** 2
    feats = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs <= hi)
        feats.append(fft[:, :, mask].mean(axis=-1))
    F = np.concatenate(feats, axis=1).astype(np.float32)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


def psd_baseline(seed):
    set_seed(seed)
    print("Computing PSD baseline features...")
    F_train = bandpower_features(X_train)
    F_test = bandpower_features(X_test)

    clf = make_pipeline(
        StandardScaler(),
        SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, random_state=seed)
    )
    clf.fit(F_train, y_train)

    prob_train = clf.predict_proba(F_train)
    prob_test = clf.predict_proba(F_test)

    E_train_psd = l2norm(prob_train)
    E_test_psd = l2norm(prob_test)

    eer, roc_auc = biometric_scores(E_train_psd, y_train, E_test_psd, y_test)

    return {
        "seed": seed,
        "variant": "psd_sgd_baseline",
        "EER": eer,
        "AUC": roc_auc
    }


# ============================================================
# CDT + CHANGE POINT
# ============================================================

def compute_cdt(df, factor=1.20):
    d = df.sort_values("mean_drift")["mean_drift"].values
    e = df.sort_values("mean_drift")["EER"].values

    if len(e) >= 5:
        e_s = savgol_filter(e, window_length=5, polyorder=2)
    else:
        e_s = e

    base = np.min(e_s)
    thr = base * factor

    for i in range(len(d)-1):
        if e_s[i] <= thr and e_s[i+1] > thr:
            x1, x2 = d[i], d[i+1]
            y1, y2 = e_s[i], e_s[i+1]
            return float(x1 + (thr-y1)*(x2-x1)/(y2-y1))

    return np.nan


def piecewise_change_test(df, B=1000):
    x = df.sort_values("mean_drift")["mean_drift"].values
    y = df.sort_values("mean_drift")["EER"].values

    def lin_mse(x,y):
        c = np.polyfit(x,y,1)
        return np.mean((y-np.polyval(c,x))**2)

    def pw_mse(x,y):
        best = (np.inf, None)
        for k in range(3, len(x)-3):
            c1 = np.polyfit(x[:k], y[:k], 1)
            c2 = np.polyfit(x[k:], y[k:], 1)
            mse = (
                np.sum((y[:k]-np.polyval(c1,x[:k]))**2) +
                np.sum((y[k:]-np.polyval(c2,x[k:]))**2)
            ) / len(y)
            if mse < best[0]:
                best = (mse, k)
        return best

    mse_l = lin_mse(x,y)
    mse_p, k = pw_mse(x,y)
    obs = mse_l - mse_p

    rng = np.random.default_rng(42)
    perms = []
    for _ in range(B):
        yp = rng.permutation(y)
        ml = lin_mse(x,yp)
        mp,_ = pw_mse(x,yp)
        perms.append(ml-mp)

    p = (np.sum(np.array(perms) >= obs)+1)/(B+1)

    return {
        "linear_mse": mse_l,
        "piecewise_mse": mse_p,
        "improvement_ratio": mse_l/mse_p,
        "split_index": int(k),
        "transition_drift": float(x[k]),
        "permutation_p": float(p)
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

all_main = []
all_cross = []
all_drift = []
all_psd = []

variants = ["softmax_cnn", "arcface_cnn", "arcface_supcon_cnn"]

for seed in SEEDS:
    all_psd.append(psd_baseline(seed))

    for variant in variants:
        encoder = train_model(seed, variant)

        (E_tr, y_tr_emb), (E_te, y_te_emb) = extract_embeddings(encoder)

        emb_dir = OUT_ROOT / "embeddings" / f"{variant}_seed{seed}"
        emb_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_dir / "E_train.npy", E_tr)
        np.save(emb_dir / "E_test.npy", E_te)
        np.save(emb_dir / "y_train.npy", y_tr_emb)
        np.save(emb_dir / "y_test.npy", y_te_emb)

        eer, roc_auc = biometric_scores(E_tr, y_tr_emb, E_te, y_te_emb)

        cross_df, drift_df = cross_emotion_analysis(
            E_tr, y_tr_emb, y_train_emo,
            E_te, y_te_emb, y_test_emo,
            seed, variant
        )

        cdt = compute_cdt(cross_df)
        cp = piecewise_change_test(cross_df)

        all_main.append({
            "seed": seed,
            "variant": variant,
            "EER": eer,
            "AUC": roc_auc,
            "CDT": cdt,
            **cp
        })

        all_cross.append(cross_df)
        all_drift.append(drift_df)

        pd.DataFrame(all_main).to_csv(TAB_DIR / "main_multiseed_results_live.csv", index=False)

# save all
main_df = pd.DataFrame(all_main)
psd_df = pd.DataFrame(all_psd)
cross_all = pd.concat(all_cross, ignore_index=True)
drift_all = pd.concat(all_drift, ignore_index=True)

main_df.to_csv(TAB_DIR / "main_multiseed_results.csv", index=False)
psd_df.to_csv(TAB_DIR / "psd_baseline_results.csv", index=False)
cross_all.to_csv(TAB_DIR / "cross_emotion_results_all_seeds.csv", index=False)
drift_all.to_csv(TAB_DIR / "identity_drift_all_seeds.csv", index=False)

summary = main_df.groupby("variant").agg(["mean", "std"])
summary.to_csv(TAB_DIR / "summary_mean_std_by_variant.csv")

print("\n================ FINAL MULTI-SEED SUMMARY ================")
print(summary)
print("\nPSD baseline:")
print(psd_df)
print("\nSaved to:", OUT_ROOT)
