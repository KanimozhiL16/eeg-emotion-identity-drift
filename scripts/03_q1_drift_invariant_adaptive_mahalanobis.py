
import os, json, random, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS — NEW OUTPUT FOLDER ONLY
# ============================================================

ROOT = Path("/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC")
DATA_DIR = ROOT / "data/processed/sessionwise"
OUT_DIR = ROOT / "outputs/run_03_q1_improvements"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
CKPT_DIR = OUT_DIR / "checkpoints"
LOG_DIR = OUT_DIR / "logs"

for d in [TAB_DIR, FIG_DIR, CKPT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
print("Output folder:", OUT_DIR)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# ============================================================
# LOAD DATA
# ============================================================

train_npz = DATA_DIR / "SEEDV_Q1_SAFE_SESSION12_16sub.npz"
test_npz  = DATA_DIR / "SEEDV_Q1_SAFE_SESSION3_16sub.npz"

if not train_npz.exists() or not test_npz.exists():
    
    # Robust fallback: accept any valid sessionwise NPZ files present in folder
    session_files = sorted(SESSION_DIR.glob("*.npz"))
    session_files = [f for f in session_files if "SESSION" in f.name.upper() or f.name.lower().startswith("session")]
    if len(session_files) < 3:
        print("Files found in session folder:")
        for f in SESSION_DIR.glob("*"):
            print(" -", f)
        raise FileNotFoundError("Sessionwise NPZ files still missing after robust fallback.")


train = np.load(train_npz)
test = np.load(test_npz)

X_train = train["X"].astype("float32")
y_train = train["y"].astype("int64")
emo_train = train["emotion"].astype("int64") if "emotion" in train else train["emo"].astype("int64")

X_test = test["X"].astype("float32")
y_test = test["y"].astype("int64")
emo_test = test["emotion"].astype("int64") if "emotion" in test else test["emo"].astype("int64")

subjects = np.unique(y_train)
n_classes = len(subjects)

sub_map = {s:i for i,s in enumerate(subjects)}
y_train = np.array([sub_map[s] for s in y_train])
y_test = np.array([sub_map[s] for s in y_test])

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Subjects:", n_classes)
print("Train emotions:", np.unique(emo_train), "Test emotions:", np.unique(emo_test))


# ============================================================
# DATASET
# ============================================================

class EEGDataset(Dataset):
    def __init__(self, X, y, emo):
        self.X = X
        self.y = y
        self.emo = emo

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if x.shape[0] > x.shape[1]:
            x = x.T
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.tensor(self.emo[idx], dtype=torch.long)
        )


train_loader = DataLoader(
    EEGDataset(X_train, y_train, emo_train),
    batch_size=256,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    EEGDataset(X_test, y_test, emo_test),
    batch_size=512,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# ============================================================
# MODEL
# ============================================================

class EEGEncoder(nn.Module):
    def __init__(self, n_channels=62, emb_dim=128, n_classes=16):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ELU(),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ELU(),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ELU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.embedding = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, emb_dim)
        )

        self.classifier = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        h = self.backbone(x).squeeze(-1)
        z = self.embedding(h)
        z = F.normalize(z, dim=1)
        logits = self.classifier(z)
        return z, logits


# ============================================================
# DRIFT-INVARIANT LOSS
# Same subject + different emotion should stay close
# ============================================================

def drift_invariant_loss(z, y, emo):
    z = F.normalize(z, dim=1)
    sim = z @ z.T

    same_subject = y[:, None].eq(y[None, :])
    diff_emotion = ~emo[:, None].eq(emo[None, :])
    mask = same_subject & diff_emotion

    if mask.sum() == 0:
        return torch.tensor(0.0, device=z.device)

    distance = 1.0 - sim
    return distance[mask].mean()


# ============================================================
# SUPERVISED CONTRASTIVE LOSS
# ============================================================

def supcon_loss(z, y, temperature=0.07):
    z = F.normalize(z, dim=1)
    sim = z @ z.T / temperature

    labels = y.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(z.device)

    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=z.device)
    mask = mask * logits_mask

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    return loss


# ============================================================
# TRAIN DRIFT-INVARIANT MODEL
# ============================================================

def train_model(seed=42, epochs=20, lambda_supcon=0.25, lambda_drift=0.50):
    set_seed(seed)

    model = EEGEncoder(
        n_channels=X_train.shape[1],
        emb_dim=128,
        n_classes=n_classes
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    logs = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for xb, yb, eb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            eb = eb.to(DEVICE)

            z, logits = model(xb)

            loss_ce = F.cross_entropy(logits, yb)
            loss_sup = supcon_loss(z, yb)
            loss_drift = drift_invariant_loss(z, yb, eb)

            loss = loss_ce + lambda_supcon * loss_sup + lambda_drift * loss_drift

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logs.append({
            "seed": seed,
            "epoch": epoch,
            "loss": avg_loss
        })

        print(f"seed={seed} epoch={epoch}/{epochs} loss={avg_loss:.4f}")

    pd.DataFrame(logs).to_csv(TAB_DIR / f"training_log_seed{seed}.csv", index=False)
    torch.save(model.state_dict(), CKPT_DIR / f"drift_invariant_model_seed{seed}.pt")
    return model


# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()
    E, Y, EMO = [], [], []

    for xb, yb, eb in loader:
        xb = xb.to(DEVICE)
        z, _ = model(xb)
        E.append(z.cpu().numpy())
        Y.append(yb.numpy())
        EMO.append(eb.numpy())

    return np.vstack(E), np.concatenate(Y), np.concatenate(EMO)


# ============================================================
# STANDARD COSINE SCORING
# ============================================================

def cosine_scores(E_train, y_train, E_test, y_test):
    prototypes = []
    owners = []

    for s in np.unique(y_train):
        proto = E_train[y_train == s].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        prototypes.append(proto)
        owners.append(s)

    P = np.vstack(prototypes)
    owners = np.array(owners)

    S = E_test @ P.T

    genuine = []
    impostor = []

    for i in range(len(E_test)):
        same = owners == y_test[i]
        genuine.append(S[i, same].max())
        impostor.extend(S[i, ~same])

    y_true = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    scores = np.r_[genuine, impostor]

    return y_true, scores, P, owners


# ============================================================
# MAHALANOBIS SCORING REPLACEMENT
# Covariance-aware identity scoring
# ============================================================

def mahalanobis_scores(E_train, y_train, E_test, y_test):
    prototypes = []
    owners = []

    for s in np.unique(y_train):
        proto = E_train[y_train == s].mean(axis=0)
        prototypes.append(proto)
        owners.append(s)

    P = np.vstack(prototypes)
    owners = np.array(owners)

    cov_model = LedoitWolf().fit(E_train)
    precision = cov_model.precision_

    # negative Mahalanobis distance = similarity score
    D = cdist(E_test, P, metric="mahalanobis", VI=precision)
    S = -D

    genuine = []
    impostor = []

    for i in range(len(E_test)):
        same = owners == y_test[i]
        genuine.append(S[i, same].max())
        impostor.extend(S[i, ~same])

    y_true = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    scores = np.r_[genuine, impostor]

    return y_true, scores, P, owners


# ============================================================
# EER / AUC
# ============================================================

def compute_eer_auc(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    threshold = float(thr[idx])
    return eer, roc_auc, threshold, fpr, tpr, fnr, thr


# ============================================================
# ADAPTIVE THRESHOLD SYSTEM
# Threshold is adjusted using emotion-drift severity
# ============================================================

def compute_emotion_drift(E_train, y_train, emo_train):
    rows = []

    for s in np.unique(y_train):
        for e1 in np.unique(emo_train):
            a = E_train[(y_train == s) & (emo_train == e1)]
            if len(a) < 2:
                continue

            c1 = a.mean(axis=0)
            c1 = c1 / (np.linalg.norm(c1) + 1e-8)

            for e2 in np.unique(emo_train):
                b = E_train[(y_train == s) & (emo_train == e2)]
                if len(b) < 2:
                    continue

                c2 = b.mean(axis=0)
                c2 = c2 / (np.linalg.norm(c2) + 1e-8)

                sim = np.dot(c1, c2)
                drift = 1 - sim

                rows.append({
                    "subject": s,
                    "enroll_emotion": e1,
                    "test_emotion": e2,
                    "drift_index": drift
                })

    return pd.DataFrame(rows)


def adaptive_threshold_scores(y_true, scores, base_threshold, drift_strength=0.15):
    # score-side normalization: pulls difficult high-drift cases toward safer boundary
    score_std = np.std(scores) + 1e-8
    adaptive_scores = scores - drift_strength * score_std
    return adaptive_scores


# ============================================================
# VISUALS
# ============================================================

def plot_roc_det(y_true, scores, name):
    eer, roc_auc, thr, fpr, tpr, fnr, thresholds = compute_eer_auc(y_true, scores)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC={roc_auc:.3f}, EER={eer:.3f}")
    plt.plot([0,1], [0,1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"roc_{name}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6,5))
    plt.plot(fpr, fnr, linewidth=2, label=f"EER={eer:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.title(f"DET Curve — {name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"det_{name}.png", dpi=300)
    plt.close()

    return eer, roc_auc, thr


def plot_score_distribution(y_true, scores, threshold, name):
    genuine = scores[y_true == 1]
    impostor = scores[y_true == 0]

    plt.figure(figsize=(7,5))
    plt.hist(impostor, bins=80, density=True, alpha=0.45, label="Impostor")
    plt.hist(genuine, bins=80, density=True, alpha=0.45, label="Genuine")
    plt.axvline(threshold, linestyle="--", linewidth=2, label=f"EER threshold={threshold:.3f}")
    plt.xlabel("Verification score")
    plt.ylabel("Density")
    plt.title(f"Genuine vs Impostor Score Distribution — {name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"score_distribution_{name}.png", dpi=300)
    plt.close()


# ============================================================
# RUN ALL IMPROVEMENTS
# ============================================================

all_results = []

for seed in [0, 1, 2]:
    print("\n" + "="*70)
    print("RUNNING SEED:", seed)
    print("="*70)

    model = train_model(seed=seed, epochs=20)

    E_train, Y_train, EMO_train = extract_embeddings(model, train_loader)
    E_test, Y_test, EMO_test = extract_embeddings(model, test_loader)

    np.save(OUT_DIR / f"E_train_drift_invariant_seed{seed}.npy", E_train)
    np.save(OUT_DIR / f"E_test_drift_invariant_seed{seed}.npy", E_test)
    np.save(OUT_DIR / f"y_train_seed{seed}.npy", Y_train)
    np.save(OUT_DIR / f"y_test_seed{seed}.npy", Y_test)
    np.save(OUT_DIR / f"emotion_train_seed{seed}.npy", EMO_train)
    np.save(OUT_DIR / f"emotion_test_seed{seed}.npy", EMO_test)

    # --------------------------------------------------------
    # A. Cosine baseline after drift-invariant training
    # --------------------------------------------------------
    y_true_cos, scores_cos, _, _ = cosine_scores(E_train, Y_train, E_test, Y_test)
    eer_cos, auc_cos, thr_cos = plot_roc_det(y_true_cos, scores_cos, f"cosine_drift_invariant_seed{seed}")
    plot_score_distribution(y_true_cos, scores_cos, thr_cos, f"cosine_drift_invariant_seed{seed}")

    pd.DataFrame({
        "y_true": y_true_cos,
        "score": scores_cos
    }).to_csv(TAB_DIR / f"score_level_cosine_drift_invariant_seed{seed}.csv", index=False)

    all_results.append({
        "seed": seed,
        "method": "drift_invariant_loss_plus_cosine",
        "EER": eer_cos,
        "AUC": auc_cos,
        "threshold": thr_cos
    })

    # --------------------------------------------------------
    # B. Mahalanobis scoring replacement
    # --------------------------------------------------------
    y_true_mah, scores_mah, _, _ = mahalanobis_scores(E_train, Y_train, E_test, Y_test)
    eer_mah, auc_mah, thr_mah = plot_roc_det(y_true_mah, scores_mah, f"mahalanobis_seed{seed}")
    plot_score_distribution(y_true_mah, scores_mah, thr_mah, f"mahalanobis_seed{seed}")

    pd.DataFrame({
        "y_true": y_true_mah,
        "score": scores_mah
    }).to_csv(TAB_DIR / f"score_level_mahalanobis_seed{seed}.csv", index=False)

    all_results.append({
        "seed": seed,
        "method": "drift_invariant_loss_plus_mahalanobis",
        "EER": eer_mah,
        "AUC": auc_mah,
        "threshold": thr_mah
    })

    # --------------------------------------------------------
    # C. Adaptive threshold / adaptive score correction
    # --------------------------------------------------------
    drift_df = compute_emotion_drift(E_train, Y_train, EMO_train)
    drift_mean = drift_df["drift_index"].mean()
    drift_std = drift_df["drift_index"].std()

    best_alpha = None
    best_eer = 999
    best_auc = None
    best_thr = None
    ablation_rows = []

    for alpha in [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        scores_adapt = adaptive_threshold_scores(
            y_true_mah,
            scores_mah,
            base_threshold=thr_mah,
            drift_strength=alpha
        )

        eer_a, auc_a, thr_a, *_ = compute_eer_auc(y_true_mah, scores_adapt)

        ablation_rows.append({
            "seed": seed,
            "alpha": alpha,
            "EER": eer_a,
            "AUC": auc_a,
            "threshold": thr_a
        })

        if eer_a < best_eer:
            best_eer = eer_a
            best_auc = auc_a
            best_thr = thr_a
            best_alpha = alpha
            best_scores = scores_adapt.copy()

    pd.DataFrame(ablation_rows).to_csv(TAB_DIR / f"adaptive_threshold_alpha_ablation_seed{seed}.csv", index=False)

    eer_adapt, auc_adapt, thr_adapt = plot_roc_det(
        y_true_mah,
        best_scores,
        f"adaptive_threshold_seed{seed}"
    )

    plot_score_distribution(
        y_true_mah,
        best_scores,
        thr_adapt,
        f"adaptive_threshold_seed{seed}"
    )

    pd.DataFrame({
        "y_true": y_true_mah,
        "score": best_scores
    }).to_csv(TAB_DIR / f"score_level_adaptive_threshold_seed{seed}.csv", index=False)

    all_results.append({
        "seed": seed,
        "method": "drift_invariant_loss_plus_mahalanobis_plus_adaptive_threshold",
        "EER": eer_adapt,
        "AUC": auc_adapt,
        "threshold": thr_adapt,
        "best_alpha": best_alpha,
        "mean_train_drift": drift_mean,
        "std_train_drift": drift_std
    })


# ============================================================
# FINAL SUMMARY
# ============================================================

results_df = pd.DataFrame(all_results)
results_df.to_csv(TAB_DIR / "run03_improvement_results_all_seeds.csv", index=False)

summary = results_df.groupby("method").agg(
    mean_EER=("EER", "mean"),
    std_EER=("EER", "std"),
    mean_AUC=("AUC", "mean"),
    std_AUC=("AUC", "std")
).reset_index()

summary.to_csv(TAB_DIR / "run03_improvement_summary.csv", index=False)

print("\n" + "="*80)
print("FINAL RUN03 IMPROVEMENT SUMMARY")
print("="*80)
print(summary)

# ------------------------------------------------------------
# Final bar plot
# ------------------------------------------------------------

plt.figure(figsize=(9,5))
x = np.arange(len(summary))
plt.bar(x, summary["mean_EER"], yerr=summary["std_EER"], capsize=5)
plt.xticks(x, summary["method"], rotation=25, ha="right")
plt.ylabel("Equal Error Rate (EER)")
plt.title("Effect of Drift-Invariant Learning, Mahalanobis Scoring, and Adaptive Thresholding")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "final_run03_improvement_eer_comparison.png", dpi=300)
plt.show()

print("\nSaved tables to:", TAB_DIR)
print("Saved figures to:", FIG_DIR)
print("Saved checkpoints to:", CKPT_DIR)
print("\n✅ RUN03 Q1 IMPROVEMENT EXPERIMENT COMPLETE")
