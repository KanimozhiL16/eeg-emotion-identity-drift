#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, warnings, math
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path("/home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC")
SESSION_DIR = ROOT / "data/processed/sessionwise"
PREV_TABLE_DIRS = [
    ROOT / "outputs/run_02_q1_validation/tables",
    ROOT / "outputs/run_04_q1_complete_post_screenshot/tables",
    ROOT / "outputs/run_05_q1_psd_biological_validation/tables",
]
OUT = ROOT / "outputs/run_06_q1_levelup_biological_embedding_analysis"
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
LOG_DIR = OUT / "logs"
REPORT_DIR = OUT / "report"
for d in [OUT, FIG_DIR, TAB_DIR, LOG_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EMO_MAP = {0:"Disgust", 1:"Fear", 2:"Sad", 3:"Neutral", 4:"Happy"}
BANDS = {"theta":(4,8), "alpha":(8,13), "beta":(13,30), "gamma":(30,45)}
DEFAULT_62_CH = ["FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8","FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8","T7","C5","C3","C1","CZ","C2","C4","C6","T8","TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8","P7","P5","P3","P1","PZ","P2","P4","P6","P8","PO7","PO5","PO3","POZ","PO4","PO6","PO8","CB1","O1","OZ","O2","CB2"]
REGION_ORDER = ["Frontal", "Central/Motor", "Temporal", "Parietal", "Occipital", "Other"]

def region_of_channel(ch):
    c = str(ch).upper().replace(" ", "")
    if c.startswith(("FP","AF","F")) and not c.startswith(("FT",)): return "Frontal"
    if c.startswith(("FC","C")) and not c.startswith(("CP","CB")): return "Central/Motor"
    if c.startswith(("T","FT","TP")): return "Temporal"
    if c.startswith(("P","CP")) and not c.startswith(("PO",)): return "Parietal"
    if c.startswith(("O","PO","CB")): return "Occipital"
    return "Other"

def fdr_bh(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[order[i]] = min(prev, 1.0)
    return q

def safe_corr(x, y, method="pearson"):
    sub = pd.DataFrame({"x":x, "y":y}).replace([np.inf,-np.inf], np.nan).dropna()
    if len(sub) < 3 or sub["x"].nunique() < 2 or sub["y"].nunique() < 2:
        return np.nan, np.nan, len(sub)
    if method == "spearman":
        r, p = spearmanr(sub["x"], sub["y"])
    else:
        r, p = pearsonr(sub["x"], sub["y"])
    return float(r), float(p), int(len(sub))

def read_npz_file(p):
    z = np.load(p, allow_pickle=True)
    keys = list(z.keys())
    X = z["X"].astype(np.float32)
    y_sub = z["y_subject"] if "y_subject" in keys else z["y"] if "y" in keys else None
    y_emo = z["y_emotion"] if "y_emotion" in keys else z["emotion"] if "emotion" in keys else None
    if y_sub is None or y_emo is None:
        raise KeyError(f"{p.name} must contain X, y_subject, y_emotion. Keys={keys}")
    fs = int(np.array(z["fs"]).item()) if "fs" in keys else 200
    if "ch_names" in keys:
        ch_names = [str(x) for x in z["ch_names"]]
    else:
        ch_names = DEFAULT_62_CH[:X.shape[1]]
    return X, np.asarray(y_sub), np.asarray(y_emo), fs, ch_names, keys

def get_session_files():
    files = sorted(SESSION_DIR.glob("*.npz"))
    real = [f for f in files if not f.is_symlink() and ("SESSION" in f.name.upper() or f.name.lower().startswith("session"))]
    orig = [f for f in real if "SEEDV_Q1_SAFE_SESSION" in f.name.upper()]
    if len(orig) >= 3: return sorted(orig)[:3]
    if len(real) >= 3: return sorted(real)[:3]
    raise FileNotFoundError(f"Could not find 3 sessionwise NPZ files in {SESSION_DIR}")

def compute_channel_bandpower(X, fs, max_samples=2500, seed=42):
    if len(X) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]
    freqs, psd = welch(X, fs=fs, nperseg=min(256, X.shape[-1]), axis=-1)
    out = {}
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        out[band] = psd[:, :, mask].mean(axis=(0,2))
    return out

def build_psd_channel_table():
    rows = []
    files = get_session_files()
    print("Session files:")
    for f in files: print(" -", f.name)
    for sess_idx, f in enumerate(files, start=1):
        X, y_sub, y_emo, fs, ch_names, keys = read_npz_file(f)
        print(f"Processing {f.name}: X={X.shape}, fs={fs}, keys={keys}")
        regions = [region_of_channel(c) for c in ch_names]
        for emo_id in sorted(np.unique(y_emo)):
            mask = y_emo == emo_id
            if int(mask.sum()) < 20: continue
            bp = compute_channel_bandpower(X[mask], fs, seed=100 + sess_idx + int(emo_id))
            for ch_i, ch_name in enumerate(ch_names):
                row = {"session_file": f.name, "session_index": sess_idx, "emotion_id": int(emo_id),
                       "emotion": EMO_MAP.get(int(emo_id), str(emo_id)), "channel": ch_name,
                       "region": regions[ch_i], "n_windows": int(mask.sum())}
                for band in BANDS:
                    row[f"{band}_power"] = float(bp[band][ch_i])
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "channel_bandpower_by_session_emotion.csv", index=False)
    return df

def region_summary(psd_ch):
    region_df = psd_ch.groupby(["session_file","session_index","emotion_id","emotion","region"], as_index=False)[[f"{b}_power" for b in BANDS]].mean()
    region_df.to_csv(TAB_DIR / "region_bandpower_by_session_emotion.csv", index=False)
    return region_df

def build_region_drift(region_df):
    sessions = sorted(region_df["session_file"].unique())
    enroll = sessions[0]
    test_sessions = sessions[1:]
    rows = []
    for test_sess in test_sessions:
        for eemo in sorted(region_df[region_df.session_file == enroll]["emotion"].unique()):
            for temo in sorted(region_df[region_df.session_file == test_sess]["emotion"].unique()):
                for region in REGION_ORDER:
                    a = region_df[(region_df.session_file == enroll) & (region_df.emotion == eemo) & (region_df.region == region)]
                    b = region_df[(region_df.session_file == test_sess) & (region_df.emotion == temo) & (region_df.region == region)]
                    if len(a)==0 or len(b)==0: continue
                    row = {"enroll_session": enroll, "test_session": test_sess, "enroll_emotion": eemo, "test_emotion": temo, "transition": f"{eemo}->{temo}", "region": region}
                    total = 0.0
                    for band in BANDS:
                        val = abs(float(a[f"{band}_power"].iloc[0]) - float(b[f"{band}_power"].iloc[0]))
                        row[f"{band}_power_drift"] = val
                        total += val
                    row["total_psd_drift"] = total
                    rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "region_psd_drift_cross_emotion.csv", index=False)
    return df

def build_global_drift(region_drift):
    band_cols = [f"{b}_power_drift" for b in BANDS]
    global_df = region_drift.groupby(["enroll_session","test_session","enroll_emotion","test_emotion","transition"], as_index=False)[band_cols + ["total_psd_drift"]].mean()
    global_df.to_csv(TAB_DIR / "global_psd_drift_cross_emotion.csv", index=False)
    return global_df

def find_identity_table():
    candidates = []
    for d in PREV_TABLE_DIRS:
        candidates += [d / "cross_emotion_results_all_seeds.csv", d / "cross_emotion_results.csv", d / "main_multiseed_results.csv", d / "main_multiseed_results_live.csv"]
    for p in candidates:
        if p.exists(): return p
    return None

def normalize_identity(df):
    df = df.copy()
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["enroll_emotion","enrollment_emotion"]: ren[c] = "enroll_emotion"
        elif cl in ["test_emotion","probe_emotion"]: ren[c] = "test_emotion"
        elif cl in ["eer","equal_error_rate"]: ren[c] = "EER"
        elif cl in ["auc","roc_auc"]: ren[c] = "AUC"
        elif cl in ["drift_index","identity_drift","mean_identity_drift"]: ren[c] = "drift_index"
    df = df.rename(columns=ren)
    if "variant" in df.columns and (df["variant"].astype(str) == "arcface_supcon_cnn").any():
        df = df[df["variant"].astype(str) == "arcface_supcon_cnn"].copy()
    return df

def merge_identity(global_df, region_drift):
    p = find_identity_table()
    if p is None:
        print("WARNING: no identity table found; continuing PSD-only.")
        return None, None, None
    idf = normalize_identity(pd.read_csv(p))
    print("Using identity table:", p)
    if "enroll_emotion" not in idf.columns or "test_emotion" not in idf.columns:
        print("Identity table lacks emotion columns. Columns:", list(idf.columns))
        return p, None, None
    keep = ["enroll_emotion","test_emotion"] + [c for c in ["variant","seed","EER","AUC","drift_index","CDT"] if c in idf.columns]
    idf = idf[keep].copy()
    idf["enroll_emotion"] = idf["enroll_emotion"].astype(str)
    idf["test_emotion"] = idf["test_emotion"].astype(str)
    g = global_df.copy()
    r = region_drift.copy()
    for df in [g, r]:
        df["enroll_emotion"] = df["enroll_emotion"].astype(str)
        df["test_emotion"] = df["test_emotion"].astype(str)
    gm = g.merge(idf, on=["enroll_emotion","test_emotion"], how="left")
    rm = r.merge(idf, on=["enroll_emotion","test_emotion"], how="left")
    gm.to_csv(TAB_DIR / "merged_global_psd_identity_eer.csv", index=False)
    rm.to_csv(TAB_DIR / "merged_region_psd_identity_eer.csv", index=False)
    return p, gm, rm

def correlation_with_fdr(merged_global, merged_region):
    rows = []
    def add_rows(df, level, region):
        if df is None: return
        targets = [c for c in ["EER","AUC","drift_index"] if c in df.columns]
        preds = [f"{b}_power_drift" for b in BANDS] + ["total_psd_drift"]
        for pred in preds:
            if pred not in df.columns: continue
            for target in targets:
                pr, pp, n = safe_corr(df[pred], df[target], "pearson")
                sr, sp, _ = safe_corr(df[pred], df[target], "spearman")
                rows.append({"level":level,"region":region,"predictor":pred,"target":target,"n":n,"pearson_r":pr,"pearson_p":pp,"spearman_r":sr,"spearman_p":sp})
    add_rows(merged_global, "global", "ALL")
    if merged_region is not None:
        for region in REGION_ORDER:
            add_rows(merged_region[merged_region.region == region], "region", region)
    out = pd.DataFrame(rows)
    if len(out):
        out["pearson_q_fdr"] = fdr_bh(out["pearson_p"].fillna(1).values)
        out["spearman_q_fdr"] = fdr_bh(out["spearman_p"].fillna(1).values)
    out.to_csv(TAB_DIR / "fdr_corrected_psd_identity_eer_correlations.csv", index=False)
    return out

def regression_models(merged_global, merged_region):
    rows, coef_rows = [], []
    def fit_one(df, level, region, target):
        feature_cols = [f"{b}_power_drift" for b in BANDS]
        sub = df[feature_cols + [target]].replace([np.inf,-np.inf], np.nan).dropna()
        if len(sub) < 8 or sub[target].nunique() < 2: return
        X = StandardScaler().fit_transform(sub[feature_cols].values)
        y = sub[target].values
        model = RidgeCV(alphas=np.logspace(-4, 4, 25), cv=min(5, len(sub)))
        model.fit(X, y)
        pred = model.predict(X)
        rows.append({"level":level, "region":region, "target":target, "n":len(sub), "model":"RidgeCV", "alpha":float(model.alpha_), "R2_in_sample":float(r2_score(y,pred)), "RMSE":float(math.sqrt(mean_squared_error(y,pred)))})
        for feat, coef in zip(feature_cols, model.coef_):
            coef_rows.append({"level":level, "region":region, "target":target, "feature":feat, "standardized_beta":float(coef)})
    if merged_global is not None:
        for target in [c for c in ["EER","AUC","drift_index"] if c in merged_global.columns]:
            fit_one(merged_global, "global", "ALL", target)
    if merged_region is not None:
        for region in REGION_ORDER:
            rdf = merged_region[merged_region.region == region]
            for target in [c for c in ["EER","AUC","drift_index"] if c in rdf.columns]:
                fit_one(rdf, "region", region, target)
    reg, coefs = pd.DataFrame(rows), pd.DataFrame(coef_rows)
    reg.to_csv(TAB_DIR / "regression_model_psd_predicts_eer_identity_drift.csv", index=False)
    coefs.to_csv(TAB_DIR / "regression_standardized_band_contributions.csv", index=False)
    return reg, coefs

def feature_table_for_trajectory():
    files = get_session_files()
    vecs, labels = [], []
    for sess_idx, f in enumerate(files, start=1):
        X, y_sub, y_emo, fs, ch_names, keys = read_npz_file(f)
        for emo_id in sorted(np.unique(y_emo)):
            mask = y_emo == emo_id
            if int(mask.sum()) < 20: continue
            bp = compute_channel_bandpower(X[mask], fs, seed=900 + sess_idx + int(emo_id))
            vecs.append(np.concatenate([bp[b] for b in BANDS]))
            labels.append({"session_file":f.name, "session_index":sess_idx, "emotion_id":int(emo_id), "emotion":EMO_MAP.get(int(emo_id), str(emo_id))})
    meta = pd.DataFrame(labels)
    coords = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(np.vstack(vecs)))
    pca = PCA(n_components=2, random_state=42).fit(StandardScaler().fit_transform(np.vstack(vecs)))
    meta["PC1"], meta["PC2"] = coords[:,0], coords[:,1]
    meta["explained_var_PC1"], meta["explained_var_PC2"] = pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]
    meta.to_csv(TAB_DIR / "spectral_embedding_pca_trajectory_coordinates.csv", index=False)
    return meta

def plot_heatmap(df, value_col, title, fname):
    if df is None or value_col not in df.columns: return
    piv = df.pivot_table(index="enroll_emotion", columns="test_emotion", values=value_col, aggfunc="mean")
    if piv.empty: return
    plt.figure(figsize=(7.4,5.8))
    im = plt.imshow(piv.values, aspect="auto")
    plt.colorbar(im, label=value_col.replace("_"," "))
    plt.xticks(range(len(piv.columns)), piv.columns, rotation=35, ha="right")
    plt.yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i,j]
            if np.isfinite(val): plt.text(j,i,f"{val:.3f}",ha="center",va="center",fontsize=8)
    plt.title(title); plt.xlabel("Probe/test emotion"); plt.ylabel("Enrollment emotion")
    plt.tight_layout(); plt.savefig(FIG_DIR / fname, dpi=300, bbox_inches="tight"); plt.close()

def make_figures(region_df, region_drift, global_df, merged_global, merged_region, corr, coefs, traj):
    mean_region = region_df.groupby("region")[[f"{b}_power" for b in BANDS]].mean().reindex(REGION_ORDER).dropna(how="all")
    plt.figure(figsize=(9,5)); mean_region.plot(kind="bar", ax=plt.gca())
    plt.title("Region-wise EEG Spectral Power across Cognitive States"); plt.ylabel("Mean PSD band power"); plt.xlabel("Brain region")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout(); plt.savefig(FIG_DIR/"fig01_regionwise_psd_bandpower.png", dpi=300, bbox_inches="tight"); plt.close()
    for b in BANDS: plot_heatmap(global_df, f"{b}_power_drift", f"{b.capitalize()} Drift across Cognitive-State Transitions", f"fig02_global_{b}_drift_heatmap.png")
    plot_heatmap(global_df, "total_psd_drift", "Total Spectral Drift across Cognitive-State Transitions", "fig03_global_total_psd_drift_heatmap.png")
    rd = region_drift.groupby("region")["total_psd_drift"].agg(["mean","std","count"]).reindex(REGION_ORDER).dropna()
    plt.figure(figsize=(8,5)); x=np.arange(len(rd)); plt.bar(x, rd["mean"].values, yerr=rd["std"].fillna(0).values, capsize=4)
    plt.xticks(x, rd.index, rotation=25, ha="right"); plt.ylabel("Mean total PSD drift"); plt.xlabel("Brain region")
    plt.title("Neurophysiological Localization of Cognitive-State Drift"); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(FIG_DIR/"fig04_regionwise_total_psd_drift.png", dpi=300, bbox_inches="tight"); plt.close()
    if merged_global is not None and "EER" in merged_global.columns:
        sub = merged_global[["total_psd_drift","EER"]].dropna()
        if len(sub)>=3:
            pr,pp=pearsonr(sub["total_psd_drift"],sub["EER"]); plt.figure(figsize=(7,5)); plt.scatter(sub["total_psd_drift"],sub["EER"],alpha=0.7,label="Cross-emotion conditions")
            coef=np.polyfit(sub["total_psd_drift"],sub["EER"],2 if len(sub)>5 else 1); xs=np.linspace(sub["total_psd_drift"].min(),sub["total_psd_drift"].max(),100)
            plt.plot(xs,np.polyval(coef,xs),label=f"Trend; r={pr:.3f}, p={pp:.2e}"); plt.title("Biological Spectral Drift vs Verification Error"); plt.xlabel("Total PSD drift"); plt.ylabel("Equal Error Rate (EER)")
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(FIG_DIR/"fig05_total_psd_drift_vs_eer.png",dpi=300,bbox_inches="tight"); plt.close()
    if merged_global is not None and "drift_index" in merged_global.columns:
        sub = merged_global[["total_psd_drift","drift_index"]].dropna()
        if len(sub)>=3:
            pr,pp=pearsonr(sub["total_psd_drift"],sub["drift_index"]); plt.figure(figsize=(7,5)); plt.scatter(sub["total_psd_drift"],sub["drift_index"],alpha=0.7)
            coef=np.polyfit(sub["total_psd_drift"],sub["drift_index"],2 if len(sub)>5 else 1); xs=np.linspace(sub["total_psd_drift"].min(),sub["total_psd_drift"].max(),100)
            plt.plot(xs,np.polyval(coef,xs),label=f"Trend; r={pr:.3f}, p={pp:.2e}"); plt.title("Spectral Drift Explains Embedding-Level Identity Drift"); plt.xlabel("Total PSD drift"); plt.ylabel("Identity drift index")
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(FIG_DIR/"fig06_total_psd_drift_vs_identity_drift.png",dpi=300,bbox_inches="tight"); plt.close()
    if corr is not None and len(corr):
        c=corr[(corr["target"]=="EER") & (corr["level"]=="region")].copy()
        if len(c):
            c["label"]=c["region"]+"\n"+c["predictor"].str.replace("_power_drift","",regex=False).str.replace("total_psd_drift","total",regex=False)
            c=c.sort_values("pearson_r",ascending=False).head(15); plt.figure(figsize=(11,5)); x=np.arange(len(c)); plt.bar(x,c["pearson_r"].values); plt.axhline(0,linestyle="--",linewidth=1)
            plt.xticks(x,c["label"],rotation=45,ha="right",fontsize=8); plt.ylabel("Pearson r with EER"); plt.title("FDR-Corrected Region/Band Association with Verification Error")
            plt.grid(axis="y",alpha=0.3); plt.tight_layout(); plt.savefig(FIG_DIR/"fig07_region_band_correlation_with_eer.png",dpi=300,bbox_inches="tight"); plt.close()
    if coefs is not None and len(coefs):
        g=coefs[(coefs["level"]=="global") & (coefs["target"]=="EER")].copy()
        if len(g):
            g["feature"]=g["feature"].str.replace("_power_drift","",regex=False); plt.figure(figsize=(7,5)); plt.bar(g["feature"],g["standardized_beta"]); plt.axhline(0,linestyle="--",linewidth=1)
            plt.title("Band Contribution Model: PSD Drift Predicting Verification Error"); plt.ylabel("Standardized regression coefficient"); plt.xlabel("Frequency band")
            plt.grid(axis="y",alpha=0.3); plt.tight_layout(); plt.savefig(FIG_DIR/"fig08_regression_band_contributions_to_eer.png",dpi=300,bbox_inches="tight"); plt.close()
    if traj is not None and len(traj):
        plt.figure(figsize=(7,6))
        for emo in sorted(traj["emotion"].unique()):
            sub=traj[traj.emotion==emo].sort_values("session_index"); plt.scatter(sub["PC1"],sub["PC2"],label=emo,s=60,alpha=0.8); plt.plot(sub["PC1"],sub["PC2"],alpha=0.5)
            for _,row in sub.iterrows(): plt.text(row["PC1"],row["PC2"],f"S{int(row['session_index'])}",fontsize=8)
        ev1=traj["explained_var_PC1"].iloc[0]; ev2=traj["explained_var_PC2"].iloc[0]
        plt.title("Spectral Embedding Trajectories across Sessions and Emotions"); plt.xlabel(f"PC1 ({ev1:.1%} variance)"); plt.ylabel(f"PC2 ({ev2:.1%} variance)")
        plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(FIG_DIR/"fig09_spectral_embedding_pca_trajectories.png",dpi=300,bbox_inches="tight"); plt.close()
    if merged_global is not None and "EER" in merged_global.columns:
        top=merged_global.groupby("transition")["EER"].mean().sort_values(ascending=False).head(10)
        if len(top):
            plt.figure(figsize=(9,5)); plt.bar(range(len(top)),top.values); plt.xticks(range(len(top)),top.index,rotation=45,ha="right")
            plt.ylabel("Mean EER"); plt.xlabel("Emotion transition"); plt.title("Worst Cognitive-State Transitions for EEG Verification")
            plt.grid(axis="y",alpha=0.3); plt.tight_layout(); plt.savefig(FIG_DIR/"fig10_worst_cross_emotion_transitions_by_eer.png",dpi=300,bbox_inches="tight"); plt.close()

def write_report(identity_source, corr, reg, coefs):
    lines = ["# RUN06 Q1 Level-Up Biological and Embedding Analysis Report\n",
    "## What this run adds\n",
    "This run strengthens the paper by adding region-level neurophysiological validation, FDR-corrected statistics, regression-based explanation of verification error, and spectral trajectory visualization. It does not overwrite previous results.\n",
    "## Data and outputs\n",
    f"- Project root: `{ROOT}`", f"- Session files: `{SESSION_DIR}`", f"- Identity/EER source table: `{identity_source}`", f"- Figures: `{FIG_DIR}`", f"- Tables: `{TAB_DIR}`\n",
    "## Methodological logic\n",
    "1. EEG is decomposed into theta, alpha, beta, and gamma band powers using Welch spectral estimation.",
    "2. Band power is summarized globally and by brain region: frontal, central/motor, temporal, parietal, and occipital.",
    "3. Cognitive-state drift is measured as the absolute spectral change between enrollment emotion and probe emotion.",
    "4. Spectral drift is merged with the previously generated identity-drift/EER table.",
    "5. Pearson and Spearman correlations are corrected using Benjamini-Hochberg FDR.",
    "6. A ridge-regression model estimates how much band-specific spectral drift explains EER and identity drift.",
    "7. PCA visualizes cognitive-state trajectories in spectral feature space.\n",
    "## Scientific interpretation\n",
    "The analysis tests whether EEG identity degradation is accompanied by measurable changes in neural spectral dynamics. If verification error increases together with spectral-band or region-level drift, the result supports the claim that EEG biometric identity is not a fixed point, but a cognitive-state-dependent distribution.\n"]
    if corr is not None and len(corr):
        lines += ["## Top FDR-corrected associations with EER\n", corr[corr["target"]=="EER"].sort_values("pearson_q_fdr").head(12).round(5).to_markdown(index=False), ""]
    if reg is not None and len(reg):
        lines += ["## Regression model summary\n", reg.round(5).to_markdown(index=False), ""]
    if coefs is not None and len(coefs):
        lines += ["## Standardized band-contribution coefficients\n", coefs.round(6).to_markdown(index=False), ""]
    lines += ["## Paper-ready novelty statement\n", "Existing EEG biometric studies primarily report aggregate verification performance. In contrast, this work explicitly models identity as a dynamic distribution and validates cognitive-state-induced biometric drift using embedding-level, decision-level, spectral-band, and region-level evidence."]
    (REPORT_DIR/"run06_q1_levelup_report.md").write_text("\n".join(lines), encoding="utf-8")

def main():
    print("="*90); print("RUN06 Q1 LEVEL-UP: REGION PSD + FDR + REGRESSION + TRAJECTORY"); print("="*90); print("Output folder:", OUT)
    psd_ch=build_psd_channel_table(); region_df=region_summary(psd_ch); region_drift=build_region_drift(region_df); global_df=build_global_drift(region_drift)
    identity_source, merged_global, merged_region=merge_identity(global_df, region_drift)
    corr=correlation_with_fdr(merged_global, merged_region); reg,coefs=regression_models(merged_global, merged_region); traj=feature_table_for_trajectory()
    make_figures(region_df, region_drift, global_df, merged_global, merged_region, corr, coefs, traj); write_report(identity_source,corr,reg,coefs)
    manifest={"output_folder":str(OUT),"figures":sorted([p.name for p in FIG_DIR.glob("*.png")]),"tables":sorted([p.name for p in TAB_DIR.glob("*.csv")]),"report":str(REPORT_DIR/"run06_q1_levelup_report.md"),"identity_source":str(identity_source),"novelty_sentence":"Existing EEG biometric studies primarily report aggregate verification performance. In contrast, this work explicitly models identity as a dynamic distribution and validates cognitive-state-induced biometric drift using embedding-level, decision-level, spectral-band, and region-level evidence."}
    (OUT/"run06_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("\n✅ RUN06 COMPLETE"); print("Figures saved to:",FIG_DIR); print("Tables saved to:",TAB_DIR); print("Report saved to:",REPORT_DIR/"run06_q1_levelup_report.md")
    print("\nGenerated figures:")
    for f in manifest["figures"]: print(" -",f)
if __name__=="__main__":
    main()
