#!/usr/bin/env python3
"""
make_manuscript_assets.py
1) Generate the 2 missing figures (baselines, capacity) from their CSVs.
2) Collect all CANONICAL valid figures + tables into manuscript_assets/{figures,tables}
   with manuscript-friendly names (Fig01..., T1...).
3) Tar everything -> manuscript_assets.tar.gz (download from Jupyter).
USAGE:  python scripts/make_manuscript_assets.py
"""
import os, shutil, tarfile, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT="/lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC"
if not os.path.isdir(ROOT): ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
O=lambda *p: os.path.join(ROOT,"outputs",*p)
ASSET=os.path.join(ROOT,"manuscript_assets"); FIGD=os.path.join(ASSET,"figures"); TABD=os.path.join(ASSET,"tables")
for d in (FIGD,TABD): os.makedirs(d,exist_ok=True)
print("="*70); print("BUILD manuscript_assets"); print("="*70)

# ---- 1) generate missing figures ----
b=O("run_12_baselines","baseline_comparison.csv")
if os.path.exists(b):
    df=pd.read_csv(b);
    plt.figure(figsize=(7,4))
    methods=df["method"].unique(); x=range(len(df))
    plt.bar([f"{m}\n{s}" for m,s in zip(df["method"],df["split"])], df["EER"], color="#4C72B0")
    plt.ylabel("EER"); plt.title("Baseline comparison (EER)"); plt.xticks(rotation=45,ha="right",fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(FIGD,"Fig10_baseline_comparison.png"),dpi=300); plt.close(); print("  generated Fig10_baseline_comparison")
c=O("run_14_capacity","capacity_ablation.csv")
if os.path.exists(c):
    df=pd.read_csv(c)
    plt.figure(figsize=(6.5,4))
    for col,lab in [("EER_within","within"),("EER_S2","S1→S2"),("EER_S3","S1→S3"),("within_to_cross_gap","gap")]:
        if col in df: plt.plot(df["params"],df[col],"o-",label=lab)
    plt.xscale("log"); plt.xlabel("model parameters (log)"); plt.ylabel("EER")
    plt.title("Capacity ablation: drift ≠ capacity"); plt.legend(fontsize=8); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIGD,"Fig11_capacity_ablation.png"),dpi=300); plt.close(); print("  generated Fig11_capacity_ablation")

# ---- 2) collect canonical figures ----
FIGS=[
 ("Fig01_session_drift_degradation.png","run_07_seedv_session_drift_subject_adaptation/figures/fig01_sessionwise_eer_degradation.png"),
 ("Fig02_subject_drift_boxplot.png","run_07_seedv_session_drift_subject_adaptation/figures/fig02_subject_session_drift_boxplot.png"),
 ("Fig03a_drift_vs_eer.png","run_07_seedv_session_drift_subject_adaptation/figures/fig03_session_drift_vs_eer.png"),
 ("Fig03b_change_point.png","run_04_q1_complete_post_screenshot/figures/fig05_phase_transition_piecewise_model.png"),
 ("Fig04a_adaptation_threshold.png","run_07_seedv_session_drift_subject_adaptation/figures/fig04_adaptive_threshold_comparison.png"),
 ("Fig04b_adaptation_curve.png","run_07_seedv_session_drift_subject_adaptation/figures/fig05_subject_adaptation_curve.png"),
 ("Fig05a_roc_det.png","run_07_seedv_session_drift_subject_adaptation/figures/fig06_score_level_session_roc.png"),
 ("Fig05b_score_distribution.png","run_07_seedv_session_drift_subject_adaptation/figures/fig07_session_score_distribution.png"),
 ("Fig06a_theta_vs_eer.png","run_13_biological/figures/fig1_theta_vs_eer.png"),
 ("Fig06b_band_betas.png","run_13_biological/figures/fig2_band_betas.png"),
 ("Fig06c_region_ranking.png","run_13_biological/figures/fig3_region_ranking.png"),
 ("Fig07a_theta_drift_heatmap.png","run_06_q1_levelup_biological_embedding_analysis/figures/fig02_global_theta_drift_heatmap.png"),
 ("Fig07b_alpha_drift_heatmap.png","run_06_q1_levelup_biological_embedding_analysis/figures/fig02_global_alpha_drift_heatmap.png"),
 ("Fig07c_beta_drift_heatmap.png","run_06_q1_levelup_biological_embedding_analysis/figures/fig02_global_beta_drift_heatmap.png"),
 ("Fig07d_gamma_drift_heatmap.png","run_06_q1_levelup_biological_embedding_analysis/figures/fig02_global_gamma_drift_heatmap.png"),
 ("Fig08_embedding_trajectory.png","run_13_biological/figures/fig4_embedding_trajectory.png"),
 ("Fig09a_aep_roc.png","run_10_aep_cross_dataset_validation/figures/fig01_aep_roc.png"),
 ("Fig09b_aep_score_distribution.png","run_10_aep_cross_dataset_validation/figures/fig02_score_distribution.png"),
]
TABLES=[
 ("T1a_session_verification.csv","run_07_seedv_session_drift_subject_adaptation/tables/session_verification_results.csv"),
 ("T1b_session_drift_summary.csv","run_07_seedv_session_drift_subject_adaptation/tables/session_drift_performance_summary.csv"),
 ("T2_statistics_H1_H5.csv","run_11_statistics/statistics_summary.csv"),
 ("T3_baselines.csv","run_12_baselines/baseline_comparison.csv"),
 ("T4_capacity_ablation.csv","run_14_capacity/capacity_ablation.csv"),
 ("T5a_regression_band_contributions.csv","run_06_q1_levelup_biological_embedding_analysis/tables/regression_standardized_band_contributions.csv"),
 ("T5b_biological_stats.csv","run_13_biological/biological_stats.csv"),
 ("T6a_adaptation_significance.csv","run_16_adaptation_significance/adaptation_significance.csv"),
 ("T6b_adaptation_persubject.csv","run_16_adaptation_significance/persubject.csv"),
 ("T7a_aep_summary.csv","run_10_aep_cross_dataset_validation/tables/summary.csv"),
 ("T7b_aep_scores.csv","run_10_aep_cross_dataset_validation/tables/verification_scores.csv"),
]
miss=[]
for dest,src in FIGS:
    s=O(*src.split("/"))
    if os.path.exists(s): shutil.copy(s,os.path.join(FIGD,dest))
    else: miss.append(("FIG",dest,src))
for dest,src in TABLES:
    s=O(*src.split("/"))
    if os.path.exists(s): shutil.copy(s,os.path.join(TABD,dest))
    else: miss.append(("TAB",dest,src))

# ---- 3) tar ----
tar=os.path.join(ROOT,"manuscript_assets.tar.gz")
with tarfile.open(tar,"w:gz") as t: t.add(ASSET,arcname="manuscript_assets")
nf=len(os.listdir(FIGD)); nt=len(os.listdir(TABD))
print(f"  figures collected: {nf} | tables collected: {nt}")
for k,d,s in miss: print(f"  MISSING {k}: {d}  (src {s})")
print(f"  TAR -> {tar}  ({os.path.getsize(tar)/1e6:.1f} MB)")
print("="*70); print("DONE. Download manuscript_assets.tar.gz from the Jupyter file browser."); print("="*70)
