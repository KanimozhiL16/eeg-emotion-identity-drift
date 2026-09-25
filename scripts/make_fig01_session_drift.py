#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate Fig01_session_drift_degradation.png (Fig. 2a) from the VERIFIED
per-participant EER values so the bar labels match the manuscript text,
the figure caption, and _HANDOVER/project_state.json.

VERIFIED per-participant EER (SEED-V, leakage-controlled enrol-then-verify):
    within-session   0.131
    S1 -> S2         0.169
    S1 -> S3         0.246
(Friedman chi2(2)=11.38, p=0.003, Kendall W=0.355)

This only REDRAWS a bar chart from already-verified numbers; it does not
re-run any experiment. Style matches the previous figure (muted blue bars,
black edges, value labels, horizontal grid).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = ["Within-session", "1→2", "1→3"]
eer    = [0.131, 0.169, 0.246]           # VERIFIED per-participant EER

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
fig, ax = plt.subplots(figsize=(6.2, 4.7))
x = range(len(labels))
bars = ax.bar(x, eer, width=0.62, color="#4C72B0",
              edgecolor="black", linewidth=1.1, zorder=3)
for xi, v in zip(x, eer):
    ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=15)

ax.set_ylabel("Equal error rate (EER)")
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylim(0, 0.285)
ax.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
ax.yaxis.grid(True, color="#d9d9d9", linewidth=0.9, zorder=0)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

plt.tight_layout()
fig.savefig("Fig01_session_drift_degradation.png", dpi=600, bbox_inches="tight")
print("saved Fig01_session_drift_degradation.png with verified EER", eer)
