# New TAC figures — build & insert

Three figures reverse-engineered from the CCIEP TAC paper's evidence style, adapted
to **our** verification/theta/identity claims. All visualise data you already have —
**no new results, no fabricated numbers.**

## 1. Protocol schematic — DONE ✅
`FigProtocol.png` is already generated and inserted into `TAC_SUBMISSION/main.tex`
(Materials & Methods → Experimental Design, `Fig.~\ref{fig:protocol}`). Nothing to run.

## 2 & 3. Data-driven figures — run on the Brev box
These need your session-wise data, so run them where run_06/run_07 live:

```bash
cd /home/nvidia/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC
# copy the two scripts here first (from this folder), then:
python -W ignore fig_theta_topomap_grid.py 2>&1 | tee outputs/figs/theta_topomap.log
python -W ignore fig_embedding_tsne.py      2>&1 | tee outputs/figs/embedding_tsne.log
```

Outputs land in `outputs/figs/`:
- `Fig_theta_topomap_grid.png`  — Subject × Session theta topomaps (identity persists across rows, drifts across sessions)
- `Fig_embedding_tsne.png`      — t-SNE coloured by subject (clusters) and by session (drift)

**Before running, sanity-check one thing:** the 62-channel name order in
`fig_theta_topomap_grid.py` (`SEED62`) must match your preprocessing order. If your
dataset config stores channel names, load those instead — the script has a comment
marking where.

## Insert the two new figures (after you have the PNGs)
Copy the PNGs into `TAC_SUBMISSION/figures/`, then paste these into `main.tex`:

Theta topomap grid → put in **Results → Spectral and Spatial Determinants of Drift**:
```latex
\begin{figure}[t!]
\centering
\includegraphics[width=\linewidth]{Fig_theta_topomap_grid.png}
\caption{Theta-band (4--8\,Hz) power as scalp topographies for representative
subjects (rows) across sessions (columns), z-scored per subject. Per-subject
patterns remain distinct (identity), while each subject's map shifts across
sessions (drift), with the temporal-cortex emphasis reported in the text.}
\label{fig:theta_topo}
\end{figure}
```

Embedding t-SNE (now 3 panels, wide) → put in **Results → Subject-Level Robustness**:
```latex
\begin{figure*}[t!]
\centering
\includegraphics[width=0.98\linewidth]{Fig_embedding_tsne.png}
\caption{t-SNE of the identity features. (a)~All subjects, coloured by subject:
samples form person-specific clusters (identity is separable). (b)~The most stable
subject, coloured by session: sessions overlap. (c)~The most drift-prone subject,
coloured by session: sessions separate. Session drift is thus a within-subject
effect that is uneven across individuals, consistent with the per-participant
coefficient of variation reported in the text.}
\label{fig:tsne}
\end{figure*}
```
_(The stable/drift-prone subjects are auto-selected by the script and printed in the log —
use those subject numbers in your text.)_

Then add a one-line text reference to each (e.g. "…as shown in Fig.~\ref{fig:theta_topo}")
and recompile on Overleaf.
```
