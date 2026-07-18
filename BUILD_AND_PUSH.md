# How to assemble the full repo and push to GitHub

These repo files (README, LICENSE, CITATION.cff, .gitignore, docs/, data/README) are the
**professional wrapper**. To make the complete submission repository, add your **code** and
**result files** from the project, excluding data/venv/checkpoints, then push.

## A. Assemble on Brev (recommended — code + evidence live there)

```bash
# 1) start from a clean repo dir
mkdir -p ~/eeg-emotion-identity-drift && cd ~/eeg-emotion-identity-drift

# 2) copy the wrapper files you downloaded from this chat into this dir
#    (README.md, LICENSE, CITATION.cff, .gitignore, requirements.txt, environment.yml,
#     data/README.md, docs/REPRODUCE.md, docs/PAPER_TABLE_MAP.md)

# 3) copy CODE
mkdir -p scripts
cp -r ~/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC/scripts/* scripts/ 2>/dev/null
find scripts -name '__pycache__' -type d -prune -exec rm -rf {} +

# 4) copy RESULTS/EVIDENCE (tables, logs, metrics) but NOT raw data / weights
mkdir -p results
rsync -a --exclude='*.npz' --exclude='*.npy' --exclude='*.pt' --exclude='*.pth' \
         --exclude='*.h5' --exclude='__pycache__' \
         ~/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC/outputs/ results/
cp -r ~/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC/manuscript_assets/tables results/manuscript_tables

# 5) copy figures + supplementary + pinned env
mkdir -p figures supplementary
cp -r ~/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC/manuscript_assets/figures/* figures/ 2>/dev/null
cp ~/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC/requirements.txt .
cp ~/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC/environment.yml   . 2>/dev/null

# 6) sanity: total size should be tens of MB, not GB
du -sh .
```

## B. Create the GitHub repo and push

On github.com create a new **empty** repository named `eeg-emotion-identity-drift`
(no README/License, since we provide our own). Then:

```bash
cd ~/eeg-emotion-identity-drift
git init
git add .
git commit -m "Initial release: code, results, and evidence for the IEEE TAC submission"
git branch -M main
git remote add origin https://github.com/<your-username>/eeg-emotion-identity-drift.git
git push -u origin main
```

If prompted for a password, use a **GitHub Personal Access Token** (Settings → Developer
settings → Personal access tokens → Fine-grained token with `repo` scope), not your account
password.

## C. Make it citable for the reviewers (recommended)

- Add the repository URL to the manuscript's **Data/Code Availability** statement.
- For a permanent DOI, link the repo to **Zenodo** (Zenodo → GitHub → flip the repo on →
  cut a GitHub *Release*; Zenodo mints a DOI). Put that DOI in the paper too. This closes the
  reviewer request for an "immediately-accessible, citable code snapshot."

## D. Before you push — checklist
- [ ] No raw EEG (`*.npz/*.npy/*.mat/*.edf`) committed (`.gitignore` handles this; verify with `git status`).
- [ ] No model weights (`*.pt/*.pth/*.h5`) committed.
- [ ] `requirements.txt` present and matches the reported versions.
- [ ] `python scripts/audit_project.py` still reports all values `OK` on the copied `results/`.
- [ ] README renders correctly on GitHub (tables, links).
