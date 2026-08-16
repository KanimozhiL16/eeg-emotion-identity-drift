# RELEASE checklist — push from the SHARED box without the wrong-contributor mistake

This box (`/lp-dev/24PHD1237/...`) is a **shared environment**. Git stamps every commit
with whatever `user.name` / `user.email` is configured, and GitHub decides who is a
"contributor" from the **commit email**. If the global config holds a lab-mate's identity,
their name will appear on *your* repo. Do the steps below **in order, before committing**.

> Golden rule: never use `git config --global` on a shared machine. Configure identity
> and credentials **per-repository** only.

---

## STEP 0 — Go to your clone
```bash
cd /lp-dev/24PHD1237/P4_SEEDV_COGNITIVE_BIOMETRIC        # your local clone of the repo
git remote -v                                            # MUST be YOUR repo:
#   origin  https://github.com/KanimozhiL16/eeg-emotion-identity-drift.git
```

## STEP 1 — See who git currently thinks you are (the trap)
```bash
git config --list --show-origin | grep -E 'user\.(name|email)'
```
If a `.../etc/gitconfig` or `~/.gitconfig` (global) line shows someone else's name/email,
that is what would be stamped on your commit. Fix it in Step 2 (do NOT edit the global file).

## STEP 2 — Set YOUR identity for THIS repo only
```bash
git config user.name  "Kanimozhi L"
git config user.email "KanimozhiL16@users.noreply.github.com"   # or the exact ID+username no-reply
```
- Get the exact no-reply address at GitHub → Settings → Emails → "Keep my email addresses
  private" (looks like `12345678+KanimozhiL16@users.noreply.github.com`). Using the
  no-reply email guarantees attribution to your account and leaks no real email.
- This writes to `.git/config` (repo-local); it does not affect other users, and their
  global config no longer overrides your commits here.

Verify:
```bash
git config user.name && git config user.email        # must print YOU
```

## STEP 3 — Do NOT cache credentials on the shared box
```bash
git config --unset-all credential.helper 2>/dev/null # no shared credential store
```
Authenticate with **your own fine-grained Personal Access Token** (GitHub → Settings →
Developer settings → Fine-grained tokens; scope = this repo only; Contents: Read/write).
Enter it when prompted at push time; do not save it to disk on this machine.

## STEP 4 — Stage and commit as yourself
```bash
git add -A
git status                                            # review what will go in
git commit -m "Sync analysis code with paper (leakage-free trial-disjoint); add REPRODUCE.md, environment.yml"
```

## STEP 5 — VERIFY AUTHORSHIP BEFORE PUSH (the actual safeguard)
```bash
git log --oneline --format='%h  %an <%ae>'  origin/main..HEAD
```
**Every line must be you.** If any pending (un-pushed) commit shows the wrong author:
```bash
# last commit only:
git commit --amend --author="Kanimozhi L <KanimozhiL16@users.noreply.github.com>" --no-edit

# several commits (interactive): mark each 'edit', then for each:
git rebase -i origin/main
git commit --amend --author="Kanimozhi L <KanimozhiL16@users.noreply.github.com>" --no-edit
git rebase --continue
```

## STEP 6 — Push, tag, verify contributors
```bash
git push origin main
git tag -a v1.0 -m "Reproducibility release for TAFFC submission"
git push origin v1.0
```
Then open and confirm **only you** are listed:
```
https://github.com/KanimozhiL16/eeg-emotion-identity-drift/graphs/contributors
```
(The list is by commit email — Step 2 is what controls it.)

## If a wrong contributor ALREADY appears (from an earlier push)
The commit email is baked into history, so it must be rewritten:
```bash
pip install git-filter-repo
cat > mailmap.txt <<'EOF'
Kanimozhi L <KanimozhiL16@users.noreply.github.com> <wrong.email@shared.box>
EOF
git filter-repo --mailmap mailmap.txt      # rewrites history
git push --force-with-lease origin main    # coordinate with any co-authors first
```
> History rewrite + force-push is disruptive — only do it if a wrong contributor is
> genuinely present, and make a backup branch first (`git branch backup-before-mailmap`).

---

### Quick pre-push self-check (run every time)
```bash
git config user.email                                   # -> your no-reply email
git log --format='%ae' origin/main..HEAD | sort -u      # -> ONLY your no-reply email
```
If both show only your email, no one else can appear as a contributor from this push.
