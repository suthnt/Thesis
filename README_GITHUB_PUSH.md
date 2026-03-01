# Pushing to GitHub Thesis Repo

## Option A: You already have a thesis repo

1. Clone your thesis repo (if not already):
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_THESIS_REPO.git
   cd YOUR_THESIS_REPO
   ```

2. Copy this folder's contents into your thesis repo (e.g. into a `cnn/` or `models/` subfolder):
   ```bash
   cp -r /scratch/gpfs/ALAINK/Suthi/thesis_cnn_export/* ./cnn/   # or your preferred path
   ```

3. Add, commit, push:
   ```bash
   git add .
   git commit -m "Add CNN training scripts and Slurm configs"
   git push origin main
   ```

## Option B: Create new repo from this export

1. cd into the export folder:
   ```bash
   cd /scratch/gpfs/ALAINK/Suthi/thesis_cnn_export
   ```

2. Init git and push to new repo:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: CNN crash classification code"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/thesis.git
   git push -u origin main
   ```

Replace YOUR_USERNAME and YOUR_THESIS_REPO with your actual GitHub username and repo name.
