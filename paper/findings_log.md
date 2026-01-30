# AMR Pipeline – Findings Log (Frozen)

This document records all verified results, decisions, and interpretations
so we do not lose scientific context later.

DO NOT EDIT past sections. Append only.

---

## 1. Dataset & Validity

- Species: Klebsiella pneumoniae
- Sources: BV-BRC + NCBI
- Phenotypes: laboratory-only, R/S only
- Intermediates removed
- Computational phenotypes excluded

### Deduplication
- Mash-based star clustering
- Threshold: ~99.9% similarity
- Result: ~1,800 unique genomes kept
- Leakage check: 0 cross-split edges

### Splits (cluster-safe)
- Train: ~362k genomes
- Val: ~47k
- Test: ~47k
- No family overlap across splits (verified)

This is **leakage-free** and defensible.

---

## 2. Feature Engineering

### Raw features
- k = 31
- hashed presence/absence
- dim = 200,000
- deterministic

### Projection
- Sparse random projection
- dim = 8192
- deterministic seed
- applied to all splits

### Representation learning
- Autoencoder trained on train only
- latent dim = 256
- used for all downstream models
- improves stability and downstream performance

---

## 3. Baseline Results (Logistic Regression)

### Raw k-mer LR
- AUROC ~0.55–0.75
- AUPRC low on rare drugs
- expected under cluster-safe splits

### Latent LR
- improves over raw k-mers
- still limited on hard drugs (carbapenems, aminoglycosides)

---

## 4. Diffusion Augmentation

- Class-conditional latent DDPM
- Train-only
- T=100, epochs=40
- CPU-friendly
- Minority class augmented per drug

### Result:
Diffusion helps selectively, not universally.

Helps:
- piperacillin-tazobactam
- levofloxacin
- cefepime

Hurts or neutral:
- imipenem
- meropenem
- TMP-SMX

Conclusion:
Diffusion is drug-dependent. This is a real scientific finding, not a failure.

---

## 5. Stronger Classifiers (MAJOR RESULT)

Models tested:
- Logistic Regression
- Linear SVM (Platt calibrated)
- XGBoost (tree-based)

### Winner: XGBoost on AE latents

Consistent AUPRC gains across hard drugs:

- gentamicin: +0.142
- imipenem: +0.127
- piptazo: +0.078
- cefoxitin: +0.078
- meropenem: +0.081
- tobramycin: +0.066
- levofloxacin: +0.061
- cefepime: +0.038

SVM helps, but XGB is stronger and more consistent.

### Final model stack (LOCKED)
Autoencoder latents → XGBoost → no diffusion (default)
Diffusion only used when it helps (secondary analysis)

---

## 6. AMRFinder Benchmark (in progress)

- AMRFinder rerun on exact ML test FASTAs
- Correct ID alignment (accn_*)
- Organism specified: Klebsiella pneumoniae
- --plus enabled
- DB version: 2026-01-21.1
- TSVs being generated for all test genomes

Fair comparison will include:
- AMRFinder
- LR
- XGB
- XGB + diffusion (where helpful)

---

## 7. What is Novel

- True cluster-safe evaluation at scale
- Representation learning helps AMR
- Tree models outperform linear on latents
- Diffusion augmentation is conditional, not universal
- Fair AMRFinder comparison on identical test set

This is publishable.

---

## 8. Frozen Decisions (DO NOT CHANGE)

- Splits frozen
- Dedup frozen
- Feature pipeline frozen
- Latent dim = 256 frozen
- Main model = XGBoost frozen
- Diffusion = ablation only
- Metrics = AUROC + AUPRC

