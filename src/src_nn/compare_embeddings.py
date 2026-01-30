"""
Compare Standard AE vs Masked AE embeddings
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

print("="*80)
print("COMPARING STANDARD AE vs MASKED AE")
print("="*80)

# Load both embeddings
Z_standard = np.load("data/ml/Z_all_complete.npy")
Z_masked = np.load("data/ml/Z_masked_ae.npy")

print(f"Standard AE embeddings: {Z_standard.shape}")
print(f"Masked AE embeddings: {Z_masked.shape}")

# Load phenotypes
pheno = pd.read_csv("data/merged/phenotypes_final.csv")
print(f"Phenotypes: {len(pheno):,}")

# Load genome ID mapping
with open("data/ml/genome_ids_all_complete.txt") as f:
    genome_ids = [line.strip().split('.')[0] for line in f]

genome_to_idx = {gid: i for i, gid in enumerate(genome_ids)}

# Get top 10 antibiotics
top_antibiotics = pheno.groupby('antibiotic').size().sort_values(ascending=False).head(10).index

results_standard = []
results_masked = []

print("\nTraining classifiers on top 10 antibiotics...\n")

for antibiotic in top_antibiotics:
    ab_data = pheno[pheno['antibiotic'] == antibiotic]
    
    # Create dataset
    indices = []
    labels = []
    
    for _, row in ab_data.iterrows():
        gid_norm = row['genome_id']
        if gid_norm in genome_to_idx:
            indices.append(genome_to_idx[gid_norm])
            labels.append(1 if row['phenotype'] == 'R' else 0)
    
    if len(set(labels)) < 2 or len(indices) < 30:
        continue
    
    # Standard AE
    X_std = Z_standard[indices]
    y = np.array(labels)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aurocs_std = []
    
    for train_idx, test_idx in cv.split(X_std, y):
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        clf.fit(X_std[train_idx], y[train_idx])
        
        y_pred = clf.predict_proba(X_std[test_idx])[:, 1]
        aurocs_std.append(roc_auc_score(y[test_idx], y_pred))
    
    # Masked AE
    X_mask = Z_masked[indices]
    aurocs_mask = []
    
    for train_idx, test_idx in cv.split(X_mask, y):
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        clf.fit(X_mask[train_idx], y[train_idx])
        
        y_pred = clf.predict_proba(X_mask[test_idx])[:, 1]
        aurocs_mask.append(roc_auc_score(y[test_idx], y_pred))
    
    # Store results
    auroc_std = np.mean(aurocs_std)
    auroc_mask = np.mean(aurocs_mask)
    improvement = ((auroc_mask - auroc_std) / auroc_std * 100)
    
    results_standard.append({
        'antibiotic': antibiotic,
        'n_samples': len(indices),
        'auroc': auroc_std
    })
    
    results_masked.append({
        'antibiotic': antibiotic,
        'n_samples': len(indices),
        'auroc': auroc_mask
    })
    
    symbol = "↑" if auroc_mask > auroc_std else "↓"
    print(f"{antibiotic:30s} (n={len(indices):4d}): "
          f"Standard={auroc_std:.3f}, Masked={auroc_mask:.3f} "
          f"{symbol} {improvement:+.1f}%")

# Save
df_std = pd.DataFrame(results_standard)
df_mask = pd.DataFrame(results_masked)

df_std.to_csv("data/ml/results_standard_ae.csv", index=False)
df_mask.to_csv("data/ml/results_masked_ae.csv", index=False)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Standard AE mean AUROC: {df_std['auroc'].mean():.3f}")
print(f"Masked AE mean AUROC:   {df_mask['auroc'].mean():.3f}")

improvement = ((df_mask['auroc'].mean() - df_std['auroc'].mean()) / df_std['auroc'].mean() * 100)
print(f"Overall improvement: {improvement:+.1f}%")

if improvement > 0:
    print("✓ Masked AE outperforms Standard AE!")
else:
    print("✗ Standard AE performs better")

print("="*80)

