#!/usr/bin/env python3
"""
Test performance on clean deduplicated data
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

print("="*80)
print("TESTING ON CLEAN DATA")
print("="*80)

# Load clean data
X = np.load("data/ml/X_k6_clean.npy")
pheno = pd.read_csv("data/merged/phenotypes_clean.csv")

with open("data/ml/genome_ids_clean.txt") as f:
    genome_ids = [line.strip() for line in f]

genome_to_idx = {gid: i for i, gid in enumerate(genome_ids)}

print(f"Genomes: {len(genome_ids)}")
print(f"Phenotype records: {len(pheno)}")

# Test on antibiotics with most samples
top_abs = pheno.groupby('antibiotic').size().sort_values(ascending=False).head(10)

print(f"\nTop 10 antibiotics:")
for ab, count in top_abs.items():
    print(f"  {ab}: {count} samples")

results = []

for antibiotic in top_abs.index:
    ab_data = pheno[pheno['antibiotic'] == antibiotic]
    
    indices = []
    labels = []
    
    for _, row in ab_data.iterrows():
        gid = row['genome_id']
        if gid in genome_to_idx:
            indices.append(genome_to_idx[gid])
            labels.append(1 if row['phenotype'] == 'R' else 0)
    
    if len(set(labels)) < 2 or len(indices) < 30:
        continue
    
    X_ab = X[indices]
    y = np.array(labels)
    
    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aurocs = []
    
    for train_idx, test_idx in cv.split(X_ab, y):
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        clf.fit(X_ab[train_idx], y[train_idx])
        y_pred = clf.predict_proba(X_ab[test_idx])[:, 1]
        aurocs.append(roc_auc_score(y[test_idx], y_pred))
    
    results.append({
        'antibiotic': antibiotic,
        'n_samples': len(indices),
        'auroc': np.mean(aurocs),
        'auroc_std': np.std(aurocs)
    })

df = pd.DataFrame(results)

print("\n" + "="*80)
print("RESULTS ON CLEAN DATA (RAW K-MERS)")
print("="*80)

print(f"\n{'Antibiotic':<25} {'N':<8} {'AUROC':<15}")
print("-"*60)

for _, row in df.iterrows():
    print(f"{row['antibiotic']:<25} {row['n_samples']:<8} {row['auroc']:.3f}±{row['auroc_std']:.3f}")

print(f"\nMean AUROC: {df['auroc'].mean():.3f}")
print(f"Median AUROC: {df['auroc'].median():.3f}")

df.to_csv("results/clean_data_performance.csv", index=False)
print("\n✓ Saved: results/clean_data_performance.csv")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

if df['auroc'].mean() >= 0.75:
    print("✓ Excellent performance on clean data!")
    print("  Dataset is high quality after deduplication")
elif df['auroc'].mean() >= 0.65:
    print("○ Good performance on clean data")
    print("  Results are honest and defensible")
elif df['auroc'].mean() >= 0.55:
    print("⚠ Moderate performance")
    print("  Consider increasing k-mer size (k=7 or k=8)")
else:
    print("✗ Poor performance")
    print("  Need more data or different features")

print("="*80)

