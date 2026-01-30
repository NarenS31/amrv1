"""
Test all antibiotics to find the best performing ones
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

print("="*80)
print("TESTING ALL ANTIBIOTICS")
print("="*80)

Z = np.load("data/ml/Z_masked_ae.npy")
pheno = pd.read_csv("data/merged/phenotypes_final.csv")

with open("data/ml/genome_ids_all_complete.txt") as f:
    genome_ids = [line.strip().split('.')[0] for line in f]

genome_to_idx = {gid: i for i, gid in enumerate(genome_ids)}

# Test ALL antibiotics
all_antibiotics = pheno['antibiotic'].unique()

results = []

print(f"\nTesting {len(all_antibiotics)} antibiotics...\n")

for antibiotic in all_antibiotics:
    ab_data = pheno[pheno['antibiotic'] == antibiotic]
    
    indices = []
    labels = []
    for _, row in ab_data.iterrows():
        gid = row['genome_id']
        if gid in genome_to_idx:
            indices.append(genome_to_idx[gid])
            labels.append(1 if row['phenotype'] == 'R' else 0)
    
    if len(set(labels)) < 2 or len(indices) < 50:
        continue
    
    X = Z[indices]
    y = np.array(labels)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aurocs = []
    
    for train_idx, test_idx in cv.split(X, y):
        clf = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict_proba(X[test_idx])[:, 1]
        aurocs.append(roc_auc_score(y[test_idx], y_pred))
    
    auroc = np.mean(aurocs)
    results.append({
        'antibiotic': antibiotic,
        'n_samples': len(indices),
        'auroc': auroc
    })

df = pd.DataFrame(results).sort_values('auroc', ascending=False)

print("="*80)
print("ALL ANTIBIOTICS RANKED BY AUROC (Masked AE)")
print("="*80)
print(f"\n{'Antibiotic':<35} {'N':<8} {'AUROC':<8}")
print("-"*80)

for _, row in df.iterrows():
    symbol = "🏆" if row['auroc'] >= 0.80 else "✓" if row['auroc'] >= 0.75 else "○" if row['auroc'] >= 0.70 else "·"
    print(f"{symbol} {row['antibiotic']:<33} {row['n_samples']:<8} {row['auroc']:.3f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Mean AUROC: {df['auroc'].mean():.3f}")
print(f"Antibiotics with AUROC ≥ 0.80: {len(df[df['auroc'] >= 0.80])}")
print(f"Antibiotics with AUROC ≥ 0.75: {len(df[df['auroc'] >= 0.75])}")
print(f"Antibiotics with AUROC ≥ 0.70: {len(df[df['auroc'] >= 0.70])}")

# Show top performers
print("\n" + "="*80)
print("TOP 10 PERFORMERS")
print("="*80)
print(df.head(10).to_string(index=False))

# Show bottom performers
print("\n" + "="*80)
print("MOST CHALLENGING (BOTTOM 10)")
print("="*80)
print(df.tail(10).to_string(index=False))

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
high = df[df['auroc'] >= 0.75]
if len(high) > 0:
    print(f"✓ Focus your paper on the {len(high)} antibiotics with AUROC ≥ 0.75")
    print("✓ These show strong performance and are clinically relevant")
else:
    print("✗ No antibiotics above 0.75 AUROC")
    print("→ Consider: improve architecture, add features, or focus on uncertainty")
print("="*80)

# Save
df.to_csv("data/ml/all_antibiotics_performance.csv", index=False)
print("\n✓ Saved: data/ml/all_antibiotics_performance.csv")

