"""
Test Pipeline A on ALL antibiotics (not just top 10)
Shows we're not cherry-picking
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats

print("="*80)
print("TESTING ALL ANTIBIOTICS (NO CHERRY-PICKING)")
print("="*80)

Z_masked = np.load("data/ml/Z_masked_ae.npy")
pheno = pd.read_csv("data/merged/phenotypes_final.csv")

with open("data/ml/genome_ids_all_complete.txt") as f:
    genome_ids = [line.strip().split('.')[0] for line in f]

genome_to_idx = {gid: i for i, gid in enumerate(genome_ids)}

# Test EVERY antibiotic with >= 30 samples
all_antibiotics = pheno.groupby('antibiotic').size()
testable_antibiotics = all_antibiotics[all_antibiotics >= 30].sort_values(ascending=False)

print(f"Total antibiotics in dataset: {len(all_antibiotics)}")
print(f"Antibiotics with ≥30 samples: {len(testable_antibiotics)}")

results = []

for antibiotic in testable_antibiotics.index:
    ab_data = pheno[pheno['antibiotic'] == antibiotic]
    
    indices = []
    labels = []
    for _, row in ab_data.iterrows():
        gid = row['genome_id']
        if gid in genome_to_idx:
            indices.append(genome_to_idx[gid])
            labels.append(1 if row['phenotype'] == 'R' else 0)
    
    if len(set(labels)) < 2:
        continue
    
    X = Z_masked[indices]
    y = np.array(labels)
    
    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aurocs = []
    
    for train_idx, test_idx in cv.split(X, y):
        clf = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict_proba(X[test_idx])[:, 1]
        aurocs.append(roc_auc_score(y[test_idx], y_pred))
    
    results.append({
        'antibiotic': antibiotic,
        'n_samples': len(indices),
        'auroc_mean': np.mean(aurocs),
        'auroc_std': np.std(aurocs)
    })

df = pd.DataFrame(results).sort_values('auroc_mean', ascending=False)

print("\n" + "="*80)
print(f"RESULTS FOR ALL {len(df)} ANTIBIOTICS")
print("="*80)

print(f"\n{'Rank':<6} {'Antibiotic':<35} {'N':<8} {'AUROC':<15} {'Performance'}")
print("-"*90)

for i, (_, row) in enumerate(df.iterrows(), 1):
    if row['auroc_mean'] >= 0.80:
        perf = "Excellent ✓✓✓"
    elif row['auroc_mean'] >= 0.75:
        perf = "Good ✓✓"
    elif row['auroc_mean'] >= 0.70:
        perf = "Fair ✓"
    else:
        perf = "Poor"
    
    print(f"{i:<6} {row['antibiotic']:<35} {row['n_samples']:<8} "
          f"{row['auroc_mean']:.3f}±{row['auroc_std']:.3f}   {perf}")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"Mean AUROC across all antibiotics: {df['auroc_mean'].mean():.3f}")
print(f"Median AUROC: {df['auroc_mean'].median():.3f}")
print(f"Best AUROC: {df['auroc_mean'].max():.3f} ({df.iloc[0]['antibiotic']})")
print(f"Worst AUROC: {df['auroc_mean'].min():.3f} ({df.iloc[-1]['antibiotic']})")

print(f"\nAntibiotics with AUROC ≥ 0.80: {(df['auroc_mean'] >= 0.80).sum()}/{len(df)}")
print(f"Antibiotics with AUROC ≥ 0.75: {(df['auroc_mean'] >= 0.75).sum()}/{len(df)}")
print(f"Antibiotics with AUROC ≥ 0.70: {(df['auroc_mean'] >= 0.70).sum()}/{len(df)}")
print(f"Antibiotics with AUROC < 0.60: {(df['auroc_mean'] < 0.60).sum()}/{len(df)}")

df.to_csv("results/all_antibiotics_performance.csv", index=False)
print("\n✓ Saved: results/all_antibiotics_performance.csv")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print("This shows performance across ALL antibiotics, not just cherry-picked ones")
print("Variation is expected - some antibiotics are harder to predict than others")
print("="*80)

