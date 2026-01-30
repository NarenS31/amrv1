"""
Compare ALL methods: AMRFinder vs Pipeline 1 vs Pipeline A vs Pipeline B
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

print("="*80)
print("FINAL COMPARISON: AMRFinder vs ML Pipelines")
print("="*80)

# Load AMRFinder benchmark
amr_bench = pd.read_csv("results/benchmark_fixed.csv")

# Load embeddings
Z_standard = np.load("data/ml/Z_all_complete.npy")
Z_masked = np.load("data/ml/Z_masked_ae.npy")

# Load phenotypes
pheno = pd.read_csv("data/merged/phenotypes_final.csv")

with open("data/ml/genome_ids_all_complete.txt") as f:
    genome_ids = [line.strip().split('.')[0] for line in f]

genome_to_idx = {gid: i for i, gid in enumerate(genome_ids)}

# Get antibiotics in AMRFinder benchmark
amr_antibiotics = amr_bench['antibiotic'].unique()

results = []

for antibiotic in amr_antibiotics:
    print(f"Testing {antibiotic}...")
    
    # Get ML data for this antibiotic
    ab_data = pheno[pheno['antibiotic'] == antibiotic.lower()]
    
    indices = []
    labels = []
    for _, row in ab_data.iterrows():
        gid = row['genome_id']
        if gid in genome_to_idx:
            indices.append(genome_to_idx[gid])
            labels.append(1 if row['phenotype'] == 'R' else 0)
    
    if len(set(labels)) < 2 or len(indices) < 30:
        continue
    
    X_std = Z_standard[indices]
    X_mask = Z_masked[indices]
    y = np.array(labels)
    
    # 5-fold CV for ML models
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Standard AE
    aurocs_std = []
    recalls_std = []
    for train_idx, test_idx in cv.split(X_std, y):
        clf = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
        clf.fit(X_std[train_idx], y[train_idx])
        y_pred_proba = clf.predict_proba(X_std[test_idx])[:, 1]
        y_pred = clf.predict(X_std[test_idx])
        aurocs_std.append(roc_auc_score(y[test_idx], y_pred_proba))
        recalls_std.append(recall_score(y[test_idx], y_pred))
    
    # Masked AE
    aurocs_mask = []
    recalls_mask = []
    for train_idx, test_idx in cv.split(X_mask, y):
        clf = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
        clf.fit(X_mask[train_idx], y[train_idx])
        y_pred_proba = clf.predict_proba(X_mask[test_idx])[:, 1]
        y_pred = clf.predict(X_mask[test_idx])
        aurocs_mask.append(roc_auc_score(y[test_idx], y_pred_proba))
        recalls_mask.append(recall_score(y[test_idx], y_pred))
    
    # AMRFinder recall on this antibiotic
    amr_ab = amr_bench[amr_bench['antibiotic'] == antibiotic]
    tp = len(amr_ab[(amr_ab['true_phenotype'] == 'R') & (amr_ab['predicted_phenotype'] == 'R')])
    fn = len(amr_ab[(amr_ab['true_phenotype'] == 'R') & (amr_ab['predicted_phenotype'] != 'R')])
    amr_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    results.append({
        'antibiotic': antibiotic,
        'n_samples': len(indices),
        'amrfinder_recall': amr_recall,
        'pipeline1_auroc': np.mean(aurocs_std),
        'pipeline1_recall': np.mean(recalls_std),
        'pipelineA_auroc': np.mean(aurocs_mask),
        'pipelineA_recall': np.mean(recalls_mask)
    })

df = pd.DataFrame(results)

print("\n" + "="*80)
print("RESULTS BY ANTIBIOTIC")
print("="*80)
print(f"\n{'Antibiotic':<35} {'AMRFinder':>12} {'Pipeline 1':>12} {'Pipeline A':>12}")
print(f"{'':35} {'Recall':>12} {'Recall':>12} {'Recall':>12}")
print("-"*80)

for _, row in df.iterrows():
    print(f"{row['antibiotic']:<35} {row['amrfinder_recall']:>12.3f} "
          f"{row['pipeline1_recall']:>12.3f} {row['pipelineA_recall']:>12.3f}")

print("\n" + "="*80)
print("OVERALL AVERAGES")
print("="*80)
print(f"AMRFinder recall:       {df['amrfinder_recall'].mean():.3f}")
print(f"Pipeline 1 AUROC:       {df['pipeline1_auroc'].mean():.3f}")
print(f"Pipeline 1 Recall:      {df['pipeline1_recall'].mean():.3f}")
print(f"Pipeline A AUROC:       {df['pipelineA_auroc'].mean():.3f}")
print(f"Pipeline A Recall:      {df['pipelineA_recall'].mean():.3f}")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)

if df['pipelineA_recall'].mean() > df['amrfinder_recall'].mean():
    print(f"✓ Pipeline A outperforms AMRFinder by {((df['pipelineA_recall'].mean() - df['amrfinder_recall'].mean()) / df['amrfinder_recall'].mean() * 100):.1f}%")
else:
    print(f"✗ AMRFinder outperforms Pipeline A by {((df['amrfinder_recall'].mean() - df['pipelineA_recall'].mean()) / df['pipelineA_recall'].mean() * 100):.1f}%")

print(f"✓ Pipeline A (masked) improves over Pipeline 1 by {((df['pipelineA_auroc'].mean() - df['pipeline1_auroc'].mean()) / df['pipeline1_auroc'].mean() * 100):.1f}%")
print("="*80)

df.to_csv("results/final_comparison.csv", index=False)
print("\n✓ Saved: results/final_comparison.csv")

