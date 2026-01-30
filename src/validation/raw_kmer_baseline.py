"""
Test classifier on RAW k-mers (no autoencoder)
Proves autoencoder actually improves performance
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

print("="*80)
print("RAW K-MER BASELINE (No Autoencoder)")
print("="*80)

# Load RAW k-mers (not embeddings)
X_raw = np.load("data/combined/X_k6_unique.npy")
Z_masked = np.load("data/ml/Z_masked_ae.npy")

pheno = pd.read_csv("data/merged/phenotypes_final.csv")

with open("data/ml/genome_ids_all_complete.txt") as f:
    genome_ids = [line.strip().split('.')[0] for line in f]

genome_to_idx = {gid: i for i, gid in enumerate(genome_ids)}

# Test on top 10 antibiotics
top_abs = pheno.groupby('antibiotic').size().sort_values(ascending=False).head(10).index

results = []

for antibiotic in top_abs:
    print(f"Testing {antibiotic}...")
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
    
    # RAW k-mers
    X_raw_sub = X_raw[indices]
    # Masked AE embeddings
    X_emb_sub = Z_masked[indices]
    y = np.array(labels)
    
    # Scale raw k-mers
    scaler = StandardScaler(with_mean=False)  # Sparse data
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Test raw k-mers
    aurocs_raw = []
    for train_idx, test_idx in cv.split(X_raw_sub, y):
        X_train_scaled = scaler.fit_transform(X_raw_sub[train_idx])
        X_test_scaled = scaler.transform(X_raw_sub[test_idx])
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y[train_idx])
        y_pred = clf.predict_proba(X_test_scaled)[:, 1]
        aurocs_raw.append(roc_auc_score(y[test_idx], y_pred))
    
    # Test masked embeddings
    aurocs_emb = []
    for train_idx, test_idx in cv.split(X_emb_sub, y):
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_emb_sub[train_idx], y[train_idx])
        y_pred = clf.predict_proba(X_emb_sub[test_idx])[:, 1]
        aurocs_emb.append(roc_auc_score(y[test_idx], y_pred))
    
    results.append({
        'antibiotic': antibiotic,
        'n': len(indices),
        'raw_auroc': np.mean(aurocs_raw),
        'masked_auroc': np.mean(aurocs_emb),
        'improvement': np.mean(aurocs_emb) - np.mean(aurocs_raw)
    })

df = pd.DataFrame(results)

print("\n" + "="*80)
print("RAW K-MERS vs MASKED AE")
print("="*80)

print(f"\n{'Antibiotic':<30} {'Raw K-mers':<12} {'Masked AE':<12} {'Improvement'}")
print("-"*80)

for _, row in df.iterrows():
    print(f"{row['antibiotic']:<30} {row['raw_auroc']:.3f}       "
          f"{row['masked_auroc']:.3f}       {row['improvement']:+.3f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Raw k-mers mean AUROC:  {df['raw_auroc'].mean():.3f}")
print(f"Masked AE mean AUROC:   {df['masked_auroc'].mean():.3f}")
print(f"Average improvement:    {df['improvement'].mean():+.3f}")

if df['improvement'].mean() > 0:
    print("\n✓ Masked AE improves over raw k-mers")
    print("  → Autoencoder provides useful dimensionality reduction")
else:
    print("\n✗ Raw k-mers perform as well or better")
    print("  → Autoencoder may not be necessary")

df.to_csv("results/raw_kmer_baseline.csv", index=False)
print("\n✓ Saved: results/raw_kmer_baseline.csv")

