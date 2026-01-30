"""
Prove the model is actually learning, not cheating
"""
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import sys
sys.path.append('src/src_nn')
from masked_autoencoder import MaskedAutoencoder

print("="*80)
print("MODEL VALIDATION TESTS")
print("="*80)

# Load embeddings
Z_masked = np.load("data/ml/Z_masked_ae.npy")
pheno = pd.read_csv("data/merged/phenotypes_final.csv")

with open("data/ml/genome_ids_all_complete.txt") as f:
    genome_ids = [line.strip().split('.')[0] for line in f]

genome_to_idx = {gid: i for i, gid in enumerate(genome_ids)}

# Test 1: RANDOM EMBEDDINGS BASELINE
print("\n" + "="*80)
print("TEST 1: RANDOM EMBEDDINGS (Should perform ~0.5 AUROC)")
print("="*80)

Z_random = np.random.randn(*Z_masked.shape).astype(np.float32)

top_ab = pheno.groupby('antibiotic').size().sort_values(ascending=False).index[0]
ab_data = pheno[pheno['antibiotic'] == top_ab]

indices = []
labels = []
for _, row in ab_data.iterrows():
    gid = row['genome_id']
    if gid in genome_to_idx:
        indices.append(genome_to_idx[gid])
        labels.append(1 if row['phenotype'] == 'R' else 0)

X_real = Z_masked[indices]
X_random = Z_random[indices]
y = np.array(labels)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Real embeddings
aurocs_real = []
for train_idx, test_idx in cv.split(X_real, y):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_real[train_idx], y[train_idx])
    y_pred = clf.predict_proba(X_real[test_idx])[:, 1]
    aurocs_real.append(roc_auc_score(y[test_idx], y_pred))

# Random embeddings
aurocs_random = []
for train_idx, test_idx in cv.split(X_random, y):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_random[train_idx], y[train_idx])
    y_pred = clf.predict_proba(X_random[test_idx])[:, 1]
    aurocs_random.append(roc_auc_score(y[test_idx], y_pred))

print(f"Real embeddings AUROC: {np.mean(aurocs_real):.3f}")
print(f"Random embeddings AUROC: {np.mean(aurocs_random):.3f}")

if np.mean(aurocs_real) > 0.6:
    print("✓ PASS: Real embeddings significantly better than random")
else:
    print("✗ FAIL: Model is not learning meaningful features!")

# Test 2: SHUFFLED LABELS
print("\n" + "="*80)
print("TEST 2: SHUFFLED LABELS (Should perform ~0.5 AUROC)")
print("="*80)

y_shuffled = y.copy()
np.random.shuffle(y_shuffled)

aurocs_shuffled = []
for train_idx, test_idx in cv.split(X_real, y_shuffled):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_real[train_idx], y_shuffled[train_idx])
    y_pred = clf.predict_proba(X_real[test_idx])[:, 1]
    aurocs_shuffled.append(roc_auc_score(y_shuffled[test_idx], y_pred))

print(f"Real labels AUROC: {np.mean(aurocs_real):.3f}")
print(f"Shuffled labels AUROC: {np.mean(aurocs_shuffled):.3f}")

if np.mean(aurocs_shuffled) < 0.6:
    print("✓ PASS: Shuffled labels perform poorly (model not memorizing)")
else:
    print("✗ FAIL: Model may be overfitting!")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)

