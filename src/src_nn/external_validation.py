"""
External validation on held-out test set
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import torch
import sys
sys.path.append('src/src_nn')
from masked_autoencoder import MaskedAutoencoder

print("="*80)
print("EXTERNAL VALIDATION")
print("="*80)

# Load embeddings
Z_masked = np.load("data/ml/Z_masked_ae.npy")

# Load phenotypes
pheno = pd.read_csv("data/merged/phenotypes_final.csv")

with open("data/ml/genome_ids_all_complete.txt") as f:
    genome_ids = [line.strip().split('.')[0] for line in f]

genome_to_idx = {gid: i for i, gid in enumerate(genome_ids)}

# Get top antibiotic
top_ab = pheno.groupby('antibiotic').size().sort_values(ascending=False).index[0]
ab_data = pheno[pheno['antibiotic'] == top_ab]

# Create dataset
indices = []
labels = []
for _, row in ab_data.iterrows():
    gid = row['genome_id']
    if gid in genome_to_idx:
        indices.append(genome_to_idx[gid])
        labels.append(1 if row['phenotype'] == 'R' else 0)

X = Z_masked[indices]
y = np.array(labels)

# Split: 70% train, 30% test (completely held out)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nAntibiotic: {top_ab}")
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples (completely held out)")

# Train on training set
clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Test on held-out set
y_pred_proba = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)

# Metrics
auroc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*80)
print("HELD-OUT TEST SET RESULTS")
print("="*80)
print(f"AUROC: {auroc:.3f}")
print(f"Accuracy: {acc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Susceptible', 'Resistant']))

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
if auroc > 0.7:
    print("✓ Strong performance on held-out data - model generalizes well!")
elif auroc > 0.6:
    print("✓ Decent performance - model generalizes reasonably")
else:
    print("✗ Poor generalization - may be overfitting")
print("="*80)

