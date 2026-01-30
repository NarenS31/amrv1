"""
Test robustness of Standard AE vs Masked AE to corruption
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import sys
sys.path.append('src/src_nn')
from masked_autoencoder import MaskedAutoencoder

print("="*80)
print("ROBUSTNESS TEST: CORRUPTED INPUT")
print("="*80)

# Load raw k-mers
X_raw = np.load("data/combined/X_k6_unique.npy")

import pickle
with open("models/scaler_masked.pkl", 'rb') as f:
    scaler = pickle.load(f)

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load Masked AE
masked_model = MaskedAutoencoder().to(device)
masked_model.load_state_dict(torch.load("models/masked_ae_final.pt", map_location=device))
masked_model.eval()

# Load Standard AE (full model with decoder)
class StandardAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 512)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 4096)
        )
    
    def encode(self, x):
        return self.encoder(x)

standard_model = StandardAE().to(device)
standard_model.load_state_dict(torch.load("models/pretrained_ae_20k.pt", map_location=device))
standard_model.eval()

# Load phenotypes
pheno = pd.read_csv("data/merged/phenotypes_final.csv")

with open("data/ml/genome_ids_all_complete.txt") as f:
    genome_ids = [line.strip().split('.')[0] for line in f]

genome_to_idx = {gid: i for i, gid in enumerate(genome_ids)}

# Get one antibiotic
top_ab = pheno.groupby('antibiotic').size().sort_values(ascending=False).index[0]
ab_data = pheno[pheno['antibiotic'] == top_ab]

indices = []
labels = []
for _, row in ab_data.iterrows():
    gid = row['genome_id']
    if gid in genome_to_idx:
        indices.append(genome_to_idx[gid])
        labels.append(1 if row['phenotype'] == 'R' else 0)

X_subset = X_raw[indices]
y = np.array(labels)

# Test different corruption levels
corruption_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

print(f"\nTesting on {top_ab} (n={len(indices)})\n")
print("Corruption | Standard AE | Masked AE | Δ")
print("-" * 50)

for corruption in corruption_levels:
    # Corrupt data (set random k-mers to 0)
    X_corrupted = X_subset.copy()
    if corruption > 0:
        mask = np.random.random(X_corrupted.shape) < corruption
        X_corrupted[mask] = 0
    
    # Normalize
    X_scaled = scaler.transform(X_corrupted.astype(np.float32))
    
    # Extract embeddings
    with torch.no_grad():
        X_torch = torch.FloatTensor(X_scaled).to(device)
        Z_standard = standard_model.encode(X_torch).cpu().numpy()
        Z_masked = masked_model.encode(X_torch).cpu().numpy()
    
    # Test classifiers
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    aurocs_std = []
    for train_idx, test_idx in cv.split(Z_standard, y):
        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(Z_standard[train_idx], y[train_idx])
        y_pred = clf.predict_proba(Z_standard[test_idx])[:, 1]
        aurocs_std.append(roc_auc_score(y[test_idx], y_pred))
    
    aurocs_mask = []
    for train_idx, test_idx in cv.split(Z_masked, y):
        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(Z_masked[train_idx], y[train_idx])
        y_pred = clf.predict_proba(Z_masked[test_idx])[:, 1]
        aurocs_mask.append(roc_auc_score(y[test_idx], y_pred))
    
    auroc_std = np.mean(aurocs_std)
    auroc_mask = np.mean(aurocs_mask)
    diff = auroc_mask - auroc_std
    
    print(f"{corruption*100:5.0f}%     |    {auroc_std:.3f}    |   {auroc_mask:.3f}   | {diff:+.3f}")

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
print("If Masked AE degrades LESS than Standard AE → Robustness proven!")
print("Masked AE was trained with 30% masking, so should handle corruption better.")
print("="*80)

