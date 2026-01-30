"""
Semi-supervised learning using unlabeled genomes
Pseudo-labeling: use confident predictions as training data
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class SemiSupervisedClassifier(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        return self.model(x)

print("="*80)
print("SEMI-SUPERVISED LEARNING")
print("="*80)

# Load embeddings
Z_all = np.load("data/ml/Z_masked_ae.npy")
pheno = pd.read_csv("data/merged/phenotypes_final.csv")

with open("data/ml/genome_ids_all_complete.txt") as f:
    all_genome_ids = [line.strip().split('.')[0] for line in f]

genome_to_idx = {gid: i for i, gid in enumerate(all_genome_ids)}

# Get labeled data for one antibiotic
top_ab = pheno.groupby('antibiotic').size().sort_values(ascending=False).index[0]
ab_data = pheno[pheno['antibiotic'] == top_ab]

labeled_indices = []
labels = []
for _, row in ab_data.iterrows():
    gid = row['genome_id']
    if gid in genome_to_idx:
        labeled_indices.append(genome_to_idx[gid])
        labels.append(1 if row['phenotype'] == 'R' else 0)

# Get unlabeled data
labeled_set = set(labeled_indices)
unlabeled_indices = [i for i in range(len(Z_all)) if i not in labeled_set]

print(f"Labeled genomes: {len(labeled_indices)}")
print(f"Unlabeled genomes: {len(unlabeled_indices)}")

X_labeled = Z_all[labeled_indices]
y_labeled = np.array(labels)
X_unlabeled = Z_all[unlabeled_indices]

# Split labeled into train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_labeled, y_labeled, test_size=0.4, random_state=42, stratify=y_labeled
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nTrain: {len(X_train)}")
print(f"Val: {len(X_val)}")  
print(f"Test: {len(X_test)}")

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Initial training on labeled data
model = SemiSupervisedClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

print("\n1. Training on labeled data only...")

for epoch in range(50):
    model.train()
    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.LongTensor(y_train).to(device)
    
    optimizer.zero_grad()
    outputs = model(X_t)
    loss = criterion(outputs, y_t)
    loss.backward()
    optimizer.step()

# Evaluate
model.eval()
with torch.no_grad():
    outputs = model(torch.FloatTensor(X_test).to(device))
    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

initial_auroc = roc_auc_score(y_test, probs)
print(f"Initial test AUROC: {initial_auroc:.3f}")

# Pseudo-labeling
print("\n2. Pseudo-labeling unlabeled data...")

model.eval()
with torch.no_grad():
    outputs = model(torch.FloatTensor(X_unlabeled).to(device))
    probs_unlabeled = torch.softmax(outputs, dim=1).cpu().numpy()

# Use high-confidence predictions (>0.9 or <0.1)
confident_mask = (probs_unlabeled[:, 1] > 0.9) | (probs_unlabeled[:, 1] < 0.1)
X_pseudo = X_unlabeled[confident_mask]
y_pseudo = (probs_unlabeled[confident_mask, 1] > 0.5).astype(int)

print(f"Confident pseudo-labels: {len(X_pseudo)}")

# Retrain with pseudo-labels
print("\n3. Retraining with pseudo-labels...")

X_combined = np.vstack([X_train, X_pseudo])
y_combined = np.concatenate([y_train, y_pseudo])

print(f"Combined training set: {len(X_combined)}")

model = SemiSupervisedClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

for epoch in range(50):
    model.train()
    X_t = torch.FloatTensor(X_combined).to(device)
    y_t = torch.LongTensor(y_combined).to(device)
    
    optimizer.zero_grad()
    outputs = model(X_t)
    loss = criterion(outputs, y_t)
    loss.backward()
    optimizer.step()

# Final evaluation
model.eval()
with torch.no_grad():
    outputs = model(torch.FloatTensor(X_test).to(device))
    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

final_auroc = roc_auc_score(y_test, probs)

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Initial AUROC (labeled only): {initial_auroc:.3f}")
print(f"Final AUROC (with pseudo-labels): {final_auroc:.3f}")
print(f"Improvement: {final_auroc - initial_auroc:+.3f}")
print("="*80)

