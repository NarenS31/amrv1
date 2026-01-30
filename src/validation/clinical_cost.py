"""
Clinical cost analysis - false negatives kill patients
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

print("="*80)
print("CLINICAL COST ANALYSIS")
print("="*80)

Z = np.load("data/ml/Z_masked_ae.npy")
pheno = pd.read_csv("data/merged/phenotypes_final.csv")

with open("data/ml/genome_ids_all_complete.txt") as f:
    genome_ids = [line.strip().split('.')[0] for line in f]

genome_to_idx = {gid: i for i, gid in enumerate(genome_ids)}

# Get top antibiotic
top_ab = pheno.groupby('antibiotic').size().sort_values(ascending=False).index[0]
ab_data = pheno[pheno['antibiotic'] == top_ab]

indices = []
labels = []
for _, row in ab_data.iterrows():
    gid = row['genome_id']
    if gid in genome_to_idx:
        indices.append(genome_to_idx[gid])
        labels.append(1 if row['phenotype'] == 'R' else 0)

X = Z[indices]
y = np.array(labels)

# Train model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_y_true = []
all_y_pred = []

for train_idx, test_idx in cv.split(X, y):
    clf = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    clf.fit(X[train_idx], y[train_idx])
    y_pred = clf.predict(X[test_idx])
    
    all_y_true.extend(y[test_idx])
    all_y_pred.extend(y_pred)

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(all_y_true, all_y_pred).ravel()

print(f"\nConfusion Matrix for {top_ab}:")
print(f"True Negatives (S predicted S):  {tn}")
print(f"False Positives (S predicted R): {fp}")
print(f"False Negatives (R predicted S): {fn}  ← DANGEROUS")
print(f"True Positives (R predicted R):  {tp}")

print("\n" + "="*80)
print("CLINICAL IMPACT")
print("="*80)

# Assign costs
cost_fn = 100  # False negative = patient death
cost_fp = 10   # False positive = unnecessary broad-spectrum antibiotic
cost_tn = 0
cost_tp = 0

total_cost = (tn * cost_tn + fp * cost_fp + fn * cost_fn + tp * cost_tp)

print(f"\nClinical cost (arbitrary units):")
print(f"  False negatives: {fn} × {cost_fn} = {fn * cost_fn}")
print(f"  False positives: {fp} × {cost_fp} = {fp * cost_fp}")
print(f"  Total cost: {total_cost}")

print(f"\nFalse negative rate: {fn / (fn + tp):.1%}")
print(f"  → In {fn} out of {fn + tp} resistant cases, we predict susceptible")
print(f"  → These patients would receive ineffective treatment")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if fn / (fn + tp) > 0.2:
    print("⚠️ High false negative rate (>20%)")
    print("  → Model should NOT be used alone for clinical decisions")
    print("  → Combine with AST confirmation for high-stakes cases")
else:
    print("✓ Acceptable false negative rate (<20%)")
    print("  → Still recommend AST confirmation when feasible")

print("="*80)

