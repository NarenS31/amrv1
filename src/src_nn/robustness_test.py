"""
Test robustness to missing data
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

print("="*80)
print("ROBUSTNESS TESTING")
print("="*80)

# Load data
Z = np.load("data/ml/Z_pretrained_20k.npy")
pheno = pd.read_csv("data/merged/all_phenotypes.csv")

# Get top antibiotic
top_ab = pheno['antibiotic'].value_counts().index[0]
print(f"Testing on: {top_ab}")

ab_data = pheno[pheno['antibiotic'] == top_ab]

# Create dataset (same as before)
labeled_indices = np.load("data/ml/labeled_indices.npy")
with open("data/combined/genome_ids_all.txt") as f:
    all_ids = [line.strip() for line in f]
labeled_genome_ids = [all_ids[i] for i in labeled_indices]
genome_to_idx = {gid: i for i, gid in enumerate(labeled_genome_ids)}

indices = []
labels = []
for _, row in ab_data.iterrows():
    gid = row['genome_id']
    if gid in genome_to_idx:
        indices.append(genome_to_idx[gid])
        labels.append(1 if row['phenotype'] == 'R' else 0)

X = Z[indices]
y = np.array(labels)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Test different missing fractions
results = []

for missing_frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    # Train model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Add noise to test set
    X_test_noisy = X_test.copy()
    n_features = X_test.shape[1]
    n_drop = int(n_features * missing_frac)

    for i in range(len(X_test_noisy)):
        drop_idx = np.random.choice(n_features, n_drop, replace=False)
        X_test_noisy[i, drop_idx] = 0

    # Predict
    y_pred = clf.predict_proba(X_test_noisy)[:, 1]
    auroc = roc_auc_score(y_test, y_pred)

    results.append({
        'missing_fraction': missing_frac,
        'auroc': auroc
    })

    print(f"Missing {missing_frac*100:.0f}%: AUROC = {auroc:.3f}")

# Save
results_df = pd.DataFrame(results)
results_df.to_csv("data/ml/robustness_results.csv", index=False)
print("\n✓ Saved: data/ml/robustness_results.csv")
