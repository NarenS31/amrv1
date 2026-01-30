
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import os

print("="*80)
print("TRAINING CLASSIFIERS ON PRE-TRAINED EMBEDDINGS")
print("="*80)

# Load embeddings and phenotypes
Z = np.load("data/ml/Z_pretrained_20k.npy")
pheno = pd.read_csv("data/merged/all_phenotypes.csv")

print(f"Embeddings: {Z.shape}")
print(f"Phenotypes: {len(pheno)} observations")

# Load genome ID mapping
with open("data/ml/labeled_genome_ids.txt") as f:
    labeled_genome_ids = [line.strip() for line in f]

# Normalize IDs for matching


def normalize_id(gid):
    return gid.split('.')[0]


genome_to_idx = {normalize_id(gid): i for i,
                 gid in enumerate(labeled_genome_ids)}

print(f"Labeled genomes: {len(labeled_genome_ids)}")

# Get top 20 antibiotics
top_antibiotics = pheno.groupby('antibiotic').size(
).sort_values(ascending=False).head(20).index

print(f"\nTraining on top {len(top_antibiotics)} antibiotics...")

results = []

for antibiotic in top_antibiotics:
    ab_data = pheno[pheno['antibiotic'] == antibiotic]

    # Create dataset
    indices = []
    labels = []

    for _, row in ab_data.iterrows():
        gid_norm = normalize_id(row['genome_id'])
        if gid_norm in genome_to_idx:
            indices.append(genome_to_idx[gid_norm])
            labels.append(1 if row['phenotype'] == 'R' else 0)

    # Need both classes and enough samples
    if len(set(labels)) < 2 or len(indices) < 30:
        print(f"{antibiotic:30s}: SKIPPED (insufficient data)")
        continue

    X_ab = Z[indices]
    y_ab = np.array(labels)

    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aurocs = []
    accs = []
    sens = []
    specs = []

    for train_idx, test_idx in cv.split(X_ab, y_ab):
        X_train, X_test = X_ab[train_idx], X_ab[test_idx]
        y_train, y_test = y_ab[train_idx], y_ab[test_idx]

        clf = LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train)

        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        aurocs.append(roc_auc_score(y_test, y_pred_proba))
        accs.append(accuracy_score(y_test, y_pred))
        sens.append(recall_score(y_test, y_pred, zero_division=0))

        # Specificity = TN / (TN + FP)
        tn = ((y_test == 0) & (y_pred == 0)).sum()
        fp = ((y_test == 0) & (y_pred == 1)).sum()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specs.append(spec)

    results.append({
        'antibiotic': antibiotic,
        'n_samples': len(indices),
        'n_resistant': sum(labels),
        'n_susceptible': len(labels) - sum(labels),
        'auroc_mean': np.mean(aurocs),
        'auroc_std': np.std(aurocs),
        'accuracy': np.mean(accs),
        'sensitivity': np.mean(sens),
        'specificity': np.mean(specs)
    })

    print(f"{antibiotic:30s}: AUROC={np.mean(aurocs):.3f}±{np.std(aurocs):.3f}  n={len(indices)}")

# Save
results_df = pd.DataFrame(results)
os.makedirs("data/ml", exist_ok=True)
results_df.to_csv("data/ml/pretrained_results.csv", index=False)

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"✓ Trained on {len(results)} antibiotics")
print(f"✓ Mean AUROC: {results_df['auroc_mean'].mean():.3f}")
print(
    f"✓ Best AUROC: {results_df['auroc_mean'].max():.3f} ({results_df.loc[results_df['auroc_mean'].idxmax(), 'antibiotic']})")
print(f"✓ Saved: data/ml/pretrained_results.csv")
print("="*80)
