"""
Fair comparison: AMRFinder vs Pipeline A on SAME test set
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
import sys
import os
import glob

# Add src to path
sys.path.append('src')
from amrfinder_drug_rules import AMRFinderPredictor

print("="*80)
print("FAIR COMPARISON: AMRFinder vs Pipeline A")
print("="*80)

# Rest of code stays the same...
# Load data
Z_masked = np.load("data/ml/Z_masked_ae.npy")
pheno = pd.read_csv("data/merged/phenotypes_final.csv")

with open("data/ml/genome_ids_all_complete.txt") as f:
    genome_ids = [line.strip().split('.')[0] for line in f]

genome_to_idx = {gid: i for i, gid in enumerate(genome_ids)}

# Get genomes that have BOTH phenotype data AND AMRFinder output
amr_files = glob.glob("results/amrfinder_raw/*.tsv")
amr_genome_ids = set()

for f in amr_files:
    basename = os.path.basename(f).replace('.tsv', '')
    gcf_id = basename.split('.')[0]
    amr_genome_ids.add(gcf_id)

print(f"Genomes with AMRFinder output: {len(amr_genome_ids)}")

# Filter phenotypes to genomes with AMRFinder
pheno_with_amr = pheno[pheno['genome_id'].isin(amr_genome_ids)].copy()

print(f"Phenotype records with AMRFinder: {len(pheno_with_amr)}")

if len(pheno_with_amr) == 0:
    print("\n" + "="*80)
    print("ERROR: No overlap between AMRFinder genomes and phenotype data")
    print("="*80)
    exit(1)

# Split into train/test (80/20)
unique_genomes = pheno_with_amr['genome_id'].unique()
train_genomes, test_genomes = train_test_split(
    unique_genomes, test_size=0.2, random_state=42
)

train_set = set(train_genomes)
test_set = set(test_genomes)

print(f"Train genomes: {len(train_set)}")
print(f"Test genomes: {len(test_set)}")

pheno_train = pheno_with_amr[pheno_with_amr['genome_id'].isin(train_set)]
pheno_test = pheno_with_amr[pheno_with_amr['genome_id'].isin(test_set)]

print(f"Train phenotypes: {len(pheno_train)}")
print(f"Test phenotypes: {len(pheno_test)}")

# Get antibiotics with enough test samples
test_ab_counts = pheno_test.groupby('antibiotic').size()
test_antibiotics = test_ab_counts[test_ab_counts >= 30].index

print(f"\nAntibiotics with ≥30 test samples: {len(test_antibiotics)}")

# Initialize AMRFinder predictor
amr_predictor = AMRFinderPredictor()

results = []

for antibiotic in test_antibiotics:
    print(f"\nTesting {antibiotic}...")
    
    # Get train data for ML
    ab_train = pheno_train[pheno_train['antibiotic'] == antibiotic]
    train_indices = []
    train_labels = []
    
    for _, row in ab_train.iterrows():
        gid = row['genome_id']
        if gid in genome_to_idx:
            train_indices.append(genome_to_idx[gid])
            train_labels.append(1 if row['phenotype'] == 'R' else 0)
    
    if len(set(train_labels)) < 2 or len(train_indices) < 10:
        continue
    
    # Train ML model
    X_train = Z_masked[train_indices]
    y_train = np.array(train_labels)
    
    clf = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Get test data
    ab_test = pheno_test[pheno_test['antibiotic'] == antibiotic]
    
    ml_preds = []
    amr_preds = []
    true_labels = []
    
    for _, row in ab_test.iterrows():
        gid = row['genome_id']
        true_pheno = row['phenotype']
        
        if gid not in genome_to_idx:
            continue
        
        # ML prediction
        genome_idx = genome_to_idx[gid]
        X_test = Z_masked[genome_idx].reshape(1, -1)
        ml_pred = clf.predict(X_test)[0]
        ml_preds.append(ml_pred)
        
        # AMRFinder prediction
        amr_file = None
        exact = f"results/amrfinder_raw/{gid}.tsv"
        if os.path.exists(exact):
            amr_file = exact
        else:
            matches = glob.glob(f"results/amrfinder_raw/{gid}.*.tsv")
            if matches:
                amr_file = matches[0]
        
        if amr_file:
            try:
                amr_df = pd.read_csv(amr_file, sep='\t')
                amr_pred = amr_predictor.predict_genome(amr_df, antibiotic)
                amr_preds.append(1 if amr_pred == 'R' else 0)
            except:
                amr_preds.append(0)
        else:
            amr_preds.append(0)
        
        true_labels.append(1 if true_pheno == 'R' else 0)
    
    if len(true_labels) < 10:
        continue
    
    ml_preds = np.array(ml_preds)
    amr_preds = np.array(amr_preds)
    true_labels = np.array(true_labels)
    
    # Compute metrics
    results.append({
        'antibiotic': antibiotic,
        'n_train': len(train_labels),
        'n_test': len(true_labels),
        'ml_accuracy': accuracy_score(true_labels, ml_preds),
        'ml_recall': recall_score(true_labels, ml_preds, zero_division=0),
        'ml_precision': precision_score(true_labels, ml_preds, zero_division=0),
        'amr_accuracy': accuracy_score(true_labels, amr_preds),
        'amr_recall': recall_score(true_labels, amr_preds, zero_division=0),
        'amr_precision': precision_score(true_labels, amr_preds, zero_division=0),
    })

df = pd.DataFrame(results)

if len(df) == 0:
    print("\n" + "="*80)
    print("ERROR: No antibiotics passed filters")
    print("="*80)
    print("This means there's not enough overlap between:")
    print("- The 200 genomes you ran AMRFinder on")
    print("- The 9,110 genomes with phenotype labels")
    print("\nYou need to run AMRFinder on MORE genomes from your phenotype set")
    print("="*80)
    exit(1)

print("\n" + "="*80)
print("FAIR COMPARISON ON SAME TEST SET")
print("="*80)

print(f"\n{'Antibiotic':<25} {'N_test':<8} {'ML Recall':<12} {'AMR Recall':<12} {'Winner'}")
print("-"*80)

for _, row in df.iterrows():
    winner = "ML" if row['ml_recall'] > row['amr_recall'] else ("AMR" if row['amr_recall'] > row['ml_recall'] else "TIE")
    print(f"{row['antibiotic']:<25} {row['n_test']:<8} "
          f"{row['ml_recall']:<12.3f} {row['amr_recall']:<12.3f} {winner}")

print("\n" + "="*80)
print("OVERALL AVERAGES")
print("="*80)

print(f"Pipeline A (ML) Recall:  {df['ml_recall'].mean():.3f}")
print(f"AMRFinder Recall:        {df['amr_recall'].mean():.3f}")
print(f"Pipeline A Accuracy:     {df['ml_accuracy'].mean():.3f}")
print(f"AMRFinder Accuracy:      {df['amr_accuracy'].mean():.3f}")

print(f"\nML wins on: {(df['ml_recall'] > df['amr_recall']).sum()}/{len(df)} antibiotics")
print(f"AMR wins on: {(df['amr_recall'] > df['ml_recall']).sum()}/{len(df)} antibiotics")

# Statistical test
from scipy import stats
t_stat, p_value = stats.ttest_rel(df['ml_recall'], df['amr_recall'])

print(f"\nPaired t-test: t={t_stat:.3f}, p={p_value:.4f}")

if p_value < 0.05:
    if df['ml_recall'].mean() > df['amr_recall'].mean():
        print("✓ Pipeline A significantly outperforms AMRFinder")
    else:
        print("✗ AMRFinder significantly outperforms Pipeline A")
else:
    print("○ No significant difference between methods")

df.to_csv("results/fair_comparison_same_testset.csv", index=False)
print("\n✓ Saved: results/fair_comparison_same_testset.csv")

