"""
Rebuild AMRFinder benchmark with CORRECT drug-specific rules
"""
import pandas as pd
import os
import glob
import sys
sys.path.append('src')
from amrfinder_drug_rules import AMRFinderPredictor

print("="*80)
print("REBUILDING AMRFINDER BENCHMARK (FIXED LOGIC)")
print("="*80)

# Load lab phenotypes
lab = pd.read_csv("results/lab_phenotypes_core.csv")
print(f"Lab phenotypes: {len(lab)}")

# Initialize predictor
predictor = AMRFinderPredictor()

# Load AMRFinder outputs
amrfinder_files = glob.glob("results/amrfinder_raw/*.tsv")
print(f"AMRFinder files: {len(amrfinder_files)}")

# Build predictions
results = []

for idx, row in lab.iterrows():
    gcf = row['genome_id']
    antibiotic = row['antibiotic']
    true_pheno = row['phenotype']
    
    # Find AMRFinder file (handle version mismatch)
    amr_file = None
    exact = f"results/amrfinder_raw/{gcf}.tsv"
    if os.path.exists(exact):
        amr_file = exact
    else:
        # Try with version
        matches = glob.glob(f"results/amrfinder_raw/{gcf}.*.tsv")
        if matches:
            amr_file = matches[0]
    
    if not amr_file:
        continue
    
    # Load AMRFinder output
    try:
        amr_df = pd.read_csv(amr_file, sep='\t')
    except:
        continue
    
    # Predict
    pred_pheno = predictor.predict_genome(amr_df, antibiotic)
    
    results.append({
        'genome_id': gcf,
        'antibiotic': antibiotic,
        'true_phenotype': true_pheno,
        'predicted_phenotype': pred_pheno
    })
    
    if (idx + 1) % 1000 == 0:
        print(f"  Processed {idx + 1}/{len(lab)}...")

# Create benchmark dataframe
bench = pd.DataFrame(results)
bench.to_csv("results/benchmark_fixed.csv", index=False)

print(f"\n✓ Predictions made: {len(bench)}")

# Compute metrics
def compute_metrics(df):
    """Compute TP/FP/TN/FN"""
    tp = len(df[(df['true_phenotype'] == 'R') & (df['predicted_phenotype'] == 'R')])
    fp = len(df[(df['true_phenotype'] != 'R') & (df['predicted_phenotype'] == 'R')])
    tn = len(df[(df['true_phenotype'] != 'R') & (df['predicted_phenotype'] != 'R')])
    fn = len(df[(df['true_phenotype'] == 'R') & (df['predicted_phenotype'] != 'R')])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / len(df) if len(df) > 0 else 0
    
    return {
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'precision': precision, 'recall': recall, 'accuracy': accuracy
    }

# Overall metrics
overall = compute_metrics(bench)

print("\n" + "="*80)
print("OVERALL PERFORMANCE (FIXED)")
print("="*80)
print(f"TP: {overall['TP']}")
print(f"FP: {overall['FP']}")
print(f"TN: {overall['TN']}")
print(f"FN: {overall['FN']}")
print(f"\nAccuracy:  {overall['accuracy']:.3f}")
print(f"Precision: {overall['precision']:.3f}")
print(f"Recall:    {overall['recall']:.3f}")

# Per-drug breakdown
print("\n" + "="*80)
print("PER-DRUG BREAKDOWN")
print("="*80)

for drug in sorted(bench['antibiotic'].unique()):
    drug_df = bench[bench['antibiotic'] == drug]
    metrics = compute_metrics(drug_df)
    
    print(f"\n{drug}:")
    print(f"  TP={metrics['TP']:4d}  FP={metrics['FP']:4d}  TN={metrics['TN']:4d}  FN={metrics['FN']:4d}")
    print(f"  Recall={metrics['recall']:.3f}  Precision={metrics['precision']:.3f}")

print("\n" + "="*80)
print("COMPARISON TO BEFORE:")
print("="*80)
print("OLD recall: 0.0084 (basically zero)")
print(f"NEW recall: {overall['recall']:.3f}")
improvement = overall['recall'] / 0.0084 if overall['recall'] > 0 else 0
print(f"Improvement: {improvement:.1f}x better!")
print("="*80)

