"""
Test if diffusion uncertainty enables better decisions
Defer uncertain cases to AST, keep confident predictions
"""
import numpy as np
import pandas as pd
import torch
import sys
sys.path.append('src/src_nn')
from diffusion_model import ConditionalDiffusionModel, DiffusionSchedule
from sklearn.metrics import accuracy_score, roc_auc_score

print("="*80)
print("DIFFUSION UTILITY ANALYSIS")
print("="*80)

# Load models
device = "cpu"
diffusion = ConditionalDiffusionModel().to(device)
diffusion.load_state_dict(torch.load('models/diffusion_final.pt', map_location=device))
diffusion.eval()
schedule = DiffusionSchedule(n_steps=1000).to(device)

# Load data
Z = np.load('data/ml/Z_masked_ae.npy')
pheno = pd.read_csv('data/merged/phenotypes_final.csv')

with open('data/ml/genome_ids_all_complete.txt') as f:
    genome_ids = [line.strip().split('.')[0] for line in f]

genome_to_idx = {gid: i for i, gid in enumerate(genome_ids)}

# Get test data
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
y_true = np.array(labels)

print(f"Testing on {top_ab}: {len(X)} samples")

# Generate predictions with uncertainty
predictions = []
uncertainties = []

print("Generating predictions...")
for i in range(len(X)):
    genome_emb = torch.FloatTensor(X[i])
    
    with torch.no_grad():
        samples = diffusion.sample(genome_emb, schedule, n_samples=100, device=device)
        mean_probs = samples.mean(dim=0).numpy()
    
    # Prediction
    pred = 1 if mean_probs[2] > 0.5 else 0
    predictions.append(pred)
    
    # Uncertainty (entropy)
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
    uncertainties.append(entropy)

predictions = np.array(predictions)
uncertainties = np.array(uncertainties)

# Test selective prediction
print("\n" + "="*80)
print("SELECTIVE PREDICTION ANALYSIS")
print("="*80)

# Baseline accuracy (use all predictions)
baseline_acc = accuracy_score(y_true, predictions)
print(f"\nBaseline (no deferral): {baseline_acc:.3f} accuracy")

# Test different deferral thresholds
defer_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

print(f"\n{'Defer %':<10} {'Coverage':<12} {'Accuracy':<12} {'Improvement'}")
print("-"*60)

for thresh_percentile in defer_thresholds:
    # Defer top X% most uncertain
    defer_threshold = np.percentile(uncertainties, thresh_percentile * 100)
    
    # Keep only confident predictions
    confident_mask = uncertainties <= defer_threshold
    
    if confident_mask.sum() > 0:
        coverage = confident_mask.mean()
        confident_acc = accuracy_score(y_true[confident_mask], predictions[confident_mask])
        improvement = confident_acc - baseline_acc
        
        print(f"{(1-thresh_percentile)*100:<10.0f}% {coverage:<12.1%} {confident_acc:<12.3f} {improvement:+.3f}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

print("If accuracy improves when deferring uncertain cases:")
print("  → Uncertainty is USEFUL for decision-making")
print("  → Justifies Pipeline B contribution")
print("\nIf accuracy doesn't improve:")
print("  → Uncertainty is NOT calibrated to correctness")
print("  → Pipeline B needs improvement")
print("="*80)

