"""
Calibration analysis on ALL antibiotics combined
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import sys
sys.path.append('src/src_nn')
from diffusion_model import ConditionalDiffusionModel, DiffusionSchedule

def expected_calibration_error(y_true, y_pred_proba, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_pred_proba[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

print("="*80)
print("CALIBRATION ANALYSIS - ALL ANTIBIOTICS COMBINED")
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

# Get top 10 antibiotics
top_abs = pheno.groupby('antibiotic').size().sort_values(ascending=False).head(10).index

# Collect predictions from ALL antibiotics
all_y_true = []
all_y_pred = []

for antibiotic in top_abs:
    print(f"Processing {antibiotic}...")
    ab_data = pheno[pheno['antibiotic'] == antibiotic]
    
    # Sample 100 per antibiotic to keep it balanced
    if len(ab_data) > 100:
        ab_data = ab_data.sample(100, random_state=42)
    
    for _, row in ab_data.iterrows():
        gid = row['genome_id']
        if gid not in genome_to_idx:
            continue
        
        genome_idx = genome_to_idx[gid]
        genome_emb = torch.FloatTensor(Z[genome_idx])
        
        with torch.no_grad():
            samples = diffusion.sample(genome_emb, schedule, n_samples=100, device=device)
            mean_probs = samples.mean(dim=0).numpy()
        
        all_y_true.append(1 if row['phenotype'] == 'R' else 0)
        all_y_pred.append(mean_probs[2])  # P(Resistant)

y_true = np.array(all_y_true)
y_pred = np.array(all_y_pred)

# Compute metrics
ece = expected_calibration_error(y_true, y_pred)
brier = brier_score_loss(y_true, y_pred)

print("\n" + "="*80)
print("COMBINED CALIBRATION METRICS")
print("="*80)
print(f"Total samples: {len(y_true)}")
print(f"Expected Calibration Error (ECE): {ece:.4f}")
print(f"Brier Score: {brier:.4f}")

# Reliability diagram
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_true, y_pred, n_bins=10, strategy='uniform'
)

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
plt.plot(mean_predicted_value, fraction_of_positives, 's-', linewidth=2, 
         markersize=10, label=f'Pipeline B (ECE={ece:.3f})')
plt.xlabel('Mean Predicted Probability', fontsize=12)
plt.ylabel('Fraction of Positives', fontsize=12)
plt.title('Reliability Diagram (All Antibiotics)', fontsize=14, weight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)

plt.subplot(2, 2, 2)
plt.hist(y_pred[y_true == 0], bins=20, alpha=0.6, label='Susceptible', color='green')
plt.hist(y_pred[y_true == 1], bins=20, alpha=0.6, label='Resistant', color='red')
plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Prediction Distribution', fontsize=14, weight='bold')
plt.legend()

plt.subplot(2, 2, 3)
bin_edges = np.linspace(0, 1, 11)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
accuracies = []

for i in range(len(bin_edges) - 1):
    in_bin = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i+1])
    if in_bin.sum() > 0:
        accuracies.append(y_true[in_bin].mean())
    else:
        accuracies.append(np.nan)

plt.bar(bin_centers, accuracies, width=0.08, alpha=0.7)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlabel('Confidence Bin', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Confidence vs Accuracy', fontsize=14, weight='bold')
plt.grid(alpha=0.3)
plt.ylim([0, 1])

plt.subplot(2, 2, 4)
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, linewidth=2, label=f'AUROC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14, weight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/calibration_all_antibiotics.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: figures/calibration_all_antibiotics.png")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

if ece < 0.05:
    print("✓ EXCELLENT: Model is well-calibrated across antibiotics")
elif ece < 0.10:
    print("○ GOOD: Model is reasonably calibrated across antibiotics")
elif ece < 0.15:
    print("⚠ FAIR: Moderate miscalibration across antibiotics")
else:
    print("✗ POOR: Significant miscalibration across antibiotics")

# Save summary
summary = pd.DataFrame({
    'Metric': ['ECE', 'Brier Score', 'AUROC', 'N Samples', 'N Antibiotics'],
    'Value': [ece, brier, roc_auc, len(y_true), len(top_abs)]
})

summary.to_csv('results/calibration_all_antibiotics.csv', index=False)
print("\n✓ Saved: results/calibration_all_antibiotics.csv")

