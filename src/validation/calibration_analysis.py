"""
Calibration analysis for Pipeline B (Diffusion)
Reliability diagrams, ECE, Brier score
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
    """
    Compute Expected Calibration Error (ECE)
    """
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

def analyze_calibration():
    """
    Full calibration analysis for diffusion model
    """
    print("="*80)
    print("CALIBRATION ANALYSIS - PIPELINE B")
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
    print("Sampling predictions...")
    
    # Generate predictions
    y_pred_proba = []
    
    for i in range(len(X)):
        genome_emb = torch.FloatTensor(X[i])
        
        with torch.no_grad():
            samples = diffusion.sample(genome_emb, schedule, n_samples=100, device=device)
            mean_probs = samples.mean(dim=0).numpy()
        
        # Probability of resistant
        y_pred_proba.append(mean_probs[2])
    
    y_pred_proba = np.array(y_pred_proba)
    
    # Compute metrics
    ece = expected_calibration_error(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)
    
    print("\n" + "="*80)
    print("CALIBRATION METRICS")
    print("="*80)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Perfect calibration: ECE = 0")
    print(f"Perfect predictions: Brier = 0")
    
    # Reliability diagram
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=10, strategy='uniform'
    )
    
    plt.figure(figsize=(10, 10))
    
    # Plot reliability curve
    plt.subplot(2, 2, 1)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'Pipeline B (ECE={ece:.3f})')
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Reliability Diagram', fontsize=14, weight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot histogram of predictions
    plt.subplot(2, 2, 2)
    plt.hist(y_pred_proba[y_true == 0], bins=20, alpha=0.5, label='Susceptible', color='green')
    plt.hist(y_pred_proba[y_true == 1], bins=20, alpha=0.5, label='Resistant', color='red')
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Prediction Distribution', fontsize=14, weight='bold')
    plt.legend()
    
    # Plot confidence vs accuracy by bin
    plt.subplot(2, 2, 3)
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    accuracies = []
    
    for i in range(len(bin_edges) - 1):
        in_bin = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i+1])
        if in_bin.sum() > 0:
            accuracies.append(y_true[in_bin].mean())
        else:
            accuracies.append(np.nan)
    
    plt.bar(bin_centers, accuracies, width=0.08, alpha=0.7, label='Accuracy in bin')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
    plt.xlabel('Confidence Bin', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Confidence vs Accuracy', fontsize=14, weight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # ECE visualization
    plt.subplot(2, 2, 4)
    bin_sizes = []
    for i in range(len(bin_edges) - 1):
        in_bin = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i+1])
        bin_sizes.append(in_bin.sum())
    
    plt.bar(bin_centers, bin_sizes, width=0.08, alpha=0.7, color='blue')
    plt.xlabel('Confidence Bin', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Sample Distribution by Confidence', fontsize=14, weight='bold')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/calibration_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: figures/calibration_analysis.png")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    if ece < 0.05:
        print("✓ EXCELLENT: Model is well-calibrated (ECE < 0.05)")
    elif ece < 0.10:
        print("○ GOOD: Model is reasonably calibrated (ECE < 0.10)")
    elif ece < 0.15:
        print("⚠ FAIR: Model shows moderate miscalibration (ECE < 0.15)")
    else:
        print("✗ POOR: Model is poorly calibrated (ECE ≥ 0.15)")
        print("  → Uncertainty estimates are unreliable")
        print("  → Consider temperature scaling or Platt scaling")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': ['ECE', 'Brier Score', 'N Samples'],
        'Value': [ece, brier, len(y_true)]
    })
    
    metrics_df.to_csv('results/calibration_metrics.csv', index=False)
    print("\n✓ Saved: results/calibration_metrics.csv")
    
    return ece, brier

if __name__ == '__main__':
    analyze_calibration()

