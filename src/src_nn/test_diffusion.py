"""
Test Pipeline B: Sample probability distributions
"""
import numpy as np
import pandas as pd
import torch
import sys
sys.path.append('src/src_nn')
from diffusion_model import ConditionalDiffusionModel, DiffusionSchedule
from sklearn.metrics import roc_auc_score, accuracy_score

print("="*80)
print("TESTING PIPELINE B: PROBABILISTIC PREDICTIONS")
print("="*80)

# Load data
Z = np.load("data/ml/Z_masked_ae.npy")
pheno = pd.read_csv("data/merged/phenotypes_final.csv")

with open("data/ml/genome_ids_all_complete.txt") as f:
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
        labels.append(row['phenotype'])

X = Z[indices]
y = np.array(labels)

# Load model
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = ConditionalDiffusionModel().to(device)
model.load_state_dict(torch.load('models/diffusion_final.pt', map_location=device))
model.eval()

schedule = DiffusionSchedule(n_steps=1000).to(device)

print(f"\nAntibiotic: {top_ab}")
print(f"Test samples: {len(X)}")
print("\nSampling probability distributions (100 samples per genome)...\n")

# Sample for first 10 genomes
predictions = []

for i in range(min(10, len(X))):
    genome_emb = torch.FloatTensor(X[i]).to(device)
    
    # Sample 100 possible outcomes
    samples = model.sample(genome_emb, schedule, n_samples=100, device=device)
    
    # Average to get probability distribution
    mean_probs = samples.mean(dim=0).cpu().numpy()
    std_probs = samples.std(dim=0).cpu().numpy()
    
    predictions.append({
        'true_label': y[i],
        'prob_S': mean_probs[0],
        'prob_I': mean_probs[1],
        'prob_R': mean_probs[2],
        'std_S': std_probs[0],
        'std_I': std_probs[1],
        'std_R': std_probs[2]
    })

# Display results
print("="*80)
print("EXAMPLE PREDICTIONS (first 10 genomes)")
print("="*80)
print(f"{'True':<6} {'P(S)':<10} {'P(I)':<10} {'P(R)':<10} {'Prediction':<12} {'Uncertainty'}")
print("-"*80)

for pred in predictions:
    true = pred['true_label']
    p_s = pred['prob_S']
    p_i = pred['prob_I']
    p_r = pred['prob_R']
    
    # Most likely prediction
    if p_s > p_i and p_s > p_r:
        prediction = "S"
    elif p_r > p_i and p_r > p_s:
        prediction = "R"
    else:
        prediction = "I"
    
    # Overall uncertainty (entropy)
    probs = np.array([p_s, p_i, p_r])
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    correct = "✓" if prediction == true else "✗"
    
    print(f"{true:<6} {p_s:<10.3f} {p_i:<10.3f} {p_r:<10.3f} {prediction:<4} {correct:<8} {entropy:.3f}")

print("\n" + "="*80)
print("WHAT PIPELINE B PROVIDES:")
print("="*80)
print("✓ Full probability distribution (not just a single prediction)")
print("✓ Uncertainty quantification (entropy, std dev)")
print("✓ Multiple possible outcomes sampled")
print("✓ Clinically actionable: 'This genome is 70% likely R, 25% likely I'")
print("="*80)

# Quick AUROC test
print("\nComputing AUROC on full test set...")

all_probs_r = []
all_labels_binary = []

for i in range(len(X)):
    genome_emb = torch.FloatTensor(X[i]).to(device)
    samples = model.sample(genome_emb, schedule, n_samples=100, device=device)
    mean_probs = samples.mean(dim=0).cpu().numpy()
    
    all_probs_r.append(mean_probs[2])  # P(Resistant)
    all_labels_binary.append(1 if y[i] == 'R' else 0)

auroc = roc_auc_score(all_labels_binary, all_probs_r)
print(f"\nPipeline B AUROC: {auroc:.3f}")

print("\n" + "="*80)

