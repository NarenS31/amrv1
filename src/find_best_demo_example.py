"""
Find a genome + antibiotic combo with high confidence for demo
"""
import numpy as np
import pandas as pd
import torch
import sys
sys.path.append('src/src_nn')
from diffusion_model import ConditionalDiffusionModel, DiffusionSchedule

print("Finding best demo example...")

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

# Test some examples
test_antibiotics = ['gentamicin', 'ciprofloxacin', 'tetracycline', 'cefepime', 'imipenem']
results = []

print("\nTesting examples...")

for ab in test_antibiotics:
    ab_data = pheno[pheno['antibiotic'] == ab]
    
    # Get a few random examples
    samples = ab_data.sample(min(20, len(ab_data)))
    
    for _, row in samples.iterrows():
        gid = row['genome_id']
        if gid not in genome_to_idx:
            continue
        
        genome_idx = genome_to_idx[gid]
        genome_emb = torch.FloatTensor(Z[genome_idx])
        
        # Sample predictions
        with torch.no_grad():
            pred_samples = diffusion.sample(genome_emb, schedule, n_samples=100, device=device)
            mean_probs = pred_samples.mean(dim=0).numpy()
        
        # Calculate confidence (max probability)
        max_prob = mean_probs.max()
        max_class = ['S', 'I', 'R'][mean_probs.argmax()]
        
        # Calculate entropy (lower = more confident)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        
        results.append({
            'genome_id': gid,
            'antibiotic': ab,
            'true_label': row['phenotype'],
            'predicted': max_class,
            'confidence': max_prob,
            'entropy': entropy,
            'prob_S': mean_probs[0],
            'prob_I': mean_probs[1],
            'prob_R': mean_probs[2]
        })

df = pd.DataFrame(results)

# Find best examples (high confidence, correct prediction)
df['correct'] = df['true_label'] == df['predicted']
df_correct = df[df['correct']].sort_values('confidence', ascending=False)

print("\n" + "="*80)
print("TOP 10 HIGH-CONFIDENCE EXAMPLES (for demo)")
print("="*80)
print(f"\n{'Genome':<20} {'Antibiotic':<15} {'True':<5} {'Pred':<5} {'Confidence':<12} {'Entropy':<10}")
print("-"*80)

for _, row in df_correct.head(10).iterrows():
    print(f"{row['genome_id']:<20} {row['antibiotic']:<15} {row['true_label']:<5} "
          f"{row['predicted']:<5} {row['confidence']:.3f} ({row['confidence']*100:.1f}%)  {row['entropy']:.3f}")

print("\n" + "="*80)
print("RECOMMENDED DEMO COMBO:")
best = df_correct.iloc[0]
print(f"Genome: {best['genome_id']}")
print(f"Antibiotic: {best['antibiotic']}")
print(f"Prediction: {best['predicted']} ({best['confidence']*100:.1f}% confident)")
print(f"Distribution: S={best['prob_S']:.3f}, I={best['prob_I']:.3f}, R={best['prob_R']:.3f}")
print("="*80)

