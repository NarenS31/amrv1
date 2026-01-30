"""
Train Conditional Diffusion Model (Pipeline B)
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import sys
sys.path.append('src/src_nn')
from diffusion_model import ConditionalDiffusionModel, DiffusionSchedule

print("="*80)
print("TRAINING PIPELINE B: CONDITIONAL DIFFUSION")
print("="*80)

Z = np.load("data/ml/Z_masked_ae.npy")
pheno = pd.read_csv("data/merged/phenotypes_final.csv")

with open("data/ml/genome_ids_all_complete.txt") as f:
    genome_ids = [line.strip().split('.')[0] for line in f]

genome_to_idx = {gid: i for i, gid in enumerate(genome_ids)}

top_ab = pheno.groupby('antibiotic').size().sort_values(ascending=False).index[0]
ab_data = pheno[pheno['antibiotic'] == top_ab]

indices = []
labels = []
for _, row in ab_data.iterrows():
    gid = row['genome_id']
    if gid in genome_to_idx:
        indices.append(genome_to_idx[gid])
        if row['phenotype'] == 'S':
            labels.append([1, 0, 0])
        elif row['phenotype'] == 'I':
            labels.append([0, 1, 0])
        else:
            labels.append([0, 0, 1])

X = Z[indices]
y = np.array(labels, dtype=np.float32)

print(f"\nAntibiotic: {top_ab}")
print(f"Samples: {len(X)}")
print(f"Label distribution: S={sum(y[:, 0]):.0f}, I={sum(y[:, 1]):.0f}, R={sum(y[:, 2]):.0f}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
    batch_size=32, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
    batch_size=32, shuffle=False
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

model = ConditionalDiffusionModel(genome_dim=512, phenotype_dim=3, hidden_dim=256).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

# Move schedule to device
schedule = DiffusionSchedule(n_steps=1000).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

best_loss = float('inf')

for epoch in range(100):
    model.train()
    train_losses = []
    
    for genome_emb, phenotype_clean in train_loader:
        genome_emb = genome_emb.to(device)
        phenotype_clean = phenotype_clean.to(device)
        
        t = torch.randint(0, schedule.n_steps, (genome_emb.size(0),), device=device)
        phenotype_noisy, noise = schedule.add_noise(phenotype_clean, t)
        noise_pred = model(phenotype_noisy, genome_emb, t)
        
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for genome_emb, phenotype_clean in val_loader:
                genome_emb = genome_emb.to(device)
                phenotype_clean = phenotype_clean.to(device)
                
                t = torch.randint(0, schedule.n_steps, (genome_emb.size(0),), device=device)
                phenotype_noisy, noise = schedule.add_noise(phenotype_clean, t)
                noise_pred = model(phenotype_noisy, genome_emb, t)
                
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        print(f"Epoch {epoch+1}/100: Train={train_loss:.4f}, Val={val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'models/diffusion_best.pt')

torch.save(model.state_dict(), 'models/diffusion_final.pt')
print("\n✓ Saved: models/diffusion_final.pt")

