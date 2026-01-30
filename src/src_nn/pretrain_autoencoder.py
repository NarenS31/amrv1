import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os
import pickle

print("="*80)
print("PRE-TRAINING AUTOENCODER ON 20K+ GENOMES")
print("="*80)

# Load all genomes
X = np.load("data/combined/X_k6_all.npy")
print(f"\nDataset: {X.shape}")

# Normalize
print("Normalizing...")
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X.astype(np.float32))

# Save scaler
os.makedirs("models", exist_ok=True)
with open("models/scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved scaler")

# Autoencoder


class Autoencoder(nn.Module):
    def __init__(self, input_dim=4096, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


# Dataset
dataset = TensorDataset(torch.FloatTensor(X_scaled))
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Train
device = "mps" if torch.backends.mps.is_available(
) else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = Autoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining...")
for epoch in range(50):
    total_loss = 0
    for (X_batch,) in loader:
        X_batch = X_batch.to(device)

        X_recon, z = model(X_batch)
        loss = nn.functional.mse_loss(X_recon, X_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/50: Loss = {total_loss/len(loader):.4f}")

# Save model
torch.save(model.state_dict(), "models/pretrained_ae_20k.pt")
print("\n✓ Saved: models/pretrained_ae_20k.pt")

# Extract embeddings for labeled genomes
print("\nExtracting embeddings for labeled genomes...")

pheno = pd.read_csv("data/merged/all_phenotypes.csv")
labeled_genomes = set(pheno['genome_id'].unique())

# Load genome ID mapping
with open("data/combined/genome_ids_all.txt") as f:
    all_genome_ids = [line.strip() for line in f]

# Find indices
labeled_indices = [i for i, gid in enumerate(
    all_genome_ids) if gid in labeled_genomes]

print(f"Found {len(labeled_indices)} labeled genomes in combined matrix")

# Extract embeddings
X_labeled = X_scaled[labeled_indices]
model.eval()
with torch.no_grad():
    _, Z_labeled = model(torch.FloatTensor(X_labeled).to(device))
    Z_labeled = Z_labeled.cpu().numpy()

# Save
os.makedirs("data/ml", exist_ok=True)
np.save("data/ml/Z_pretrained_20k.npy", Z_labeled)
np.save("data/ml/labeled_indices.npy", np.array(labeled_indices))

# Also save the mapping
labeled_genome_ids = [all_genome_ids[i] for i in labeled_indices]
with open("data/ml/labeled_genome_ids.txt", 'w') as f:
    for gid in labeled_genome_ids:
        f.write(gid + '\n')

print(f"✓ Saved embeddings: {Z_labeled.shape}")
print(f"✓ Saved {len(labeled_indices)} labeled indices")
print(f"✓ Saved labeled genome IDs mapping")

print("\n" + "="*80)
print("PRE-TRAINING COMPLETE!")
print("="*80)
print(f"\nSummary:")
print(f"  Total genomes trained: {X.shape[0]:,}")
print(f"  Labeled genomes: {len(labeled_indices):,}")
print(f"  Embedding dimension: {Z_labeled.shape[1]}")
print(f"  Model saved: models/pretrained_ae_20k.pt")
print("="*80)
