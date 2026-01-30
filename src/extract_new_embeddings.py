"""
Extract embeddings for new genomes using pre-trained autoencoder
"""
import numpy as np
import torch
import torch.nn as nn
import pickle

print("="*80)
print("EXTRACTING EMBEDDINGS FOR NEW GENOMES")
print("="*80)

# Load scaler
with open("models/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# Load unique k-mer matrix
X_all = np.load("data/combined/X_k6_unique.npy")
print(f"Total genomes: {X_all.shape}")

# Load genome IDs
with open("data/combined/genome_ids_unique.txt") as f:
    all_ids = [line.strip() for line in f]

# Find which are new (not in original labeled set)
with open("data/ml/labeled_genome_ids.txt") as f:
    labeled_ids = set(line.strip().split('.')[0] for line in f)

# Normalize all IDs
all_ids_norm = [gid.split('.')[0] for gid in all_ids]

# Find new genomes
new_indices = [i for i, gid in enumerate(all_ids_norm) if gid not in labeled_ids]

print(f"New genomes to process: {len(new_indices):,}")

# Scale
X_all_scaled = scaler.transform(X_all.astype(np.float32))

# Load autoencoder
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

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = Autoencoder().to(device)
model.load_state_dict(torch.load("models/pretrained_ae_20k.pt", map_location=device))
model.eval()

print(f"Model loaded on {device}")

# Extract ALL embeddings
print("\nExtracting embeddings for all genomes...")

with torch.no_grad():
    _, Z_all = model(torch.FloatTensor(X_all_scaled).to(device))
    Z_all = Z_all.cpu().numpy()

print(f"All embeddings: {Z_all.shape}")

# Save complete embeddings + mapping
np.save("data/ml/Z_all_complete.npy", Z_all)

with open("data/ml/genome_ids_all_complete.txt", 'w') as f:
    for gid in all_ids:
        f.write(gid + '\n')

print(f"\n✓ Saved: data/ml/Z_all_complete.npy ({Z_all.shape})")
print(f"✓ Saved: data/ml/genome_ids_all_complete.txt ({len(all_ids):,} IDs)")

print("\n" + "="*80)

