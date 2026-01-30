
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle

print("="*80)
print("EXTRACTING EMBEDDINGS (FIXED ID MATCHING)")
print("="*80)

# Load scaler and data
with open("models/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

X = np.load("data/combined/X_k6_all.npy")
X_scaled = scaler.transform(X.astype(np.float32))

print(f"Combined data: {X_scaled.shape}")

# Load model


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
model.load_state_dict(torch.load(
    "models/pretrained_ae_20k.pt", map_location=device))
model.eval()

print("✓ Model loaded")

# Load phenotype genome IDs
pheno = pd.read_csv("data/merged/all_phenotypes.csv")
pheno_ids = set(pheno['genome_id'].unique())
print(f"Phenotype genomes: {len(pheno_ids)}")

# Load combined genome IDs
with open("data/combined/genome_ids_all.txt") as f:
    combined_ids = [line.strip() for line in f]

print(f"Combined genomes: {len(combined_ids)}")

# Create normalized mapping: GCF_XXXXXXX.X -> GCF_XXXXXXX


def normalize_id(gid):
    return gid.split('.')[0]


# Build mapping: normalized_id -> list of indices in combined matrix
normalized_to_indices = {}
for i, gid in enumerate(combined_ids):
    norm_id = normalize_id(gid)
    if norm_id not in normalized_to_indices:
        normalized_to_indices[norm_id] = []
    normalized_to_indices[norm_id].append(i)

# Find indices for labeled genomes
labeled_indices = []
labeled_genome_ids = []

for pheno_id in pheno_ids:
    norm_id = normalize_id(pheno_id)
    if norm_id in normalized_to_indices:
        # Use first matching index
        idx = normalized_to_indices[norm_id][0]
        labeled_indices.append(idx)
        # Use the full ID with version
        labeled_genome_ids.append(combined_ids[idx])

print(f"\n✓ Found {len(labeled_indices)} labeled genomes!")

# Extract embeddings
X_labeled = X_scaled[labeled_indices]

print("Extracting embeddings...")
with torch.no_grad():
    _, Z_labeled = model(torch.FloatTensor(X_labeled).to(device))
    Z_labeled = Z_labeled.cpu().numpy()

# Save
np.save("data/ml/Z_pretrained_20k.npy", Z_labeled)
np.save("data/ml/labeled_indices.npy", np.array(labeled_indices))

with open("data/ml/labeled_genome_ids.txt", 'w') as f:
    for gid in labeled_genome_ids:
        f.write(gid + '\n')

print(f"\n✓ Saved embeddings: {Z_labeled.shape}")
print(f"✓ Saved {len(labeled_indices)} labeled indices")

print("\n" + "="*80)
print("EMBEDDING EXTRACTION COMPLETE!")
print("="*80)
