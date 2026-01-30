"""
Train Masked Autoencoder
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import sys
sys.path.append('src/src_nn')
from masked_autoencoder import MaskedAutoencoder, MaskedAELoss

print("="*80)
print("TRAINING MASKED AUTOENCODER")
print("="*80)

# Load data
X = np.load("data/combined/X_k6_unique.npy")
print(f"Dataset: {X.shape}")

# Normalize
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X.astype(np.float32))

with open("models/scaler_masked.pkl", 'wb') as f:
    pickle.dump(scaler, f)

# Split
X_train, X_val = train_test_split(X_scaled, test_size=0.1, random_state=42)
print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# Dataloaders
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val)), batch_size=64, shuffle=False)

# Model
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

model = MaskedAutoencoder(input_dim=4096, latent_dim=512, mask_ratio=0.3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
criterion = MaskedAELoss()

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("\nTraining...\n")

# Train
for epoch in range(100):
    model.train()
    train_losses = []
    
    for (batch_x,) in train_loader:
        batch_x = batch_x.to(device)
        
        x_recon, z, mask = model(batch_x, return_mask=True)
        loss, masked_loss, _ = criterion(x_recon, batch_x, mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/100: Loss={np.mean(train_losses):.4f}")

# Save
torch.save(model.state_dict(), 'models/masked_ae_final.pt')
print("\n✓ Saved model")

# Extract embeddings
print("Extracting embeddings...")
model.eval()
Z = []

with torch.no_grad():
    for i in range(0, len(X_scaled), 64):
        batch = torch.FloatTensor(X_scaled[i:i+64]).to(device)
        z = model.encode(batch)
        Z.append(z.cpu().numpy())

Z = np.vstack(Z)
np.save("data/ml/Z_masked_ae.npy", Z)
print(f"✓ Saved embeddings: {Z.shape}")

