""""
Masked Autoencoder for Genomic AMR Prediction
Inspired by BERT/MAE - masks random k-mers during training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskedAutoencoder(nn.Module):
    """
    Masks random k-mers during training and learns to reconstruct them.
    This naturally creates robust, denoising representations.
    """

    def __init__(self, input_dim=4096, latent_dim=512, mask_ratio=0.3):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.mask_ratio = mask_ratio

        # Learnable mask token (what we replace masked k-mers with)
        self.mask_token = nn.Parameter(torch.zeros(1, 1))

        # Encoder: learns from corrupted input
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, latent_dim),
        )

        # Decoder: reconstructs original from corrupted
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(2048, input_dim)
        )

    def create_mask(self, batch_size, device):
        """
        Create random mask for k-mers.
        Returns: Boolean mask where True = masked position
        """
        mask = torch.rand(batch_size, self.input_dim,
                          device=device) < self.mask_ratio
        return mask

    def apply_mask(self, x, mask):
        """
        Replace masked positions with mask token.
        """
        x_masked = x.clone()
        x_masked[mask] = self.mask_token
        return x_masked

    def forward(self, x, return_mask=False):
        """
        Forward pass with random masking.

        Args:
            x: Original k-mer counts (batch_size, 4096)
            return_mask: If True, return mask for loss computation

        Returns:
            x_recon: Reconstructed k-mers
            z: Latent embedding (512-dim)
            mask: Boolean mask (if return_mask=True)
        """
        batch_size = x.size(0)

        # Create random mask
        mask = self.create_mask(batch_size, x.device)

        # Apply mask
        x_masked = self.apply_mask(x, mask)

        # Encode masked input
        z = self.encoder(x_masked)

        # Decode to reconstruct
        x_recon = self.decoder(z)

        if return_mask:
            return x_recon, z, mask
        return x_recon, z

    def encode(self, x):
        """
        Extract embeddings without masking (for inference).
        """
        with torch.no_grad():
            z = self.encoder(x)
        return z


class MaskedAELoss(nn.Module):
    """
    Loss function for Masked Autoencoder.
    Only compute reconstruction loss on MASKED positions.
    """

    def __init__(self, lambda_recon=1.0, lambda_unmask=0.1):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_unmask = lambda_unmask

    def forward(self, x_recon, x_orig, mask):
        """
        Args:
            x_recon: Reconstructed k-mers
            x_orig: Original k-mers
            mask: Boolean mask (True = was masked)

        Returns:
            total_loss, masked_loss, unmasked_loss
        """
        # Loss on masked positions (main objective)
        masked_loss = F.mse_loss(x_recon[mask], x_orig[mask])

        # Small loss on unmasked positions (regularization)
        unmasked_loss = F.mse_loss(x_recon[~mask], x_orig[~mask])

        # Total loss
        total_loss = (self.lambda_recon * masked_loss +
                      self.lambda_unmask * unmasked_loss)

        return total_loss, masked_loss, unmasked_loss


def train_masked_autoencoder(
    model,
    train_loader,
    val_loader=None,
    n_epochs=100,
    lr=0.001,
    device='mps'
):
    """
    Train the masked autoencoder.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs)
    criterion = MaskedAELoss()

    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        train_masked_losses = []

        for batch_x in train_loader:
            batch_x = batch_x[0].to(device)  # Unpack from tuple

            # Forward pass with masking
            x_recon, z, mask = model(batch_x, return_mask=True)

            # Compute loss
            loss, masked_loss, unmask_loss = criterion(x_recon, batch_x, mask)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_masked_losses.append(masked_loss.item())

        scheduler.step()

        # Validation
        if val_loader is not None:
            model.eval()
            val_losses = []

            with torch.no_grad():
                for batch_x in val_loader:
                    batch_x = batch_x[0].to(device)
                    x_recon, z, mask = model(batch_x, return_mask=True)
                    loss, _, _ = criterion(x_recon, batch_x, mask)
                    val_losses.append(loss.item())

            val_loss = np.mean(val_losses)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'models/masked_ae_best.pt')

        # Print progress
        if (epoch + 1) % 5 == 0:
            train_loss = np.mean(train_losses)
            train_masked = np.mean(train_masked_losses)

            if val_loader:
                print(f"Epoch {epoch+1}/{n_epochs}: "
                      f"Train Loss={train_loss:.4f} (Masked={train_masked:.4f}), "
                      f"Val Loss={val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{n_epochs}: "
                      f"Train Loss={train_loss:.4f} (Masked={train_masked:.4f})")

    return model
