"""
Masked Autoencoder for Genomic AMR Prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaskedAutoencoder(nn.Module):
    def __init__(self, input_dim=4096, latent_dim=512, mask_ratio=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.mask_ratio = mask_ratio
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1))
        
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
        mask = torch.rand(batch_size, self.input_dim, device=device) < self.mask_ratio
        return mask
    
    def apply_mask(self, x, mask):
        x_masked = x.clone()
        x_masked[mask] = self.mask_token
        return x_masked
    
    def forward(self, x, return_mask=False):
        batch_size = x.size(0)
        mask = self.create_mask(batch_size, x.device)
        x_masked = self.apply_mask(x, mask)
        z = self.encoder(x_masked)
        x_recon = self.decoder(z)
        
        if return_mask:
            return x_recon, z, mask
        return x_recon, z
    
    def encode(self, x):
        with torch.no_grad():
            z = self.encoder(x)
        return z


class MaskedAELoss(nn.Module):
    def __init__(self, lambda_recon=1.0, lambda_unmask=0.1):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_unmask = lambda_unmask
    
    def forward(self, x_recon, x_orig, mask):
        masked_loss = F.mse_loss(x_recon[mask], x_orig[mask])
        unmasked_loss = F.mse_loss(x_recon[~mask], x_orig[~mask])
        total_loss = self.lambda_recon * masked_loss + self.lambda_unmask * unmasked_loss
        return total_loss, masked_loss, unmasked_loss

