"""
Conditional Diffusion Model for Probabilistic AMR Prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiffusionSchedule:
    """Noise schedule for forward/reverse diffusion"""
    
    def __init__(self, n_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.n_steps = n_steps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # Precompute values
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
    
    def to(self, device):
        """Move schedule to device"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        return self
    
    def add_noise(self, x0, t):
        """Forward diffusion: add noise at timestep t"""
        noise = torch.randn_like(x0)
        
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        
        return x_t, noise


class ConditionalDiffusionModel(nn.Module):
    """Learns p(phenotype | genome_embedding)"""
    
    def __init__(self, genome_dim=512, phenotype_dim=3, hidden_dim=256):
        super().__init__()
        
        self.genome_dim = genome_dim
        self.phenotype_dim = phenotype_dim
        
        # Encode genome
        self.genome_encoder = nn.Sequential(
            nn.Linear(genome_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Time embedding
        self.time_dim = 128
        
        # Denoising network
        self.denoiser = nn.Sequential(
            nn.Linear(phenotype_dim + hidden_dim // 2 + self.time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, phenotype_dim)
        )
    
    def get_time_embedding(self, t, max_period=10000):
        """Sinusoidal time embedding"""
        half_dim = self.time_dim // 2
        embeddings = np.log(max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings
    
    def forward(self, phenotype_noisy, genome_embedding, t):
        """Predict noise to remove"""
        genome_context = self.genome_encoder(genome_embedding)
        t_emb = self.get_time_embedding(t)
        x = torch.cat([phenotype_noisy, genome_context, t_emb], dim=-1)
        noise_pred = self.denoiser(x)
        return noise_pred
    
    @torch.no_grad()
    def sample(self, genome_embedding, schedule, n_samples=100, device='cpu'):
        """Sample phenotype distribution"""
        if genome_embedding.dim() == 1:
            genome_embedding = genome_embedding.unsqueeze(0)
        
        genome_embedding = genome_embedding.repeat(n_samples, 1).to(device)
        x = torch.randn(n_samples, self.phenotype_dim, device=device)
        
        for t in reversed(range(schedule.n_steps)):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            noise_pred = self(x, genome_embedding, t_batch)
            
            alpha_t = schedule.alphas[t]
            alpha_bar_t = schedule.alpha_bars[t]
            beta_t = schedule.betas[t]
            
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
            mean = coef1 * (x - coef2 * noise_pred)
            
            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = mean + sigma_t * noise
            else:
                x = mean
        
        phenotype_probs = F.softmax(x, dim=-1)
        return phenotype_probs

