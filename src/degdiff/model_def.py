"""Model definitions for time-series diffusion models.
Contains TimeEmbedding, TransformerBlock, DegradationDiffusion (abstract base), TimeSeriesDiffusionModel, and DegDiffusion.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional

class TimeEmbedding(nn.Module):
    """Embeds time steps into a fixed dimension."""
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t: torch.Tensor, L: int) -> torch.Tensor:
        # t: (N,) → expand to (N, L, dim)
        emb = self.mlp(t[:, None].float())  # (N, dim)
        return emb[:, None, :].repeat(1, L, 1)  # (N, L, dim)

class TransformerBlock(nn.Module):
    """Transformer block for sequence processing."""
    def __init__(self, dim: int, heads: int = 1, dim_ff: int = 128):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # Feedforward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class DegradationDiffusion(ABC, nn.Module):
    """Abstract base class for degradation diffusion models.
    Handles diffusion parameters (betas, alphas) and sampling logic.
    Subclasses must implement forward for noise prediction.
    """
    def __init__(self, T: int = 100, seq_len: Optional[int] = None):
        super().__init__()
        self.T = T
        self.seq_len = seq_len  # Optional: fixed sequence length (for fixed-length models)
        betas = torch.linspace(1e-6, 1e-2, steps=T)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # Register as buffers (device-flexible, saved in state_dict)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Diffuse the data (add Gaussian noise) at step t."""
        device = x0.device
        sqrt_ab = torch.sqrt(self.alphas_bar.to(device)[t])[:, None, None]
        sqrt_one_minus_ab = torch.sqrt(1 - self.alphas_bar.to(device)[t])[:, None, None]
        return sqrt_ab * x0 + sqrt_one_minus_ab * noise

    @torch.no_grad()
    def sample(self, x0: torch.Tensor, s0_len: int) -> torch.Tensor:
        """Sample: start from s0 (available signal) and predict the missing part.
        Supports variable L via subclass forward.
        """
        device = x0.device
        N, C, L = x0.shape
        if self.seq_len is not None and L != self.seq_len:
            raise ValueError(f"Model fixed to seq_len={self.seq_len}, but input L={L}")
        x = torch.randn(N, C, L, device=device)
        betas = self.betas.to(device)
        alphas = self.alphas.to(device)
        alphas_bar = self.alphas_bar.to(device)
        for step in reversed(range(self.T)):
            t_batch = torch.full((N,), step, device=device, dtype=torch.long)
            noise_pred = self.forward(x0[:, :, :s0_len], x[:, :, s0_len:], t_batch)
            alpha = alphas[step]
            alpha_bar = alphas_bar[step]
            sqrt_alpha = torch.sqrt(alpha)
            sqrt_one_minus_ab = torch.sqrt(1 - alpha_bar)
            x = (1.0 / sqrt_alpha) * (x - (1 - alpha) / sqrt_one_minus_ab * noise_pred)
            if step > 0:
                x = x + torch.sqrt(betas[step]) * torch.randn_like(x)
        out = torch.cat([x0[:, :, :s0_len], x[:, :, s0_len:]], dim=2)
        return out

    @abstractmethod
    def forward(self, s0: torch.Tensor, ns1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise; must be implemented by subclasses."""
        pass

    def _init_weights(self):
        """Optional weight initialization; can be overridden in subclasses."""
        pass

class TransformerDiffusionModel(DegradationDiffusion):
    """Diffusion model using transformer blocks; supports variable sequence lengths."""
    def __init__(self, channels: int = 1, hidden_dim: int = 32, num_blocks: int = 2, T: int = 100):
        super().__init__(T=T)  # seq_len=None for variable length
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(channels, hidden_dim)
        self.time_emb = TimeEmbedding(hidden_dim)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, channels)

    def forward(self, s0: torch.Tensor, ns1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s0, ns1], dim=2)  # (N, C, L)
        N, C, L = x.shape
        if C != self.channels:
            raise ValueError(f"Expected channels={self.channels}, got {C}")
        x = x.permute(0, 2, 1)  # (N, L, C) → tokens
        x = self.input_proj(x)  # (N, L, hidden_dim)
        time_emb = self.time_emb(t, L)  # (N, L, hidden_dim)
        x = x + time_emb  # Shapes match: add time embedding
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)  # (N, L, C)
        x = x.permute(0, 2, 1)  # back to (N, C, L)
        return x

class DegDiffusion(DegradationDiffusion):
    """Fixed-length diffusion model using conv and linear layers."""
    def __init__(self, channels: int = 1, seq_len: int = 100, hidden: int = 64, T: int = 100):
        super().__init__(T=T, seq_len=seq_len)  # Required seq_len for fixed architecture
        self.channels = channels
        self.hidden = hidden
        self.net1 = nn.Sequential(
            nn.Conv1d(channels, channels * 4, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(channels * 4 * seq_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.net2 = nn.Sequential(
            nn.Linear(hidden + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, seq_len)
        )
        self.net3 = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                gain = 0.01 if m.out_features == self.seq_len else 1.0
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                m.weight.data.mul_(gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, s0: torch.Tensor, ns1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        N, C, L_s0 = s0.shape
        _, _, L_ns = ns1.shape
        full_L = L_s0 + L_ns
        if C != self.channels or full_L != self.seq_len:
            raise ValueError(f"Expected channels={self.channels}, seq_len={self.seq_len}, got C={C}, L={full_L}")
        t = t.float().unsqueeze(1) / self.T  # Normalize t
        x_net1 = torch.cat([s0, ns1], dim=2)  # (N, C, seq_len)
        y_net1 = self.net1(x_net1)
        x_net2 = torch.cat([y_net1, t], dim=1)
        y_net2 = self.net2(x_net2)
        x_net3 = y_net2.unsqueeze(1)  # (N, 1, seq_len)
        y_net3 = self.net3(x_net3)
        return y_net3