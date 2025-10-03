"""Model definitions ported from learner.ipynb.

Contains TimeEmbedding, TransformerBlock, TimeSeriesDiffusionModel, and DegDiffusion.
"""
import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t, L):
        # t: (N,) → expand to (N, L, dim)
        emb = self.mlp(t[:, None].float())   # (N, dim)
        return emb[:, None, :].repeat(1, L, 1)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=1, dim_ff=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # Feedforward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class TimeSeriesDiffusionModel(nn.Module):
    def __init__(self, channels, hidden_dim=32, num_blocks=2, T: int = 100):
        super().__init__()
        self.T = T
        self.input_proj = nn.Linear(channels, hidden_dim)
        self.time_emb = TimeEmbedding(hidden_dim)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, channels)

    def forward(self, s0, ns1, t):
        x = torch.cat([s0, ns1], dim=2)  # (N, C, L)
        N, C, L = x.shape
        x = x.permute(0, 2, 1)    # (N, L, C) → tokens
        x = self.input_proj(x)    # (N, L, hidden_dim)
        # 🚨 NOTE: Double-check adding here!
        x = x + self.time_emb(t, L)  # add time embedding (NOTE: this could be problematic)

        for block in self.blocks:
            x = block(x)

        x = self.output_proj(x)   # (N, L, C)
        x = x.permute(0, 2, 1)    # back to (N, C, L)
        return x

    def q_sample(self, x0, t, noise):
        """Diffuse the data (add Gaussian noise) at step t.

        Note: this method sets `self.betas` as a side-effect; sampling code expects that.
        """
        betas = torch.linspace(1e-6, 1e-1, steps=self.T).to(x0.device)
        self.betas = betas
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        sqrt_ab = torch.sqrt(alphas_bar[t])[:, None, None]
        sqrt_one_minus_ab = torch.sqrt(1 - alphas_bar[t])[:, None, None]

        return sqrt_ab * x0 + sqrt_one_minus_ab * noise


class DegDiffusion(nn.Module):
    def __init__(self, channels, sl, hidden=64, T: int = 100):
        super().__init__()
        self.T = T  # total diffusion steps
        self.sl = sl
        self.net1 = nn.Sequential(
            nn.Conv1d(channels, channels * 4, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(channels * 4 * self.sl, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.net2 = nn.Sequential(
            nn.Linear(hidden + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.sl)
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
                gain = 0.01 if m.out_features == self.sl else 1.0
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                m.weight.data.mul_(gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, s0, ns1, t):
        t = t.float().unsqueeze(1) / self.T
        x_net1 = torch.cat([s0, ns1], dim=2)  # available signal + noisy signal
        y_net1 = self.net1(x_net1)
        x_net2 = torch.cat([y_net1, t], dim=1)
        y_net2 = self.net2(x_net2)
        x_net3 = y_net2.unsqueeze(1)  # add channel dimension
        y_net3 = self.net3(x_net3)

        return y_net3

    def q_sample(self, x0, t, noise):
        """Diffuse the data (add Gaussian noise) at step t.

        Sets `self.betas` used by sampling.
        """
        betas = torch.linspace(1e-6, 1e-2, steps=self.T).to(x0.device)
        self.betas = betas
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        sqrt_ab = torch.sqrt(alphas_bar[t])[:, None, None]
        sqrt_one_minus_ab = torch.sqrt(1 - alphas_bar[t])[:, None, None]

        return sqrt_ab * x0 + sqrt_one_minus_ab * noise
