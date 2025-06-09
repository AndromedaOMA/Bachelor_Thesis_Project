import torch
import torch.nn as nn


class FrequencyAttention(nn.Module):
    def __init__(self, hidden_dim: int = 32, heads: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(1, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """IN/OUT: (batch_frames, 1, freq_bins)"""
        x = x.permute(0, 2, 1)
        x_proj = self.input_proj(x)
        attn_output, _ = self.attn(x_proj, x_proj, x_proj)
        out = self.output_proj(attn_output)
        out = out.permute(0, 2, 1)
        mask = self.sigmoid(out)
        out = x.permute(0, 2, 1) * mask
        return out
