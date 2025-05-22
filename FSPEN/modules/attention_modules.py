import torch
import torch.nn as nn


class FrequencyAttention(nn.Module):
    def __init__(self, freq_bins, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=1),  # across freqs
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),  # back to 1 channel
            nn.Sigmoid()  # soft mask
        )

    def forward(self, x):
        # x: (batch, frames, 1, freq_bins)
        b, t, c, f = x.shape
        x_perm = x.reshape(b * t, c, f)  # (batch*frames, 1, freq)
        print("Shape input conv1d:", x_perm.shape)

        attn = self.net(x_perm)  # (batch*frames, 1, freq)
        attn = attn.reshape(b, t, c, f)  # reconstrucție
        return x * attn  # (batch, frames, 1, freq)
