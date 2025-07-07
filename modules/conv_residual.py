import torch.nn as nn
import torch

import sys
sys.path.insert(0, "C:/Users/asalvi/Documents/Ameya_workspace/DiffusionDataset/ConeCamAngEst/training/") 

from modules.conv1d import Conv1dBlock  # Import from conv1d module

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8,cond_predict_scale=None):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        self.cond_dim = cond_dim
        self.out_channels = out_channels

        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
        )

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        if cond.size(1) != self.cond_dim:
            cond = cond[:, :self.cond_dim]

        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.view(embed.size(0), 2, self.out_channels)
        scale, bias = embed[:, 0, :].unsqueeze(-1), embed[:, 1, :].unsqueeze(-1)
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
