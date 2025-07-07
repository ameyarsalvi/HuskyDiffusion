import torch.nn as nn
import torch
import sys

sys.path.insert(0, "C:/Users/asalvi/Documents/Ameya_workspace/DiffusionDataset/ConeCamAngEst/training/") 

from modules.conv_residual import ConditionalResidualBlock1D
from modules.pose_embedding import SinusoidalPosEmb

class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, down_dims=[256, 512, 1024], kernel_size=5, n_groups=8):
        super().__init__()
        self.global_cond_dim = global_cond_dim
        dims = [input_dim] + down_dims
        self.down_modules = nn.ModuleList()
        self.up_modules = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.down_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(dims[i], dims[i + 1], global_cond_dim, kernel_size, n_groups),
                    nn.Conv1d(dims[i + 1], dims[i + 1], kernel_size=4, stride=2, padding=1)
                ])
            )
            self.up_modules.insert(0, nn.ModuleList([
                nn.ConvTranspose1d(dims[i + 1], dims[i], kernel_size=4, stride=2, padding=1),
                ConditionalResidualBlock1D(dims[i] * 2, dims[i], global_cond_dim, kernel_size, n_groups)
            ]))

        self.channel_reduction = nn.Conv1d(dims[1], dims[0], kernel_size=1)
        self.final_conv = nn.Conv1d(dims[0], input_dim, 1)

    def forward(self, x, timesteps, global_cond):
        # Ensure input shape is (batch, seq_len, channels)
        if x.ndim == 3:
            x = x.permute(0, 2, 1)  # Convert to (batch, channels, seq_len)
        elif x.ndim == 2:
            x = x.unsqueeze(1).permute(0, 2, 1)  # Expand and permute if needed
        else:
            raise ValueError(f"Unexpected input dimensions for x: {x.shape}")

        # Positional Embedding & Global Conditioning
        global_feature = SinusoidalPosEmb(self.global_cond_dim)(timesteps)
        global_feature = torch.cat([global_feature, global_cond], dim=-1)
        #print(f"global feature shape{global_feature.shape}")

        skips = []
        '''
        for i, (resblock, downsample) in enumerate(self.down_modules):
            print(f"Downsampling step {i}: Before resblock, x shape = {x.shape}")
            x = resblock(x, global_feature)
            print(f"Downsampling step {i}: After resblock, x shape = {x.shape}")
            skips.append(x)  # Store for skip connections
            x = downsample(x)  # Downsample
            print(f"Downsampling step {i}: After downsample, x shape = {x.shape}")

        for upsample, resblock in self.up_modules:
            if not skips:
                break
            skip = skips.pop()

            # Upsample and ensure output size matches
            x = upsample(x, output_size=(skip.shape[-1],))

            # Ensure x and skip have the same sequence length
            min_len = min(x.shape[-1], skip.shape[-1])
            x, skip = x[:, :, :min_len], skip[:, :, :min_len]

            # Ensure x has the same number of channels as skip before concatenation
            if x.shape[1] != skip.shape[1]:
                channel_match = nn.Conv1d(x.shape[1], skip.shape[1], kernel_size=1).to(x.device)
                x = channel_match(x)  # ✅ Match channels dynamically

            x = torch.cat((x, skip), dim=1)  # Concatenate
            x = resblock(x, global_feature)  # Apply residual block'
        '''
        for resblock, downsample in self.down_modules:
            x = resblock(x, global_feature)
            if x.size(-1) < 2:  # Prevent further downsampling if spatial dimensions are too small
                break
            skips.append(x)
            x = downsample(x)
        for upsample, resblock in self.up_modules:
            if not skips:
                break
            skip = skips.pop()
            x = upsample(x)
            x = torch.cat((x, skip), dim=1)
            x = resblock(x, global_feature)
        x = self.channel_reduction(x)

        # Final processing
        x = self.channel_reduction(x)
        return self.final_conv(x).permute(0, 2, 1)  # Convert back to (batch, seq_len, channels)


# ✅ **Testing the Model**
if __name__ == '__main__':
    x = torch.randn(4, 100, 512)
    print(f"x is:{x}")
    timesteps = torch.randint(0, 1000, (4,))
    print(f"timesteps are is:{timesteps}")
    global_cond = torch.randn(4, 16)
    print(f"global_cond are is:{global_cond}")

    model = ConditionalUnet1D(512, 16)
    #print(model)
    output = model(x, timesteps, global_cond)

    #print(f"Output Shape: {output.shape}")  # Should match input shape
