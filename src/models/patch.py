import torch.nn as nn
import torch

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, d_model=128):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.proj(x)                # [B, d_model, H/ps, W/ps]
        x = x.flatten(2).transpose(1,2) # [B, N, d_model]
        return x
