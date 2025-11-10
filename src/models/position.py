import torch.nn as nn
import torch

class PositionEmbed(nn.Module):
    def __init__(self, num_patches, d_model):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1), :]
