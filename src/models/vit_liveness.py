from typing import Optional
from src.models.encoder import Encoder
from src.models.patch import PatchEmbed
import torch
import torch.nn as nn

class LivenessViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2 + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.encoder = Encoder(d_model, nhead, num_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)          # [B, N, d_model]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)              # [B, 1+N, d_model]
        cls_out = x[:, 0]                # lấy CLS token
        return self.head(cls_out)
