import torch
from src.models.vit_liveness import LivenessViT


def test_forward_basic():
    model = LivenessViT(img_size=224, patch_size=16, d_model=64, nhead=4, num_layers=1, num_classes=2)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 2)


def test_forward_different_width():
    model = LivenessViT(img_size=224, patch_size=16, d_model=128, nhead=4, num_layers=2, num_classes=2)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 2)


def test_patch_count_consistency():
    # 224 / 16 = 14 => 196 patches + 1 CLS
    model = LivenessViT(img_size=224, patch_size=16, d_model=32, nhead=4, num_layers=1, num_classes=2)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        # Tap into internal embeddings
        patches = model.patch_embed(x)
        assert patches.shape[1] == (224 // 16) ** 2  # N
        # Simulate forward up to encoder to ensure positional trim works
        cls = model.cls_token.expand(1, -1, -1)
        seq = torch.cat((cls, patches), dim=1)
        seq = seq + model.pos_embed[:, :seq.size(1), :]
        out = model.encoder(seq)
        assert out.shape[1] == patches.shape[1] + 1


if __name__ == '__main__':
    test_forward_basic()
    test_forward_different_width()
    test_patch_count_consistency()
    print('All forward tests passed')