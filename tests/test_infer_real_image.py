import os
import torch
from PIL import Image
import torchvision.transforms as T

from src.models.vit_liveness import LivenessViT
from utils.build_transforms import build_transforms


def test_infer_real_image_exists_and_forward():
    # Image placed under tests/ per user input
    img_path = os.path.join(os.path.dirname(__file__), 'spoof_sample.png')
    assert os.path.isfile(img_path), f"Test image not found: {img_path}"

    img = Image.open(img_path).convert('RGB')
    tfm = build_transforms(224)
    x = tfm(img).unsqueeze(0)  # [1,3,224,224]

    model = LivenessViT(img_size=224, patch_size=16, d_model=64, nhead=4, num_layers=1, num_classes=2)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    # Shape check
    assert logits.shape == (1, 2)
    # Probabilities sanity (sum ~ 1)
    s = float(probs.sum().item())
    assert 0.99 <= s <= 1.01

    # Print for debug visibility when running directly
    print({'live_prob': float(probs[0]), 'spoof_prob': float(probs[1])})


if __name__ == '__main__':
    test_infer_real_image_exists_and_forward()
    print('Real image inference test passed')
