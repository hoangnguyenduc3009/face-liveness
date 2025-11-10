import argparse
import os
from typing import Optional

import torch
import torchvision.transforms as T
from PIL import Image

from src.models.vit_liveness import LivenessViT
from src.utils.config import load_yaml_config
from utils.build_transforms import build_transforms

essential_keys = [
    ("model", "image_size", 224),
    ("model", "patch_size", 16),
    ("model", "d_model", 128),
    ("model", "nhead", 4),
    ("model", "num_layers", 2),
    ("model", "num_classes", 2),
]


def build_model(cfg: dict) -> LivenessViT:
    # Pull values with safe defaults if missing
    kwargs = {}
    for section, key, default in essential_keys:
        val = cfg.get(section, {}).get(key, default)
        kwargs[key] = val
    model = LivenessViT(
        img_size=kwargs["image_size"],
        patch_size=kwargs["patch_size"],
        d_model=kwargs["d_model"],
        nhead=kwargs["nhead"],
        num_layers=kwargs["num_layers"],
        num_classes=kwargs["num_classes"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Inference on a single image for liveness detection")
    parser.add_argument("--image", required=True, help="Path to input image (absolute or relative)")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--root", default=None, help="Dataset root for resolving relative 'Data/...' paths")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device selection")
    args = parser.parse_args()

    # Resolve image path
    img_path = args.image
    if args.root and not os.path.isabs(img_path):
        img_path = os.path.join(args.root, img_path)
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Load config and build model
    cfg = load_yaml_config(args.config)
    model = build_model(cfg)

    # Load checkpoint if provided
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device)
    model.eval()

    # Transforms
    val_tfm = build_transforms(cfg)['val']

    # Load and preprocess image
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        x = val_tfm(im).unsqueeze(0).to(device)  # [1,3,H,W]

    # Inference
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        live_prob = float(probs[0].item())
        spoof_prob = float(probs[1].item())
        pred = int(spoof_prob >= 0.5)

    print({
        "image": args.image,
        "pred": pred,            # 0=live, 1=spoof
        "live_prob": round(live_prob, 6),
        "spoof_prob": round(spoof_prob, 6),
        "checkpoint": args.checkpoint,
        "device": str(device),
    })


if __name__ == "__main__":
    main()
