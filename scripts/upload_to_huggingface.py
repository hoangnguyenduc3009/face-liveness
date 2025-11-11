"""
Upload Face Liveness Detection model to Hugging Face Hub.

This script uploads the trained model checkpoint, configuration, and necessary files
to Hugging Face Hub as a sandbox repository.

Usage:
    python scripts/upload_to_huggingface.py --checkpoint checkpoints/best.pt --repo_name your-username/face-liveness-vit --token YOUR_HF_TOKEN
"""

import argparse
import os
import sys
import torch
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.vit_liveness import LivenessViT


def create_model_card(repo_name: str, config: dict, threshold: float = 0.5) -> str:
    """Create a comprehensive README.md for the model card."""
    
    model_cfg = config.get('model', {})
    train_cfg = config.get('train', {})
    
    model_card = f"""---
tags:
- face-liveness-detection
- anti-spoofing
- computer-vision
- pytorch
- vision-transformer
license: apache-2.0
---

# Face Liveness Detection Model

This model performs **face liveness detection** to distinguish between real faces and spoofing attempts (e.g., printed photos, video replay attacks, masks).

## Model Description

- **Architecture**: Vision Transformer (ViT) based encoder
- **Image Size**: {model_cfg.get('image_size', 224)}x{model_cfg.get('image_size', 224)}
- **Patch Size**: {model_cfg.get('patch_size', 16)}x{model_cfg.get('patch_size', 16)}
- **Embedding Dimension**: {model_cfg.get('d_model', 128)}
- **Attention Heads**: {model_cfg.get('nhead', 4)}
- **Transformer Layers**: {model_cfg.get('num_layers', 2)}
- **Classes**: Live (0) vs Spoof (1)
- **Optimal Threshold**: {threshold:.4f}

## Training Details

- **Dataset**: CelebA-Spoof
- **Batch Size**: {train_cfg.get('batch_size', 64)}
- **Learning Rate**: {train_cfg.get('lr', 3e-4)}
- **Weight Decay**: {train_cfg.get('weight_decay', 0.05)}
- **Epochs**: {train_cfg.get('epochs', 30)}
- **Optimizer**: AdamW
- **Mixed Precision**: {train_cfg.get('amp', True)}

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
import torchvision.transforms as transforms

# Download model
model_path = hf_hub_download(repo_id="{repo_name}", filename="model.pt")

# Load checkpoint
checkpoint = torch.load(model_path, map_location='cpu')

# Initialize model (you'll need the model class definition)
model = LivenessViT(
    img_size={model_cfg.get('image_size', 224)},
    patch_size={model_cfg.get('patch_size', 16)},
    d_model={model_cfg.get('d_model', 128)},
    nhead={model_cfg.get('nhead', 4)},
    num_layers={model_cfg.get('num_layers', 2)},
    num_classes=2
)
model.load_state_dict(checkpoint['model'])
model.eval()

# Prepare image
size = 224
transform = transforms.Compose([
    transforms.Resize(int(size * 1.14)),
    transforms.CenterCrop(size),
    # Ensure validation images are also grayscaled (keeps 3 channels)
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("face.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)
    spoof_prob = probs[0, 1].item()
    
    # Use optimal threshold
    is_live = spoof_prob < {threshold:.4f}
    print(f"Live: {{is_live}}, Spoof probability: {{spoof_prob:.4f}}")
```

## Model Performance

The model was trained on CelebA-Spoof dataset with class balancing and data augmentation.
For detailed evaluation metrics, please refer to the training logs.

## Limitations

- Trained specifically on CelebA-Spoof dataset
- Performance may vary on different demographics or imaging conditions
- Should be used as part of a comprehensive security system, not as sole authentication

## Citation

If you use this model, please cite:

```bibtex
@misc{{face-liveness-vit,
  author = {{Your Name}},
  title = {{Face Liveness Detection with Vision Transformer}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{repo_name}}}}}
}}
```

## License

Apache 2.0
"""
    return model_card


def create_inference_script() -> str:
    """Create a standalone inference script."""
    
    inference_code = """import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x):
        return self.encoder(x)

class LivenessViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        dim_feedforward: int = None,
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
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


def load_model(checkpoint_path: str, device: str = 'cpu') -> Tuple[LivenessViT, dict]:
    \"\"\"Load model from checkpoint.\"\"\"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    cfg = checkpoint.get('cfg', {})
    model_cfg = cfg.get('model', {})
    
    model = LivenessViT(
        img_size=model_cfg.get('image_size', 224),
        patch_size=model_cfg.get('patch_size', 16),
        d_model=model_cfg.get('d_model', 128),
        nhead=model_cfg.get('nhead', 4),
        num_layers=model_cfg.get('num_layers', 2),
        num_classes=2,
        dropout=model_cfg.get('drop_rate', 0.0),
    )
    
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    return model, checkpoint


def predict(model: LivenessViT, image_path: str, threshold: float = 0.5, device: str = 'cpu') -> dict:
    \"\"\"Predict if face is live or spoofed.\"\"\"
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        spoof_prob = probs[0, 1].item()
    
    is_live = spoof_prob < threshold
    
    return {
        'is_live': is_live,
        'spoof_probability': spoof_prob,
        'live_probability': 1 - spoof_prob,
        'prediction': 'LIVE' if is_live else 'SPOOF'
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [checkpoint_path] [threshold]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else "model.pt"
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading model from {checkpoint_path}...")
    model, checkpoint = load_model(checkpoint_path, device)
    model = model.to(device)
    
    # Use threshold from checkpoint if available
    if 'threshold' in checkpoint:
        threshold = checkpoint['threshold']
        print(f"Using optimal threshold from checkpoint: {threshold:.4f}")
    
    print(f"Predicting on {image_path}...")
    result = predict(model, image_path, threshold, device)
    
    print(f"\\nResult: {result['prediction']}")
    print(f"Live probability: {result['live_probability']:.4f}")
    print(f"Spoof probability: {result['spoof_probability']:.4f}")
"""
    return inference_code


def upload_to_huggingface(
    checkpoint_path: str,
    repo_name: str,
    token: str,
    private: bool = False,
    commit_message: str = "Upload face liveness detection model"
):
    """Upload model and files to Hugging Face Hub."""
    
    try:
        from huggingface_hub import HfApi, create_repo, upload_file
    except ImportError as e:
        print(f"Error: huggingface_hub is not installed.")
        print(f"Import error details: {e}")
        print(f"Python executable: {sys.executable}")
        print("Please install it: pip install huggingface_hub")
        sys.exit(1)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('cfg', {})
    threshold = checkpoint.get('threshold', 0.5)
    
    # Ensure HuggingFace cache directory exists
    hf_cache_dir = Path.home() / ".cache" / "huggingface"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize HF API with token
    api = HfApi(token=token)
    
    # Create repository
    print(f"Creating repository {repo_name}...")
    try:
        create_repo(
            repo_id=repo_name,
            token=token,
            private=private,
            repo_type="model",
            exist_ok=True
        )
        print(f"Repository created/verified: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create temporary directory for files to upload
    temp_dir = Path("temp_hf_upload")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Save model checkpoint
        model_path = temp_dir / "model.pt"
        torch.save(checkpoint, model_path)
        print(f"Saved model to {model_path}")
        
        # Save config
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Saved config to {config_path}")
        
        # Create model card
        readme_path = temp_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(create_model_card(repo_name, config, threshold))
        print(f"Created model card at {readme_path}")
        
        # Create inference script
        inference_path = temp_dir / "inference.py"
        with open(inference_path, 'w') as f:
            f.write(create_inference_script())
        print(f"Created inference script at {inference_path}")
        
        # Upload files
        print("\nUploading files to Hugging Face Hub...")
        files_to_upload = [
            ("model.pt", model_path),
            ("config.yaml", config_path),
            ("README.md", readme_path),
            ("inference.py", inference_path),
        ]
        
        for filename, filepath in files_to_upload:
            print(f"Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=repo_name,
                token=token,
                commit_message=f"{commit_message}: {filename}"
            )
        
        print(f"\n✅ Successfully uploaded model to: https://huggingface.co/{repo_name}")
        print(f"\nOptimal threshold: {threshold:.4f}")
        
    finally:
        # Cleanup temp directory
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")


def main():
    parser = argparse.ArgumentParser(description="Upload Face Liveness Detection model to Hugging Face Hub")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g., checkpoints/best.pt)"
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="Hugging Face repository name (format: username/repo-name)"
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Hugging Face API token (get from https://huggingface.co/settings/tokens)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload face liveness detection model",
        help="Commit message for upload"
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
    
    # Validate repo name format
    if "/" not in args.repo_name:
        print("Error: repo_name must be in format 'username/repo-name'")
        sys.exit(1)
    
    upload_to_huggingface(
        checkpoint_path=args.checkpoint,
        repo_name=args.repo_name,
        token=args.token,
        private=args.private,
        commit_message=args.commit_message
    )


if __name__ == "__main__":
    main()
