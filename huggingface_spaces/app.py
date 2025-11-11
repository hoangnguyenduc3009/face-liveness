"""
Hugging Face Spaces app for Face Liveness Detection.
Deploy this to HF Spaces for permanent hosting.
"""

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


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
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x):
        return self.transformer_encoder(x)


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


# Load model
def load_model():
    """Load model from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download
    
    model_path = hf_hub_download(
        repo_id="hoangnguyenduc3009/face-liveness-vit", 
        filename="model.pt"
    )
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
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
    
    threshold = 0.09
    
    return model, threshold


# Initialize model
print("Loading model...")
model, threshold = load_model()
print(f"Model loaded! Using threshold: {threshold:.4f}")

# Transform
size = 224
transform = transforms.Compose([
    transforms.Resize(int(size * 1.14)),
    transforms.CenterCrop(size),
    # Ensure validation images are also grayscaled (keeps 3 channels)
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_liveness(image):
    """Predict if face is live or spoofed."""
    print("Predicting liveness...")
    if image is None:
        return "Please upload an image", None
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Ensure RGB
    image = image.convert("RGB")
    
    # Transform
    input_tensor = transform(image).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        live_prob = probs[0, 0].item()
        spoof_prob = probs[0, 1].item()
    
    # Determine result
    is_live = spoof_prob < threshold
    
    # Create result text
    if is_live:
        result = f"✅ **LIVE FACE** (Confidence: {live_prob*100:.2f}%)"
    else:
        result = f"❌ **SPOOF DETECTED** (Spoof probability: {spoof_prob*100:.2f}%)"
    
    # Create confidence dict for gradio
    confidence = {
        "Live": live_prob,
        "Spoof": spoof_prob
    }
    
    return result, confidence


# Create Gradio interface
demo = gr.Interface(
    fn=predict_liveness,
    inputs=gr.Image(type="pil", label="Upload Face Image"),
    outputs=[
        gr.Markdown(label="Result"),
        gr.Label(label="Confidence Scores", num_top_classes=2)
    ],
    title="🔒 Face Liveness Detection",
    description="""
    Upload a face image to detect if it's a **real person** or a **spoofing attempt** (photo, video replay, mask, etc.).
    
    This model uses a Vision Transformer (ViT) trained on CelebA-Spoof dataset.
    
    **Note**: Best results with clear, front-facing face images.
    
    **Model**: [hoangnguyenduc3009/face-liveness-vit](https://huggingface.co/hoangnguyenduc3009/face-liveness-vit)
    """,
    examples=[],
    theme="soft",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(share=True)
