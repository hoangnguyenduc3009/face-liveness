---
title: Face Liveness Detection
emoji: 🔒
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
python_version: 3.12.3
---

# Face Liveness Detection Space

This Hugging Face Space provides a demo for the Face Liveness Detection model using Vision Transformer (ViT).

## Model

The model is hosted at: [hoangnguyenduc3009/face-liveness-vit](https://huggingface.co/hoangnguyenduc3009/face-liveness-vit)

## Features

- ✅ Detect live faces vs spoofing attempts
- 🎯 Vision Transformer architecture
- 📊 Trained on CelebA-Spoof dataset
- ⚡ Fast inference on CPU

## Usage

1. Upload a face image
2. Get instant prediction: LIVE or SPOOF
3. View confidence scores

## Deployment Instructions

### Option 1: Manual Upload to HF Spaces

1. Create a new Space at https://huggingface.co/new-space
2. Select "Gradio" as SDK
3. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `README.md` (this file)

### Option 2: Using Git

```bash
# Clone your space
git clone https://huggingface.co/spaces/your-username/face-liveness-demo
cd face-liveness-demo

# Copy files
cp app.py requirements.txt README.md /path/to/cloned/space/

# Commit and push
git add .
git commit -m "Initial commit"
git push
```

### Option 3: Using Hugging Face CLI

```bash
# Install huggingface-cli
pip install huggingface_hub[cli]

# Login
huggingface-cli login

# Create and upload space
huggingface-cli upload-space . your-username/face-liveness-demo
```

## Local Testing

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:7860

## Model Details

- **Architecture**: Vision Transformer (ViT)
- **Image Size**: 224x224
- **Classes**: Live (0) vs Spoof (1)
- **Optimal Threshold**: 0.1199

## License

Apache 2.0
