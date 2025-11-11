#!/bin/bash

# Face Liveness Model Upload Helper Script
# This script helps you upload your trained model to Hugging Face Hub

echo "=========================================="
echo "Face Liveness Model Upload to Hugging Face"
echo "=========================================="
echo ""

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo "Usage: ./scripts/upload_helper.sh <checkpoint_path> [repo_name] [hf_token]"
    echo ""
    echo "Example:"
    echo "  ./scripts/upload_helper.sh checkpoints/best.pt your-username/face-liveness-vit YOUR_HF_TOKEN"
    echo ""
    echo "Available checkpoints:"
    ls -lh checkpoints/
    exit 1
fi

CHECKPOINT=$1
REPO_NAME=${2:-""}
HF_TOKEN=${3:-""}

# Validate checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo "✅ Using checkpoint: $CHECKPOINT"
echo ""

# Get repo name if not provided
if [ -z "$REPO_NAME" ]; then
    echo "Enter your Hugging Face repository name (format: username/repo-name):"
    echo "Example: hoangnguyenduc3009/face-liveness-vit"
    read -p "Repo name: " REPO_NAME
fi

if [ -z "$REPO_NAME" ]; then
    echo "❌ Error: Repository name is required"
    exit 1
fi

echo "✅ Repository: $REPO_NAME"
echo ""

# Get HF token if not provided
if [ -z "$HF_TOKEN" ]; then
    echo "Enter your Hugging Face API token:"
    echo "Get it from: https://huggingface.co/settings/tokens"
    echo "(Make sure it has 'write' access)"
    read -sp "Token: " HF_TOKEN
    echo ""
fi

if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: Hugging Face token is required"
    exit 1
fi

echo ""
echo "=========================================="
echo "Upload Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Repository: $REPO_NAME"
echo "  Visibility: Public (use --private flag for private)"
echo "=========================================="
echo ""

read -p "Proceed with upload? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "Upload cancelled."
    exit 0
fi

echo ""
echo "🚀 Starting upload..."
echo ""

# Run the upload script
python scripts/upload_to_huggingface.py \
    --checkpoint "$CHECKPOINT" \
    --repo_name "$REPO_NAME" \
    --token "$HF_TOKEN" \
    --commit_message "Upload face liveness detection model trained on CelebA-Spoof"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Upload completed successfully!"
    echo "=========================================="
    echo ""
    echo "Your model is now available at:"
    echo "https://huggingface.co/$REPO_NAME"
    echo ""
    echo "You can test it with:"
    echo "  python scripts/test_huggingface_model.py --repo $REPO_NAME --image path/to/test/image.jpg"
else
    echo ""
    echo "❌ Upload failed. Please check the error messages above."
    exit 1
fi
