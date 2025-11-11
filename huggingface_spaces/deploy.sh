#!/bin/bash
# Script tự động deploy lên Hugging Face Spaces

set -e

echo "🚀 Deploying Face Liveness Detection to Hugging Face Spaces..."

# Configuration
SPACE_NAME="hoangnguyenduc3009/face-liveness-vit"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "📦 Installing huggingface-cli..."
    pip install "huggingface_hub[cli]"
fi

# Login check
echo "🔐 Checking authentication..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "❌ Not logged in. Please login to Hugging Face:"
    huggingface-cli login
    exit 1
fi

echo "✅ Authentication verified"
echo ""

# Create Space using Python API (CLI doesn't have create command)
echo "📁 Creating Space: $SPACE_NAME..."
python3 << 'EOF'
from huggingface_hub import HfApi
api = HfApi()
try:
    api.create_repo(
        repo_id="hoangnguyenduc3009/face-liveness-vit",
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True
    )
    print("✅ Space created/verified")
except Exception as e:
    print(f"⚠️ Warning: {e}")
    print("Continuing with upload...")
EOF

echo ""
echo "📤 Uploading files..."

# Upload each file individually
cd "$SCRIPT_DIR"
huggingface-cli upload $SPACE_NAME app.py app.py --repo-type=space
huggingface-cli upload $SPACE_NAME requirements.txt requirements.txt --repo-type=space
huggingface-cli upload $SPACE_NAME README.md README.md --repo-type=space

echo ""
echo "✅ Deployment complete!"
echo "🌐 Your Space is available at: https://huggingface.co/spaces/$SPACE_NAME"
echo ""
echo "⏳ Note: It may take 2-5 minutes for the Space to build and start running."
echo "📊 Check build status at: https://huggingface.co/spaces/$SPACE_NAME/logs"
