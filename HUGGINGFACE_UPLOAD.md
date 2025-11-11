# Uploading Face Liveness Model to Hugging Face Hub

This guide walks you through uploading your trained face liveness detection model to Hugging Face Hub.

## Prerequisites

1. **Trained Model Checkpoint**: You should have a trained model checkpoint (e.g., `checkpoints/best.pt`)
2. **Hugging Face Account**: Create an account at [huggingface.co](https://huggingface.co)
3. **API Token**: Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Make sure to create a token with **write** access

## Installation

The required library `huggingface_hub` is already installed. If needed, install it with:

```bash
pip install huggingface_hub
```

## Method 1: Using the Helper Script (Recommended)

The easiest way to upload your model:

```bash
./scripts/upload_helper.sh checkpoints/best.pt
```

The script will:
1. Prompt you for your Hugging Face repository name (e.g., `your-username/face-liveness-vit`)
2. Ask for your Hugging Face API token
3. Confirm the upload details
4. Upload the model, config, README, and inference script

## Method 2: Direct Python Script

For more control, use the Python script directly:

```bash
python scripts/upload_to_huggingface.py \
    --checkpoint checkpoints/best.pt \
    --repo_name your-username/face-liveness-vit \
    --token YOUR_HF_TOKEN
```

### Optional Arguments

- `--private`: Make the repository private
- `--commit_message "Custom message"`: Set a custom commit message

Example with all options:

```bash
python scripts/upload_to_huggingface.py \
    --checkpoint checkpoints/best_20251111_011345.pt \
    --repo_name hoangnguyenduc3009/face-liveness-vit \
    --token hf_xxxxxxxxxxxxxxxxxxxxx \
    --private \
    --commit_message "Upload improved model v2"
```

## What Gets Uploaded

The script uploads the following files to your Hugging Face repository:

1. **model.pt** - Your trained model checkpoint including:
   - Model weights
   - Configuration
   - Optimal threshold for classification

2. **config.yaml** - Training configuration including:
   - Model architecture parameters
   - Training hyperparameters
   - Data augmentation settings

3. **README.md** - Comprehensive model card with:
   - Model description
   - Architecture details
   - Training information
   - Usage examples
   - Performance metrics
   - Citations

4. **inference.py** - Standalone inference script that includes:
   - Model class definitions
   - Loading functions
   - Prediction utilities

## Testing the Uploaded Model

After uploading, test your model:

```bash
python scripts/test_huggingface_model.py \
    --repo your-username/face-liveness-vit \
    --image path/to/test/image.jpg
```

## Using the Model from Hugging Face

### Option 1: Using the Test Script

```bash
# Download and test
python scripts/test_huggingface_model.py \
    --repo hoangnguyenduc3009/face-liveness-vit \
    --image data/CelebA_Spoof/Data/test/10001/1.jpg
```

### Option 2: In Your Own Code

```python
from huggingface_hub import hf_hub_download
import torch
from PIL import Image
import torchvision.transforms as transforms

# Download model
repo_id = "your-username/face-liveness-vit"
model_path = hf_hub_download(repo_id=repo_id, filename="model.pt")

# Load checkpoint
checkpoint = torch.load(model_path, map_location='cpu')

# Initialize model (you'll need the model class)
# See the inference.py file in the repo for complete code
```

### Option 3: Clone the Inference Script

```bash
# Download the standalone inference script
wget https://huggingface.co/your-username/face-liveness-vit/resolve/main/inference.py

# Use it directly
python inference.py path/to/image.jpg model.pt
```

## Repository Structure on Hugging Face

After upload, your repository will look like:

```
your-username/face-liveness-vit/
├── README.md              # Model card
├── model.pt               # Model checkpoint
├── config.yaml            # Configuration
└── inference.py           # Standalone inference script
```

## Troubleshooting

### Authentication Error

If you get an authentication error:
1. Make sure your token has **write** access
2. Try logging in manually first: `huggingface-cli login`

### Repository Already Exists

The script will use `exist_ok=True`, so it won't fail if the repo exists. It will just upload/update the files.

### Private vs Public

By default, repositories are **public**. To make it private:
```bash
python scripts/upload_to_huggingface.py \
    --checkpoint checkpoints/best.pt \
    --repo_name your-username/face-liveness-vit \
    --token YOUR_TOKEN \
    --private
```

### Large File Warning

Model files are typically 2-3 MB for this architecture, which is well within Hugging Face's limits. For larger models, Git LFS is automatically handled.

## Available Checkpoints

Check your available checkpoints:

```bash
ls -lh checkpoints/
```

Current checkpoints:
- `best.pt` - Best model by AUC (older)
- `best_20251111_010316.pt` - Best model from training run 1
- `best_20251111_011345.pt` - Best model from training run 2 (most recent)
- `last.pt` - Last checkpoint (older)
- `last_20251111_010316.pt` - Last checkpoint from run 1
- `last_20251111_011345.pt` - Last checkpoint from run 2 (most recent)

**Recommendation**: Use the most recent `best_*.pt` checkpoint for best performance.

## Next Steps

After uploading:

1. **View your model**: Visit `https://huggingface.co/your-username/face-liveness-vit`
2. **Edit the README**: Add more details about your specific training
3. **Add examples**: Upload sample images to the repo
4. **Create a Space**: Build a Gradio/Streamlit demo
5. **Share**: Share your model with the community!

## Example: Complete Upload Flow

```bash
# 1. Choose your best checkpoint
ls -lh checkpoints/

# 2. Run the upload helper
./scripts/upload_helper.sh checkpoints/best_20251111_011345.pt

# Enter when prompted:
# Repo name: hoangnguyenduc3009/face-liveness-vit
# Token: hf_xxxxxxxxxxxxxxxxxxxxx

# 3. Wait for upload to complete

# 4. Visit your model page
# https://huggingface.co/hoangnguyenduc3009/face-liveness-vit
```

## Deploying to Hugging Face Spaces

Deploy your interactive Gradio demo in one command:

```bash
./huggingface_spaces/deploy.sh
```

Your Space will be available at: `https://huggingface.co/spaces/hoangnguyenduc3009/face-liveness-vit`

**Note**: It may take 2-5 minutes for the Space to build and become active.

## Additional Resources

- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub/index)
- [Model Cards Guide](https://huggingface.co/docs/hub/model-cards)
- [Uploading Models](https://huggingface.co/docs/hub/models-uploading)
- [Hugging Face Spaces Guide](https://huggingface.co/docs/hub/spaces-overview)

## Support

If you encounter any issues:
1. Check the error messages carefully
2. Verify your token has write access
3. Ensure the checkpoint file exists
4. Try with a different repository name
