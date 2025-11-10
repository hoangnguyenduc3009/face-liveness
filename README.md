# Face Liveness Detection with Vision Transformer

Project scaffold for training a Vision Transformer (ViT) model to detect live vs spoof faces using CelebA-Spoof style list files.

## Features
- Vision Transformer backbone via `timm`
- Config-driven training (`configs/default.yaml`)
- Mixed precision (AMP)
- Cosine LR schedule
- AUC / Accuracy metrics, best checkpoint saving
- Simple dataset list format: `image_path label`

## Directory Structure
```
configs/
  default.yaml
src/
  datasets/celeba_spoof.py
  models/vit_liveness.py
  utils/config.py
train.py
evaluate.py
scripts/download_celeba_spoof.sh
tests/test_model_forward.py
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # (macOS/Linux)
pip install -r requirements.txt
```

## Dataset Preparation
1. Request / download CelebA-Spoof from the official site (academic use only).
2. Extract under `data/CelebA-Spoof/` (or any path you prefer).
3. Create list files (train/val/test) each line:
   ```
   relative/or/absolute/path/to/image.jpg  LABEL
   ```
   Where `LABEL` is `1` for live and `0` for spoof (stay consistent).
4. Place them at:
   - `data/lists/train.txt`
   - `data/lists/val.txt`
   - `data/lists/test.txt` (optional; will fallback to val if missing)

Tip: Ensure total counts and label balance; you can stratify split with a small Python script.

### Automated Download Helper (Optional)
If you have the Google Drive folder with split archives `CelebA_Spoof.zip.001 ...`, you can use the helper script:

```bash
bash scripts/download_celeba_spoof.sh               # downloads into ./data
bash scripts/download_celeba_spoof.sh /custom/path  # alternative root
```

The script will:
- Download all parts using `gdown` (auto-installs if missing)
- Merge them into `CelebA_Spoof.zip`
- Extract to `data/CelebA-Spoof/`
- Leave a marker file `.extracted_ok` for idempotent re-runs

Afterwards, generate list files (train/val/test) with your preferred splitting logic and place under `data/lists/`.

Licensing: Ensure you have academic permission to use CelebA-Spoof. Do not redistribute the raw data.

## Training
```bash
python train.py --config configs/default.yaml
```
Logs & checkpoints will appear in `logs/` and `checkpoints/` (auto-created).

## Evaluation
```bash
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

Output: AUC, Accuracy, Confusion Matrix.

## Customizing
- Change model size: set `model.name` to e.g. `vit_small_patch16_224` (see `timm` docs).
- Freeze backbone: set `freeze_backbone: true` in config for linear probing.
- Class imbalance: set `train.class_weights`, e.g. `[1.0, 2.0]`.

## Testing
Quick forward test:
```bash
python tests/test_model_forward.py
```

## Roadmap / Next Steps
- Add logging to TensorBoard or WandB
- Add early stopping & gradient accumulation
- Add multi-scale / frequency domain spoof cues

## License
This scaffold contains no dataset and respects CelebA-Spoof licensing (manual download required).

---
Happy modeling!