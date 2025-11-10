import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from src.datasets.celeba_spoof import CelebaSpoofDataset
from src.models.vit_liveness import LivenessViT
from src.utils.config import load_yaml_config

from utils.build_transforms import build_transforms

def main(args):
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg = ckpt.get('cfg') if 'cfg' in ckpt else load_yaml_config(args.config)
    model = LivenessViT(
    )
    model.load_state_dict(ckpt['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    val_tfm = build_transforms(cfg)['val']
    data_cfg = cfg['data']
    test_list = data_cfg.get('test_list') or data_cfg['val_list']
    data_root = data_cfg.get('root')
    dataset = CelebaSpoofDataset(
        root=data_root,
        index_file=test_list,
        transform=val_tfm,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg['eval']['batch_size'],
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 8)
    )

    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds.extend(probs.cpu().numpy().tolist())
            labels.extend(lbls.cpu().numpy().tolist())

    preds_np = np.array(preds)
    labels_np = np.array(labels)
    auc = roc_auc_score(labels_np, preds_np)
    acc = accuracy_score(labels_np, (preds_np >= 0.5).astype(int))
    cm = confusion_matrix(labels_np, (preds_np >= 0.5).astype(int))
    print(f"AUC: {auc:.4f}  ACC: {acc:.4f}\nConfusion Matrix:\n{cm}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=False,
                        help='Model checkpoint path. Not required if only visualizing.')
    # Visualization aids
    parser.add_argument('--visualize_image', type=str, default='',
                        help='Path to an image to visualize resize/crop steps. If set, tool will save previews and exit.')
    parser.add_argument('--vis_size', type=int, default=224, help='Target image size for visualization')
    parser.add_argument('--vis_mode', type=str, default='center_crop', choices=['center_crop', 'letterbox'],
                        help='Visualization mode: center_crop (Resize+CenterCrop) or letterbox (resize longest side + pad)')
    parser.add_argument('--vis_outdir', type=str, default='debug_vis', help='Output directory for visualization images')
    args = parser.parse_args()
    main(args)
