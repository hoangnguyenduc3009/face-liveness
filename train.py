import os
import argparse
import random

import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from src.utils.config import load_yaml_config, TrainConfig, ensure_dirs
from src.datasets.celeba_spoof import CelebaSpoofDataset
from src.models.vit_liveness import LivenessViT
from src.training.find_threshold import find_best_threshold

from src.utils.build_transforms import build_transforms
import wandb

def worker_init_fn(worker_id: int):
    """Picklable worker init for DataLoader.

    Uses torch.initial_seed() provided by the DataLoader/Generator to
    deterministically seed Python's random and NumPy for each worker.
    """
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def main(args):
    cfg = load_yaml_config(args.config)
    set_seed(cfg.get('seed', 42))

    work_dir = cfg.get('work_dir', '.')
    out_dir = os.path.join(work_dir, cfg.get('output_dir', 'checkpoints'))
    log_dir = os.path.join(work_dir, cfg.get('log_dir', 'logs'))
    ensure_dirs(out_dir, log_dir)
    # Create a timestamp for this run so saved checkpoints are unique per run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Device selection with MPS fallback (Apple Silicon) then CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    transforms = build_transforms(cfg)
    data_cfg = cfg['data']
    data_root = data_cfg.get('root')
    train_ds = CelebaSpoofDataset(
        root=data_root,
        index_file=data_cfg['train_list'],
        transform=transforms['train'],
    )
    val_ds = CelebaSpoofDataset(
        root=data_root,
        index_file=data_cfg['val_list'],
        transform=transforms['val'],
    )

    # Deterministic-ish worker seeding via torch.Generator and top-level worker_init_fn
    base_seed = cfg.get('seed', 42)
    g = torch.Generator()
    g.manual_seed(base_seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 8),
        pin_memory=(data_cfg.get('pin_memory', True) and device.type == 'cuda'),
        persistent_workers=data_cfg.get('persistent_workers', True) and data_cfg.get('num_workers', 8) > 0,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['eval']['batch_size'],
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 8),
        pin_memory=(data_cfg.get('pin_memory', True) and device.type == 'cuda'),
        persistent_workers=data_cfg.get('persistent_workers', True) and data_cfg.get('num_workers', 8) > 0,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    model_cfg = cfg['model']
    model = LivenessViT(
        img_size=model_cfg.get('image_size', 224),
        patch_size=model_cfg.get('patch_size', 16),
        d_model=model_cfg.get('d_model', 128),
        nhead=model_cfg.get('nhead', 4),
        num_layers=model_cfg.get('num_layers', 2),
        num_classes=model_cfg.get('num_classes', 2),
        dropout=model_cfg.get('drop_rate', 0.0),
    ).to(device)

    train_cfg = TrainConfig.from_dict(cfg['train'])

    if train_cfg.class_weights is not None:
        class_weights = torch.tensor(train_cfg.class_weights, dtype=torch.float32, device=device)
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
    )
    # Warmup + cosine decay
    total_epochs = cfg['train']['epochs']
    warmup_epochs = cfg['train'].get('warmup_epochs', 0)

    def lr_lambda(current_epoch: int):
        if current_epoch < warmup_epochs and warmup_epochs > 0:
            return float(current_epoch + 1) / float(warmup_epochs)
        # cosine over remaining epochs
        progress = (current_epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Use new AMP API; enable only on CUDA
    scaler = torch.amp.GradScaler('cuda', enabled=(train_cfg.amp and torch.cuda.is_available()))

    best_auc = -1.0
    best_threshold = 0.5
    # Use timestamped filenames so each run's checkpoints are unique
    best_path = os.path.join(out_dir, f'best_{timestamp}.pt')
    last_path = os.path.join(out_dir, f'last_{timestamp}.pt')

    # Initialize Weights & Biases only if a key is provided (explicit opt-in)
    wandb_run = None
    if (wandb is not None) and getattr(args, 'wandb_key', None):
        try:
            wandb.login(key=args.wandb_key)
            project_name = getattr(args, 'wandb_project', None) or cfg.get('project_name', 'face-liveness-transformer')
            # include timestamp in run name for easy identification
            run_name = f"{project_name}_{timestamp}"
            wandb_run = wandb.init(project=project_name, name=run_name, config=args.__dict__)
            wandb.watch(model, log='gradients', log_freq=100)
        except Exception as e:
            print(f"WandB init failed: {e}. Proceeding without WandB.")
            wandb_run = None

    try:
        for epoch in range(cfg['train']['epochs']):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [train]")
            all_preds, all_labels = [], []
            # track mean positive prob to see if model moves off class prior
            batch_pos_means = []
            running_loss = 0.0
            for images, labels in pbar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(train_cfg.amp and device.type == 'cuda')):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                if train_cfg.max_grad_norm and train_cfg.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * images.size(0)
                probs = torch.softmax(logits.detach(), dim=1)[:, 1]
                all_preds.extend(probs.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                batch_pos_means.append(probs.mean().item())

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}"
                })

            train_loss = running_loss / len(train_ds)
            try:
                train_auc = roc_auc_score(all_labels, all_preds)
                train_thresh, _ = find_best_threshold(all_labels, all_preds)
            except Exception:
                train_auc = float('nan')
                train_thresh = 0.5
            # Compute training accuracy from thresholded probabilities
            train_acc = accuracy_score(all_labels, (np.array(all_preds) >= train_thresh).astype(int))
            train_pos_mean = float(np.mean(batch_pos_means)) if batch_pos_means else float('nan')

            # Validation
            model.eval()
            val_preds, val_labels = [], []
            val_loss_accum = 0.0
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [val]"):
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    val_loss_accum += loss.item() * images.size(0)
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    val_preds.extend(probs.cpu().numpy().tolist())
                    val_labels.extend(labels.cpu().numpy().tolist())
            val_pos_mean = float(np.mean(val_preds)) if val_preds else float('nan')

            val_loss = val_loss_accum / len(val_ds)
            try:
                val_auc = roc_auc_score(val_labels, val_preds)
                best_threshold, _ = find_best_threshold(val_labels, val_preds)
            except Exception:
                val_auc = float('nan')
                best_threshold = 0.5
            print('val_preds_head:', val_preds[:16])
            val_acc = accuracy_score(val_labels, (np.array(val_preds) >= best_threshold).astype(int))

            # Capture the LR used during this epoch before stepping the scheduler
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            log_line = (
                f"Epoch {epoch+1}: lr={current_lr:.6f} "
                f"train_loss={train_loss:.4f} train_auc={train_auc:.4f} train_acc={train_acc:.4f} train_pos_mean={train_pos_mean:.4f} "
                f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_acc={val_acc:.4f} val_pos_mean={val_pos_mean:.4f}"
                f" best_thresh={best_threshold:.4f}"
            )
            print(log_line)
            if wandb_run is not None:
                wandb_run.log({
                    'epoch': epoch + 1,
                    'lr': current_lr,
                    'train/loss': train_loss,
                    'train/auc': train_auc,
                    'train/acc': train_acc,
                        'train/pos_mean': train_pos_mean,
                    'val/loss': val_loss,
                    'val/auc': val_auc,
                    'val/acc': val_acc,
                        'val/pos_mean': val_pos_mean,
                        'val/best_threshold': best_threshold,
                }, step=epoch + 1)

            # Save last checkpoint
            torch.save({'model': model.state_dict(), 'cfg': cfg}, last_path)
            if wandb_run is not None:
                try:
                    art_last = wandb.Artifact(name=f"model_last_{timestamp}", type="model")
                    art_last.add_file(last_path)
                    wandb_run.log_artifact(art_last)
                    # Save a lightweight reference in the run summary
                    wandb_run.summary['last_checkpoint'] = os.path.basename(last_path)
                except Exception as e:
                    print(f"Failed to log last checkpoint to WandB: {e}")

            # Save best by AUC
            if val_auc == val_auc and val_auc > best_auc:  # NaN-safe comparison
                best_auc = val_auc
                torch.save({'model': model.state_dict(), 'cfg': cfg, 'threshold': best_threshold}, best_path)
                print(f"Saved best checkpoint to {best_path} (AUC={best_auc:.4f})")
                if wandb_run is not None:
                    try:
                        art_best = wandb.Artifact(name=f"model_best_{timestamp}", type="model")
                        art_best.add_file(best_path)
                        # attach a little metadata about why this is best
                        art_best.metadata = {'best_auc': best_auc, 'epoch': epoch + 1, 'timestamp': timestamp, 'threshold': best_threshold}
                        wandb_run.log_artifact(art_best)
                        wandb_run.summary['best_checkpoint'] = os.path.basename(best_path)
                        wandb_run.summary['best_auc'] = best_auc
                        wandb_run.summary['best_epoch'] = epoch + 1
                        wandb_run.summary['best_threshold'] = best_threshold
                    except Exception as e:
                        print(f"Failed to log best checkpoint to WandB: {e}")
    finally:
        if wandb_run is not None:
            # Prefer global finish to ensure proper closure even if run object changes
            try:
                wandb.finish()
            except Exception:
                try:
                    wandb_run.finish()
                except Exception:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--wandb_key', type=str, default=None, help='WandB API key for login. If not provided, WandB is disabled for this run.')
    parser.add_argument('--wandb_project', type=str, default=None, help='Optional WandB project name override. Defaults to config project_name.')
    args = parser.parse_args()
    main(args)
