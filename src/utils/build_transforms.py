from typing import Dict
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os
from PIL import Image
import torch
import random

def build_transforms(cfg: Dict) -> Dict[str, T.Compose]:
    """Create training/validation transforms honoring optional augment config.

    Adds optional Gaussian blur (common for ViT regularization) controlled by
    augment.gaussian_blur_p.
    """
    size = cfg['model']['image_size']
    aug = cfg.get('augment', {})

    gaussian_blur_p = aug.get('gaussian_blur_p', 0.0)
    blur_kernel = int(round(size * 0.05)) // 2 * 2 + 1  # ensure odd
    blur_tf = T.RandomApply([T.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0))], p=gaussian_blur_p) if gaussian_blur_p > 0 else None

    train_list = [
        T.RandomResizedCrop(size, scale=tuple(aug.get('random_resized_crop_scale', [0.8, 1.0]))),
        T.RandomHorizontalFlip(p=aug.get('horizontal_flip_p', 0.5)),
        T.ColorJitter(*aug.get('color_jitter', [0.2, 0.2, 0.2, 0.1])),
        T.RandomGrayscale(p=aug.get('random_grayscale_p', 0.05)),
    ]
    if blur_tf:
        train_list.append(blur_tf)
    train_list += [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    train_tfms = T.Compose(train_list)
    val_tfms = T.Compose([
        T.Resize(int(size * 1.14)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return {'train': train_tfms, 'val': val_tfms}


def visualize_transform_steps(
    cfg: Dict,
    image_path: str,
    split: str = 'val',
    out_dir: str = 'logs',
    seed: int = 0,
    save_tensor: bool = False,
) -> str:
    """Visualize sequential application of build_transforms pipelines.

    Args:
        cfg: Full config dict used for build_transforms (needs model.image_size + augment).
        image_path: Input image path.
        split: 'train' or 'val'.
        out_dir: Directory to dump intermediate PNGs.
        seed: RNG seed for reproducible random augment preview.
        save_tensor: If True, also saves final normalized tensor converted back to PIL.

    Returns:
        Absolute path to output directory.
    """
    assert os.path.isfile(image_path), f"Image not found: {image_path}"
    assert split in ('train', 'val'), "split must be 'train' or 'val'"

    os.makedirs(out_dir, exist_ok=True)

    # Deterministic random transforms for preview
    torch.manual_seed(seed)
    random.seed(seed)

    pipelines = build_transforms(cfg)
    tfms = pipelines[split]

    img = Image.open(image_path).convert('RGB')
    img.save(os.path.join(out_dir, '0_original.png'))

    current = img
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    for idx, t in enumerate(getattr(tfms, 'transforms', [tfms])):
        name = t.__class__.__name__
        # Apply transform and save if possible
        if isinstance(t, T.ToTensor):
            current = t(current)
            if save_tensor:
                pil_un = TF.to_pil_image(current)
                pil_un.save(os.path.join(out_dir, f"{idx+1}_{name.lower()}.png"))
            continue
        elif isinstance(t, T.Normalize):
            # current is expected to be tensor here
            current = t(current)
            if save_tensor:
                unnorm = current.clone()
                unnorm = unnorm * std[:, None, None] + mean[:, None, None]
                unnorm = torch.clamp(unnorm, 0, 1)
                pil_final = TF.to_pil_image(unnorm)
                pil_final.save(os.path.join(out_dir, f"{idx+1}_{name.lower()}_unnorm.png"))
            continue
        else:
            # PIL -> PIL transforms
            current = t(current)
            if isinstance(current, Image.Image):
                current.save(os.path.join(out_dir, f"{idx+1}_{name.lower()}.png"))
            else:
                # If unexpectedly tensor, convert back just for visualization
                pil_tmp = TF.to_pil_image(current)
                pil_tmp.save(os.path.join(out_dir, f"{idx+1}_{name.lower()}.png"))

    return os.path.abspath(out_dir)


# cfg = {
#     'model': {'image_size': 224},
#     'augment': {
#         'random_resized_crop_scale': [0.8, 1.0],
#         'horizontal_flip_p': 0.5,
#         'color_jitter': [0.2, 0.2, 0.2, 0.1],
#         'random_grayscale_p': 0.05,
#         'gaussian_blur_p': 0.2
#     }
# }
# visualize_transform_steps(cfg=cfg, image_path='spoof_sample.png', split='train', out_dir='logs', seed=0, save_tensor=False)