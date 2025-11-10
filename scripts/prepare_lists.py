#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import random


def is_image(p: Path) -> bool:
    return p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


def infer_label_from_path(path: Path):
    low = str(path).lower()
    if 'spoof' in low or 'fake' in low or 'attack' in low:
        return 0
    if 'live' in low or 'real' in low:
        return 1
    return None


def main(args):
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_items = []
    for p in root.rglob('*'):
        if p.is_file() and is_image(p):
            label = infer_label_from_path(p)
            if label is None:
                continue
            rel = os.path.relpath(p, start=root) if args.relative else str(p)
            all_items.append((rel, label))

    if not all_items:
        print('No images found or could not infer labels. Ensure folder names contain live/spoof keywords.')
        return

    random.seed(args.seed)
    random.shuffle(all_items)
    n = len(all_items)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train = all_items[:n_train]
    val = all_items[n_train:n_train + n_val]
    test = all_items[n_train + n_val:]

    def write_list(path: Path, items):
        with open(path, 'w') as f:
            for p, y in items:
                f.write(f"{p} {y}\n")

    write_list(out_dir / 'train.txt', train)
    write_list(out_dir / 'val.txt', val)
    write_list(out_dir / 'test.txt', test)
    print(f"Wrote {len(train)} train, {len(val)} val, {len(test)} test to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare simple list files from a folder tree with live/spoof names.')
    parser.add_argument('--root', type=str, required=True, help='Root directory of images')
    parser.add_argument('--out-dir', type=str, default='data/lists')
    parser.add_argument('--relative', action='store_true', help='Write relative paths (relative to root)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
