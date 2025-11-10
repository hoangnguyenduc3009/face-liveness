from typing import Any, Callable, Dict, List, Optional, Tuple
import os
from PIL import Image
import torch
from torch.utils.data import Dataset


def parse_index_file(index_file: str) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with open(index_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # supports space or comma separated: path label
            if ',' in line:
                path, label = line.split(',', 1)
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                path, label = parts[0], parts[1]
            items.append((path, int(label)))
    return items




class CelebaSpoofDataset(Dataset):
    """
    CelebA-Spoof dataset reader with multiple input modes.

    Supported sources:
    - index_file: a plain text file with lines "path label" or "path,label" (backward compatible)
 
    Parameters
    - root: Optional[str] — dataset root. If provided and paths in index/json are relative, joined with root
    - index_file: Optional[str] — path to simple list file. Use when not providing label_json
    - label_index: int — which index in the label array to use as the primary target (default 43 = live/spoof)
    - transform: Optional[Callable] — image transform
    - target_transform: Optional[Callable] — transform for the primary label
    - return_meta: bool — if True, also return the full label array (or a dict) as metadata

    __getitem__ returns:
    - (image, label) by default
    - (image, label, meta) if return_meta=True where meta contains {"labels": full_label_list, "path": path}
    """

    def __init__(
        self,
        root: Optional[str] = None,
        index_file: Optional[str] = None,
        *,
        label_index: int = 43,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_meta: bool = False,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.return_meta = return_meta
        self.label_index = label_index

        if index_file:
            pairs2 = parse_index_file(index_file)
            self.items = [(p, int(lbl), None) for (p, lbl) in pairs2]
        else:
            raise ValueError("Either index_file or label_json must be provided.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label, labels_full = self.items[idx]
        img_path = os.path.join(self.root, path) if self.root and not os.path.isabs(path) else path

        with Image.open(img_path) as im:
            im = im.convert('RGB')

        if self.transform is not None:
            im = self.transform(im)

        y: torch.Tensor = torch.tensor(label, dtype=torch.long)
        if self.target_transform is not None:
            y = self.target_transform(y)

        if self.return_meta:
            meta: Dict[str, Any] = {"path": path}
            if labels_full is not None:
                meta["labels"] = labels_full
            return im, y, meta
        return im, y
    def __sample__(self):
        return self.__getitem__(0)