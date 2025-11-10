import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    betas: tuple
    warmup_epochs: int
    max_grad_norm: float
    amp: bool
    class_weights: Optional[list]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'TrainConfig':
        return TrainConfig(
            batch_size=d.get('batch_size', 64),
            epochs=d.get('epochs', 20),
            lr=d.get('lr', 3e-4),
            weight_decay=d.get('weight_decay', 0.05),
            betas=tuple(d.get('betas', (0.9, 0.999))),
            warmup_epochs=d.get('warmup_epochs', 2),
            max_grad_norm=d.get('max_grad_norm', 1.0),
            amp=d.get('amp', True),
            class_weights=d.get('class_weights')
        )


def ensure_dirs(*paths: str):
    for p in paths:
        if p and not os.path.exists(p):
            os.makedirs(p, exist_ok=True)
