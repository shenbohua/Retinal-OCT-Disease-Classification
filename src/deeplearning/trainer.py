from __future__ import annotations

"""Reusable trainer components for manifest-driven DL experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.config import CLASS_NAMES
from src.evaluate import compute_metrics

from .augmentations import build_transforms
from .models import build_model

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment,misc]
    Dataset = object  # type: ignore[assignment,misc]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass
class DLRunConfig:
    model_name: str = "resnet18"
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    image_size: int = 224
    num_workers: int = 0
    eval_split: str = "val_final"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    checkpoint_dir: Path = Path("outputs/models")
    random_seed: int = 42
    use_class_weight: bool = True
    device: str = "auto"
    pretrained: bool = True
    freeze_backbone: bool = False
    data_mode: str = "raw"
    processed_profile: str = "oct224_gray_png_v2"
    final_test: bool = False


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch stack is required for DL commands. Install `torch` and `torchvision` first."
        ) from _TORCH_IMPORT_ERROR


class OCTManifestDataset(Dataset):
    """Dataset backed by resolved file paths and class names from split manifest."""

    def __init__(
        self,
        filepaths: list[str],
        class_names: list[str],
        image_size: int,
        augment: bool,
    ) -> None:
        _require_torch()
        self.filepaths = filepaths
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        self.transform = build_transforms(image_size=image_size, augment=augment)

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> tuple[Any, int, str]:
        path = self.filepaths[idx]
        class_name = self.class_names[idx]
        label = self.class_to_idx[class_name]

        with Image.open(path) as img:
            x = self.transform(img)
        return x, label, path


class DLTrainer:
    """Framework: device resolution, model creation, loops, and checkpointing."""

    def __init__(self, config: DLRunConfig) -> None:
        _require_torch()
        self.config = config
        self.device = self._resolve_device(config.device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        _require_torch()
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def make_model(self) -> Any:
        _require_torch()
        model = build_model(
            model_name=self.config.model_name,
            num_classes=len(CLASS_NAMES),
            pretrained=self.config.pretrained,
            freeze_backbone=self.config.freeze_backbone,
        )
        return model.to(self.device)

    def make_optimizer(self, model: Any) -> Any:
        _require_torch()
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def make_criterion(self, class_weights: np.ndarray | None = None) -> Any:
        _require_torch()
        if class_weights is None:
            return nn.CrossEntropyLoss()
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        return nn.CrossEntropyLoss(weight=weight_tensor)

    @staticmethod
    def class_weight_from_labels(y_indices: np.ndarray, num_classes: int) -> np.ndarray:
        counts = np.bincount(y_indices, minlength=num_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        inv = counts.sum() / (num_classes * counts)
        return inv.astype(np.float32)

    @staticmethod
    def make_loader(
        filepaths: list[str],
        class_names: list[str],
        image_size: int,
        batch_size: int,
        num_workers: int,
        augment: bool,
        shuffle: bool,
    ) -> Any:
        _require_torch()
        dataset = OCTManifestDataset(
            filepaths=filepaths,
            class_names=class_names,
            image_size=image_size,
            augment=augment,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )

    @staticmethod
    def _metrics_from_indices(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
        true_names = np.array([CLASS_NAMES[i] for i in y_true])
        pred_names = np.array([CLASS_NAMES[i] for i in y_pred])
        return compute_metrics(true_names, pred_names)

    def train_one_epoch(self, model: Any, loader: Any, optimizer: Any, criterion: Any) -> dict[str, float]:
        _require_torch()
        model.train()
        losses: list[float] = []
        y_true: list[int] = []
        y_pred: list[int] = []

        for x, y, _paths in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))
            y_true.extend(y.detach().cpu().numpy().tolist())
            y_pred.extend(logits.detach().argmax(dim=1).cpu().numpy().tolist())

        metrics = self._metrics_from_indices(y_true, y_pred)
        metrics["loss"] = float(np.mean(losses)) if losses else 0.0
        return metrics

    def validate(self, model: Any, loader: Any, criterion: Any) -> dict[str, float]:
        _require_torch()
        model.eval()
        losses: list[float] = []
        y_true: list[int] = []
        y_pred: list[int] = []

        with torch.no_grad():
            for x, y, _paths in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = model(x)
                loss = criterion(logits, y)

                losses.append(float(loss.item()))
                y_true.extend(y.detach().cpu().numpy().tolist())
                y_pred.extend(logits.detach().argmax(dim=1).cpu().numpy().tolist())

        metrics = self._metrics_from_indices(y_true, y_pred)
        metrics["loss"] = float(np.mean(losses)) if losses else 0.0
        return metrics

    def predict(self, model: Any, loader: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Return true/pred labels, probabilities, and file paths."""
        _require_torch()
        model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []
        y_prob: list[np.ndarray] = []
        all_paths: list[str] = []

        with torch.no_grad():
            for x, y, paths in loader:
                x = x.to(self.device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)

                y_true.extend(y.detach().cpu().numpy().tolist())
                y_pred.extend(pred.detach().cpu().numpy().tolist())
                y_prob.extend(probs.detach().cpu().numpy())
                all_paths.extend(list(paths))

        return np.array(y_true), np.array(y_pred), np.array(y_prob), all_paths

    def save_checkpoint(
        self,
        state: dict[str, Any],
        filename: str | None = None,
        path: Path | None = None,
    ) -> Path:
        _require_torch()
        if path is None:
            if not filename:
                raise ValueError("Either `filename` or `path` must be provided.")
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = self.config.checkpoint_dir / filename
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        return path
