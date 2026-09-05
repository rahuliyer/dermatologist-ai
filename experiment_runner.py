"""Training and evaluation utilities for the lesion classifiers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


class BinaryImageFolder(datasets.ImageFolder):
    """ImageFolder that maps one named class to 1 and every other class to 0."""

    def __init__(self, root: str | Path, positive_class: str, transform=None):
        super().__init__(root=root, transform=transform)
        if positive_class not in self.class_to_idx:
            available = ", ".join(sorted(self.class_to_idx))
            raise ValueError(
                f"Class {positive_class!r} was not found in {root}. "
                f"Available classes: {available}"
            )
        self.positive_idx = self.class_to_idx[positive_class]
        self.binary_targets = [
            int(target == self.positive_idx) for target in self.targets
        ]

    def __getitem__(self, index: int):
        image, _ = super().__getitem__(index)
        return image, self.binary_targets[index]


class BinaryImageFolderWithPaths(BinaryImageFolder):
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        return self.samples[index][0], image, target


class ExperimentRunner:
    """Stream image batches from disk and train one binary classifier."""

    def __init__(
        self,
        loss_fn: nn.Module | None,
        train_dataset_root: str | Path | None,
        test_dataset_root: str | Path,
        train_transforms: Sequence | None,
        test_transforms: Sequence,
        *,
        positive_class: str,
        batch_size: int = 64,
        workers: int = 4,
        device: torch.device | None = None,
        seed: int = 42,
        amp: bool = True,
    ):
        self.loss_fn = loss_fn
        self.train_dataset_root = (
            Path(train_dataset_root) if train_dataset_root else None
        )
        self.test_dataset_root = Path(test_dataset_root)
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.positive_class = positive_class
        self.batch_size = batch_size
        self.workers = workers
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seed = seed
        self.amp = amp and self.device.type == "cuda"

        self.train_loader: DataLoader | None = None
        self.valid_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.best_valid_loss: float | None = None

    def _loader_options(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "num_workers": self.workers,
            "pin_memory": self.device.type == "cuda",
            "persistent_workers": self.workers > 0,
        }

    def _dataset(self, split: str, transform, *, with_paths: bool = False):
        root = self.test_dataset_root if split == "test" else self.train_dataset_root
        if root is None:
            raise ValueError(
                "A training dataset root is required for train/valid loaders"
            )
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Dataset split does not exist: {split_dir}")
        dataset_type = BinaryImageFolderWithPaths if with_paths else BinaryImageFolder
        return dataset_type(
            split_dir,
            positive_class=self.positive_class,
            transform=transforms.Compose(transform),
        )

    def get_train_loader(self) -> DataLoader:
        dataset = self._dataset("train", self.train_transforms)
        counts = np.bincount(dataset.binary_targets, minlength=2)
        if np.any(counts == 0):
            raise ValueError(
                f"Training split needs both binary classes; found counts {counts.tolist()}"
            )
        sample_weights = torch.tensor(
            [1.0 / counts[target] for target in dataset.binary_targets],
            dtype=torch.double,
        )
        generator = torch.Generator().manual_seed(self.seed)
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        return DataLoader(dataset, sampler=sampler, **self._loader_options())

    def get_valid_loader(self) -> DataLoader:
        dataset = self._dataset("valid", self.test_transforms)
        return DataLoader(dataset, shuffle=False, **self._loader_options())

    def get_test_loader(self) -> DataLoader:
        dataset = self._dataset("test", self.test_transforms, with_paths=True)
        return DataLoader(dataset, shuffle=False, **self._loader_options())

    def _average_loss(self, model: nn.Module, loader: DataLoader) -> float:
        if self.loss_fn is None:
            raise ValueError("A loss function is required for training")
        total_loss = 0.0
        total_items = 0
        model.eval()
        with torch.inference_mode():
            for inputs, labels in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.float32, non_blocking=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp,
                ):
                    logits = model(inputs).squeeze(1)
                    loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * inputs.size(0)
                total_items += inputs.size(0)
        return total_loss / total_items

    def train(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        *,
        patience: int = 5,
    ) -> nn.Module:
        if self.loss_fn is None:
            raise ValueError("A loss function is required for training")
        self.train_loader = self.train_loader or self.get_train_loader()
        self.valid_loader = self.valid_loader or self.get_valid_loader()
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        best_valid_loss = float("inf")
        best_state = None
        stale_epochs = 0

        print(
            f"Training on {self.device} with {len(self.train_loader.dataset)} train and "
            f"{len(self.valid_loader.dataset)} validation images"
        )
        for epoch in range(1, num_epochs + 1):
            model.train()
            total_loss = 0.0
            total_items = 0
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.float32, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp,
                ):
                    logits = model(inputs).squeeze(1)
                    loss = self.loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item() * inputs.size(0)
                total_items += inputs.size(0)

            train_loss = total_loss / total_items
            valid_loss = self._average_loss(model, self.valid_loader)
            print(
                f"Epoch {epoch:03d}/{num_epochs:03d} - "
                f"train_loss={train_loss:.5f} valid_loss={valid_loss:.5f}"
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if patience and stale_epochs >= patience:
                    print(f"Early stopping after {epoch} epochs")
                    break

        if best_state is None:
            raise RuntimeError(
                "Training completed without producing a model checkpoint"
            )
        model.load_state_dict(best_state)
        self.best_valid_loss = best_valid_loss
        return model

    def test(self, model: nn.Module):
        model.eval()
        self.test_loader = self.test_loader or self.get_test_loader()
        paths: list[str] = []
        labels: list[int] = []
        probabilities: list[float] = []

        with torch.inference_mode():
            for batch_paths, inputs, targets in self.test_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp,
                ):
                    logits = model(inputs).squeeze(1)
                    probs = torch.sigmoid(logits)
                paths.extend(batch_paths)
                probabilities.extend(probs.float().cpu().tolist())
                labels.extend(targets.tolist())

        return paths, labels, probabilities

    def accuracy(self, model: nn.Module) -> float:
        _, labels, predictions = self.test(model)
        return accuracy_score(labels, np.round(predictions))

    def confusion_matrix(self, model: nn.Module):
        _, labels, predictions = self.test(model)
        return confusion_matrix(labels, np.round(predictions))

    def recall_score(self, model: nn.Module) -> float:
        _, labels, predictions = self.test(model)
        return recall_score(labels, np.round(predictions))
