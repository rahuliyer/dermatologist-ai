import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets

from sklearn.metrics import roc_auc_score


try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def normalize_image_id(path):
    path = str(path).replace("\\", "/")
    marker = "data/test/"
    if marker in path:
        return path[path.index(marker):]
    return os.path.relpath(path) if not os.path.isabs(path) else path


class BinaryLesionDataset(Dataset):
    """ImageFolder wrapper that maps a multi-class dermoscopy set onto one binary task."""

    def __init__(self, root, transform, positive_classes, return_paths=False):
        if not os.path.isdir(root):
            raise FileNotFoundError("Dataset directory not found: {}".format(root))

        self.folder = datasets.ImageFolder(root)
        self.transform = transform
        self.return_paths = return_paths

        available = self.folder.class_to_idx
        pos_idxs = {available[name] for name in positive_classes if name in available}
        if not pos_idxs:
            raise ValueError(
                "None of the positive classes {} were found in {}. Available: {}".format(
                    sorted(positive_classes), root, sorted(available)
                )
            )

        self.targets = [1 if y in pos_idxs else 0 for _, y in self.folder.samples]

    def __len__(self):
        return len(self.folder.samples)

    def __getitem__(self, index):
        path, _ = self.folder.samples[index]
        image = self.folder.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        label = self.targets[index]
        if self.return_paths:
            return path, image, label
        return image, label


def _tta_views(inputs):
    return (
        inputs,
        torch.flip(inputs, dims=[-1]),
        torch.flip(inputs, dims=[-2]),
        torch.flip(inputs, dims=[-1, -2]),
        torch.rot90(inputs, 1, dims=[-2, -1]),
        torch.rot90(inputs, 3, dims=[-2, -1]),
    )


class ExperimentRunner(object):
    def __init__(
        self,
        loss_fn,
        train_dir,
        valid_dir,
        test_dir,
        train_transforms,
        eval_transforms,
        positive_classes,
        batch_size=32,
        num_workers=4,
        device=None,
        use_tta=True,
    ):
        self.loss_fn = loss_fn
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.train_transforms = train_transforms
        self.eval_transforms = eval_transforms
        self.positive_classes = set(positive_classes)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_tta = use_tta
        self.use_amp = self.device.type == "cuda"
        self.last_best_auc = float("-inf")

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def _loader_kwargs(self):
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.device.type == "cuda",
        }

    def get_dataset(self, root, transform, return_paths=False):
        return BinaryLesionDataset(
            root,
            transform=transform,
            positive_classes=self.positive_classes,
            return_paths=return_paths,
        )

    def get_train_loader(self):
        dataset = self.get_dataset(self.train_dir, self.train_transforms)
        targets = np.asarray(dataset.targets, dtype=np.int64)
        counts = np.bincount(targets, minlength=2).astype(np.float64)
        print(
            "Train samples: {} (neg={}, pos={})".format(
                len(dataset), int(counts[0]), int(counts[1])
            )
        )

        sampler = None
        shuffle = True
        if counts.min() > 0:
            weights = torch.as_tensor((1.0 / counts)[targets], dtype=torch.double)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            shuffle = False

        return DataLoader(
            dataset,
            sampler=sampler,
            shuffle=shuffle,
            drop_last=False,
            **self._loader_kwargs()
        )

    def get_eval_loader(self, root, return_paths=False):
        dataset = self.get_dataset(root, self.eval_transforms, return_paths=return_paths)
        return DataLoader(dataset, shuffle=False, **self._loader_kwargs())

    def _forward_logits(self, model, inputs):
        outputs = model(inputs)
        if outputs.dim() > 1:
            outputs = outputs.squeeze(-1)
        return outputs

    def train(self, model, optimizer, num_epochs, scheduler=None, patience=10, print_every=1):
        if self.train_loader is None:
            print("Setting up train dataloader")
            self.train_loader = self.get_train_loader()
        if self.valid_loader is None:
            print("Setting up valid dataloader")
            self.valid_loader = self.get_eval_loader(self.valid_dir)

        scaler = None
        if self.use_amp:
            try:
                from torch.cuda.amp import GradScaler
                scaler = GradScaler()
            except ImportError:
                scaler = None
                self.use_amp = False

        best_auc = float("-inf")
        best_state = None
        epochs_without_improvement = 0

        print("Starting training on {} for {} epochs...".format(self.device, num_epochs))
        val_loss, val_auc, _, _ = self.evaluate(model, self.valid_loader)
        if val_auc == val_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print("Initial val ROC-AUC: {:.4f}".format(best_auc))

        for epoch_nr in range(num_epochs):
            model.train()
            running_loss = 0.0
            seen = 0

            for inputs, labels in tqdm(self.train_loader, desc="Epoch {}".format(epoch_nr), leave=False):
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).float()

                optimizer.zero_grad()
                if self.use_amp and scaler is not None:
                    from torch.cuda.amp import autocast
                    with autocast():
                        logits = self._forward_logits(model, inputs)
                        loss = self.loss_fn(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = self._forward_logits(model, inputs)
                    loss = self.loss_fn(logits, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                seen += inputs.size(0)

            if scheduler is not None:
                scheduler.step()

            train_loss = running_loss / max(seen, 1)
            val_loss, val_auc, _, _ = self.evaluate(model, self.valid_loader)

            lr = optimizer.param_groups[0]["lr"]
            if epoch_nr % print_every == 0:
                print(
                    "Epoch: {} - Train loss: {:.4f}, val loss: {:.4f}, val ROC-AUC: {:.4f}, lr: {:.2e}".format(
                        epoch_nr, train_loss, val_loss, val_auc, lr
                    )
                )
                sys.stdout.flush()

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
                print("New best validation ROC-AUC: {:.4f}".format(best_auc))
            else:
                epochs_without_improvement += 1
                if patience and epochs_without_improvement >= patience:
                    print(
                        "Early stopping at epoch {} (best val ROC-AUC {:.4f})".format(
                            epoch_nr, best_auc
                        )
                    )
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.last_best_auc = best_auc
        return model

    def evaluate(self, model, loader):
        model.eval()
        running_loss = 0.0
        seen = 0
        labels = []
        probs = []

        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs = inputs.to(self.device, non_blocking=True)
                targets_dev = targets.to(self.device, non_blocking=True).float()

                logits = self._forward_logits(model, inputs)
                batch_probs = torch.sigmoid(logits)
                if self.loss_fn is not None:
                    loss = self.loss_fn(logits, targets_dev)
                    running_loss += float(loss.item()) * inputs.size(0)
                seen += inputs.size(0)
                labels.extend(targets.cpu().numpy().tolist())
                probs.extend(batch_probs.detach().cpu().float().view(-1).numpy().tolist())

        val_loss = running_loss / max(seen, 1)
        try:
            val_auc = float(roc_auc_score(labels, probs))
        except ValueError:
            val_auc = float("nan")
        return val_loss, val_auc, labels, probs

    def test(self, model):
        if self.test_loader is None:
            print("Setting up test dataloader")
            self.test_loader = self.get_eval_loader(self.test_dir, return_paths=True)

        model.eval()
        paths = []
        labels = []
        probs = []

        with torch.no_grad():
            for batch_paths, inputs, targets in self.test_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                if self.use_tta:
                    view_probs = [
                        torch.sigmoid(self._forward_logits(model, view))
                        for view in _tta_views(inputs)
                    ]
                    batch_probs = torch.mean(torch.stack(view_probs, dim=0), dim=0)
                else:
                    batch_probs = torch.sigmoid(self._forward_logits(model, inputs))

                paths.extend(normalize_image_id(p) for p in batch_paths)
                labels.extend(targets.cpu().numpy().tolist())
                probs.extend(batch_probs.detach().cpu().float().view(-1).numpy().tolist())

        return paths, labels, probs
