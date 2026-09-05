"""Train melanoma and seborrheic-keratosis classifiers."""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

from experiment_runner import ExperimentRunner

TASKS = {
    "melanoma": "melanoma",
    "sk": "seborrheic_keratosis",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA device"
        )
    return torch.device(requested)


def get_model(
    trainable_layers: int,
    device: torch.device,
    *,
    pretrained: bool = True,
) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, 1),
    )

    layers = list(model.children())
    trainable_layers = max(1, min(trainable_layers, len(layers)))
    for layer in layers[:-trainable_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False

    model = model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    return model


def get_train_transforms():
    return [
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]


def get_test_transforms():
    return [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def train_task(args, task: str, device: torch.device):
    positive_class = TASKS[task]
    ensemble_probs = []
    test_paths = None
    test_labels = None
    best_valid_loss = float("inf")
    best_checkpoint = args.output_dir / f"best_{task}_model.pt"

    for model_number in range(1, args.models + 1):
        model_seed = args.seed + model_number - 1
        seed_everything(model_seed)
        print(
            f"\n{task}: model {model_number}/{args.models}, epochs={args.epochs}, "
            f"lr={args.learning_rate:g}, trainable_layers={args.trainable_layers}"
        )
        runner = ExperimentRunner(
            nn.BCEWithLogitsLoss(),
            args.data_dir,
            args.data_dir,
            get_train_transforms(),
            get_test_transforms(),
            positive_class=positive_class,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
            seed=model_seed,
            amp=not args.no_amp,
        )
        model = get_model(
            args.trainable_layers,
            device,
            pretrained=not args.no_pretrained,
        )
        optimizer = optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        model = runner.train(model, optimizer, args.epochs, patience=args.patience)
        paths, labels, probs = runner.test(model)
        auc = roc_auc_score(labels, probs)
        print(f"{task}: test ROC AUC={auc:.5f}")
        ensemble_probs.append(np.asarray(probs))
        test_paths, test_labels = paths, labels

        if (
            runner.best_valid_loss is not None
            and runner.best_valid_loss < best_valid_loss
        ):
            best_valid_loss = runner.best_valid_loss
            torch.save(
                {
                    "task": task,
                    "positive_class": positive_class,
                    "state_dict": {
                        name: tensor.detach().cpu()
                        for name, tensor in unwrap_model(model).state_dict().items()
                    },
                    "validation_loss": best_valid_loss,
                    "test_roc_auc": auc,
                    "seed": model_seed,
                },
                best_checkpoint,
            )
            print(f"Saved {best_checkpoint}")

    return test_paths, test_labels, np.mean(ensemble_probs, axis=0)


def write_results_csv(path: Path, image_paths, melanoma_probs, sk_probs) -> None:
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Id", "task_1", "task_2"])
        for image_path, melanoma_prob, sk_prob in zip(
            image_paths, melanoma_probs, sk_probs, strict=True
        ):
            writer.writerow([image_path, melanoma_prob, sk_prob])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--task", choices=["both", *TASKS], default="both")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--models", type=int, default=1, help="Models per task")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--trainable-layers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.models < 1 or args.batch_size < 1:
        parser.error("epochs, models, and batch-size must all be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"PyTorch {torch.__version__}; device={device}; GPUs={torch.cuda.device_count()}"
    )

    selected_tasks = list(TASKS) if args.task == "both" else [args.task]
    results = {}
    paths_by_task = {}
    for task in selected_tasks:
        paths, _, probabilities = train_task(args, task, device)
        paths_by_task[task] = paths
        results[task] = probabilities

    if args.task == "both":
        if paths_by_task["melanoma"] != paths_by_task["sk"]:
            raise RuntimeError("Task test loaders produced different image ordering")
        results_path = args.output_dir / "model_results.csv"
        write_results_csv(
            results_path,
            paths_by_task["melanoma"],
            results["melanoma"],
            results["sk"],
        )
        print(f"Wrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
