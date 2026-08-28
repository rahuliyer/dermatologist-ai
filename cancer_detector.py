"""Train binary melanoma and seborrheic-keratosis detectors for ISIC 2017.

The original training code had several issues that limited ROC-AUC:
- Random augmentations were applied once, then cached in a TensorDataset
- The classifier stacked Linear layers with no nonlinearities
- Checkpoints were chosen with train-time flips on the validation set
- Class imbalance was handled by duplicating files on disk
- SGD used a tiny LR on a randomly initialized head plus the backbone
"""

from __future__ import print_function

import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import densenet121, resnet18, resnet50

from experiment_runner import ExperimentRunner, unwrap


TASK_POSITIVE_CLASSES = {
    "melanoma": ("melanoma", "1"),
    "sk": ("seborrheic_keratosis", "1"),
}

TASK_CHECKPOINT = {
    "melanoma": "best_melanoma_model.pt",
    "sk": "best_sk_model.pt",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_backbone(arch, pretrained):
    constructors = {
        "resnet18": resnet18,
        "resnet50": resnet50,
        "densenet121": densenet121,
    }
    if arch not in constructors:
        raise ValueError("Unsupported architecture: {}".format(arch))

    builder = constructors[arch]
    try:
        if pretrained:
            from torchvision.models import ResNet18_Weights, ResNet50_Weights, DenseNet121_Weights
            weights = {
                "resnet18": ResNet18_Weights.DEFAULT,
                "resnet50": ResNet50_Weights.DEFAULT,
                "densenet121": DenseNet121_Weights.DEFAULT,
            }[arch]
            return builder(weights=weights)
        return builder(weights=None)
    except (ImportError, TypeError, AttributeError):
        return builder(pretrained=pretrained)


def _make_head(in_features, dropout):
    layers = []
    if dropout and dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    layers.extend([
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout if dropout else 0.0),
        nn.Linear(512, 1),
    ])
    for module in layers:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            nn.init.zeros_(module.bias)
    return nn.Sequential(*layers)


def get_model(arch="resnet50", dropout=0.5, pretrained=True, device=None):
    model = _load_backbone(arch, pretrained=pretrained)

    if arch.startswith("resnet"):
        model.fc = _make_head(model.fc.in_features, dropout)
    elif arch.startswith("densenet"):
        model.classifier = _make_head(model.classifier.in_features, dropout)
    else:
        raise ValueError("Unsupported architecture: {}".format(arch))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def head_prefixes(arch):
    if arch.startswith("resnet"):
        return ("fc",)
    if arch.startswith("densenet"):
        return ("classifier",)
    raise ValueError(arch)


def finetune_prefixes(arch):
    if arch.startswith("resnet"):
        return ("layer3", "layer4", "fc")
    if arch.startswith("densenet"):
        return ("features.denseblock4", "features.norm5", "classifier")
    raise ValueError(arch)


def set_trainable(model, prefixes):
    core = unwrap(model)
    prefix_tuple = tuple(prefixes)
    for name, param in core.named_parameters():
        param.requires_grad = any(
            name == prefix or name.startswith(prefix + ".") for prefix in prefix_tuple
        )


def build_optimizer(model, head_lr, backbone_lr, weight_decay=1e-4):
    core = unwrap(model)
    head_params = []
    backbone_params = []
    head_names = ("fc", "classifier")
    for name, param in core.named_parameters():
        if not param.requires_grad:
            continue
        if any(name == h or name.startswith(h + ".") for h in head_names):
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups = [{"params": head_params, "lr": head_lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    return optim.AdamW(groups, weight_decay=weight_decay)


def get_train_transforms(image_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=90, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size=224):
    resize_to = int(round(image_size * 256 / 224))
    return transforms.Compose([
        transforms.Resize(resize_to),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _split_dirs(data_root):
    return (
        os.path.join(data_root, "train"),
        os.path.join(data_root, "valid"),
        os.path.join(data_root, "test"),
    )


def make_runner(args, task, device, loss_fn=None):
    train_dir, valid_dir, test_dir = _split_dirs(args.data_root)
    return ExperimentRunner(
        loss_fn=loss_fn,
        train_dir=train_dir,
        valid_dir=valid_dir,
        test_dir=test_dir,
        train_transforms=get_train_transforms(args.image_size),
        eval_transforms=get_eval_transforms(args.image_size),
        positive_classes=TASK_POSITIVE_CLASSES[task],
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=device,
        use_tta=not args.no_tta,
    )


def save_model(model, path):
    torch.save(unwrap(model).state_dict(), path)


def load_state_dict(model, path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    # Handle checkpoints saved from DataParallel.
    if any(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}

    unwrap(model).load_state_dict(state)
    return model


def train_one_model(args, task, device, seed):
    set_seed(seed)
    print("\n=== Training {} detector (seed={}) ===".format(task, seed))
    model = get_model(
        arch=args.arch,
        dropout=args.dropout,
        pretrained=not args.no_pretrained,
        device=device,
    )
    unwrap(model).arch = args.arch

    loss_fn = nn.BCEWithLogitsLoss()
    runner = make_runner(args, task, device, loss_fn=loss_fn)

    print("Stage 1: training classifier head")
    set_trainable(model, head_prefixes(args.arch))
    optimizer = build_optimizer(model, head_lr=args.head_lr, backbone_lr=0.0)
    model = runner.train(
        model,
        optimizer,
        num_epochs=args.head_epochs,
        scheduler=None,
        patience=args.patience,
    )
    stage1_auc = runner.last_best_auc

    print("Stage 2: fine-tuning later backbone blocks")
    set_trainable(model, finetune_prefixes(args.arch))
    optimizer = build_optimizer(model, head_lr=args.head_lr, backbone_lr=args.backbone_lr)
    scheduler = None
    if args.finetune_epochs > 0:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.finetune_epochs, eta_min=args.backbone_lr * 0.01
        )
    model = runner.train(
        model,
        optimizer,
        num_epochs=args.finetune_epochs,
        scheduler=scheduler,
        patience=args.patience,
    )
    print(
        "Finished {} seed {}: stage1 val AUC={:.4f}, stage2 val AUC={:.4f}".format(
            task, seed, stage1_auc, runner.last_best_auc
        )
    )
    return model, runner, runner.last_best_auc


def train_task(args, task, device):
    save_path = TASK_CHECKPOINT[task]
    best_auc = float("-inf")
    best_model = None
    ensemble_paths = []

    for member in range(args.ensemble):
        seed = args.seed + member
        model, runner, val_auc = train_one_model(args, task, device, seed)
        member_path = save_path if args.ensemble == 1 else "{}.e{}".format(save_path, member)
        save_model(model, member_path)
        ensemble_paths.append(member_path)
        print("Saved {}".format(member_path))
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model
            if args.ensemble > 1:
                save_model(model, save_path)
                print("Best {} member so far; copied to {}".format(task, save_path))

    if best_model is None:
        raise RuntimeError("Training produced no model for task {}".format(task))
    return best_model, ensemble_paths


def collect_predictions(args, task, device, model_paths):
    runner = make_runner(args, task, device, loss_fn=None)
    member_probs = []
    paths = labels = None
    for path in model_paths:
        model = get_model(
            arch=args.arch,
            dropout=args.dropout,
            pretrained=False,
            device=device,
        )
        load_state_dict(model, path, device)
        paths, labels, probs = runner.test(model)
        member_probs.append(np.asarray(probs, dtype=np.float64))
    probs = np.mean(np.stack(member_probs, axis=0), axis=0)
    return paths, labels, probs


def write_results_csv(fname, paths, m_probs, sk_probs, order_csv="ground_truth.csv"):
    rows = {
        path: (float(m_prob), float(sk_prob))
        for path, m_prob, sk_prob in zip(paths, m_probs, sk_probs)
    }

    ids = list(paths)
    if os.path.isfile(order_csv):
        with open(order_csv, "r") as handle:
            ordered = [row["Id"] for row in csv.DictReader(handle)]
        if ordered:
            ids = ordered

    with open(fname, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Id", "task_1", "task_2"])
        missing = 0
        for image_id in ids:
            if image_id not in rows:
                missing += 1
                continue
            m_prob, sk_prob = rows[image_id]
            writer.writerow([image_id, m_prob, sk_prob])
        if missing:
            print("Warning: {} ids from {} were missing in predictions".format(missing, order_csv))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train ISIC 2017 lesion detectors")
    parser.add_argument("--data-root", default="data", help="Folder with train/valid/test splits")
    parser.add_argument("--task", choices=["melanoma", "sk", "both"], default="both")
    parser.add_argument("--arch", choices=["resnet18", "resnet50", "densenet121"], default="resnet50")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="model_results.csv")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Using device: {}".format(device))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    tasks = ["melanoma", "sk"] if args.task == "both" else [args.task]
    model_paths = {task: [TASK_CHECKPOINT[task]] for task in tasks}

    if not args.eval_only:
        for task in tasks:
            _, paths = train_task(args, task, device)
            model_paths[task] = paths

    for task in tasks:
        for path in model_paths[task]:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    "Missing {} checkpoint '{}'. Train first or pass --eval-only after training.".format(
                        task, path
                    )
                )

    # Always score on the original 3-class test directory so the CSV matches ground_truth.csv.
    eval_args = argparse.Namespace(**vars(args))
    if os.path.isdir(os.path.join("data", "test")):
        eval_args.data_root = "data"

    melanoma_paths = model_paths.get("melanoma", [TASK_CHECKPOINT["melanoma"]])
    sk_paths = model_paths.get("sk", [TASK_CHECKPOINT["sk"]])
    if args.task == "melanoma":
        paths, _, m_probs = collect_predictions(eval_args, "melanoma", device, melanoma_paths)
        sk_probs = np.zeros(len(m_probs))
    elif args.task == "sk":
        paths, _, sk_probs = collect_predictions(eval_args, "sk", device, sk_paths)
        m_probs = np.zeros(len(sk_probs))
    else:
        paths, _, m_probs = collect_predictions(eval_args, "melanoma", device, melanoma_paths)
        _, _, sk_probs = collect_predictions(eval_args, "sk", device, sk_paths)

    write_results_csv(args.output, paths, m_probs, sk_probs)
    print("Wrote predictions to {}".format(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
