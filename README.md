# dermatologist-ai

ResNet-50 transfer learning for the lesion-diagnosis task from the 2017 ISIC
Challenge. The project trains separate binary classifiers for melanoma and
seborrheic keratosis and writes challenge-format probabilities.

## Setup

Install the locked Python environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Download the [training data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip),
[validation data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip),
and [test data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip).
Extract them into this layout:

```text
data/
├── train/
│   ├── melanoma/
│   ├── nevus/
│   └── seborrheic_keratosis/
├── valid/
│   └── ...
└── test/
    └── ...
```

No copied or upsampled dataset is needed. The training loader balances each
binary task with weighted sampling.

## Training

Train both classifiers with the defaults (20 epochs, mixed precision, and all
visible CUDA devices):

```bash
uv run python -u cancer_detector.py
```

For a short end-to-end run:

```bash
uv run python -u cancer_detector.py --epochs 1
```

Use `--help` for batch size, worker count, fine-tuning depth, task selection,
and other controls. On a multi-GPU host the model automatically uses PyTorch
`DataParallel`.

## Results

A training run on September 4, 2026 used two NVIDIA RTX 2080 GPUs, PyTorch
2.14.0 with CUDA 13.0, batch size 64, mixed precision, ImageNet-pretrained
ResNet-50 weights, and the default optimizer settings. Early stopping selected
the lowest-validation-loss epoch for each task.

| Task | Best epoch | Validation loss | Test ROC AUC |
| --- | ---: | ---: | ---: |
| Melanoma | 3 | 0.39596 | 0.83339 |
| Seborrheic keratosis | 12 | 0.30722 | 0.90946 |
| Mean | — | — | 0.87143 |

The generated CSV contains all 600 test predictions. The repository's previous
`model_results.csv` scores 0.835 mean ROC AUC with the updated evaluator.

Checkpoints and `model_results.csv` are written to `outputs/`. The CSV can be
scored with:

```bash
uv run python get_results.py outputs/model_results.csv
```

See [PROJECT_README.md](PROJECT_README.md) for the original project and
challenge background.
