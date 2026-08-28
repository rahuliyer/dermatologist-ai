# dermatologist-ai

ResNet/DenseNet detectors for Task 3 (lesion diagnosis) of the [ISIC 2017 challenge](https://challenge.isic-archive.com/landing/2017/). Project details are in [PROJECT_README.md](PROJECT_README.md).

The original training loop capped ROC-AUC (about **0.79** melanoma / **0.88** seborrheic keratosis on the saved `model_results.csv`). This revision fixes the issues that were holding the models back and switches to a recipe that matches published ISIC 2017 practice.

## What was limiting results

1. **Augmentations ran once.** `ExperimentRunner` decoded every image, applied `RandomHorizontalFlip` / `RandomVerticalFlip`, then cached the tensors. Two hundred epochs trained on that single frozen view of each photo.
2. **The classifier was linear.** Three `nn.Linear` layers were stacked with no ReLU or dropout, which is equivalent to one linear layer, then a `Sigmoid` + `BCELoss` pair that is less stable than logits + `BCEWithLogitsLoss`.
3. **Validation used training flips** and checkpoints followed that noisy loss (and, in `cancer_detector.py`, test-set ROC-AUC).
4. **Class imbalance was handled by copying files** on disk instead of sampling, so the same melanoma/SK images were overfit.
5. **SGD at 1e-5** on a randomly initialized head plus most of the backbone is a poor fine-tuning schedule.

## What changed

- Fresh geometric + color augmentation **every epoch** (flips, 90° rotation, shear, scale, `ColorJitter`), following the ISIC “scenario J” recipe.
- Dropout + ReLU classifier head and `BCEWithLogitsLoss`.
- Two-stage fine-tune: train the head, then unfreeze later backbone blocks with **AdamW** and cosine LR.
- Checkpoint on **validation ROC-AUC**, early stopping, no test-set peeking.
- `WeightedRandomSampler` instead of duplicated files.
- Test-time augmentation (flips + 90°/270° rotations).
- Optional DenseNet-121 and small ensembles.

Retrain before scoring: the new head is not compatible with the checked-in `best_*_model.pt` weights.

## Instructions

1. Create the conda environment:
```
conda env create -f environment.yml
```
A current PyTorch install also works (`pip install torch torchvision scikit-learn pandas matplotlib pillow`).

2. Download the [training](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip), [validation](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip), and [test](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip) data into `data/train`, `data/valid`, and `data/test`. Each split should contain `melanoma/`, `nevus/`, and `seborrheic_keratosis/`.

3. Train both binary detectors and write `model_results.csv`:
```
python -u cancer_detector.py
```

Useful flags:

```
python -u cancer_detector.py --arch densenet121 --ensemble 3
python -u cancer_detector.py --task melanoma --finetune-epochs 40
python -u cancer_detector.py --eval-only --output model_results.csv
```

`make_datasets.sh` is no longer required. Score a submission with:

```
python get_results.py model_results.csv
```
