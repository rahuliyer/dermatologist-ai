# dermatologist-ai

Train a CNN for [ISIC 2017 Task 3](https://challenge.isic-archive.com/landing/2017/) (lesion diagnosis): melanoma vs. nevus vs. seborrheic keratosis. Challenge details are in [PROJECT_README.md](PROJECT_README.md).

## Setup

1. Install dependencies (a current PyTorch install with CUDA is recommended):

```
pip install -r requirements.txt
```

2. Download the [training](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip) (5.3 GB), [validation](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip) (825 MB), and [test](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip) (5.1 GB) archives. Unzip them into `data/train`, `data/valid`, and `data/test`. Each split should contain `melanoma/`, `nevus/`, and `seborrheic_keratosis/`.

```
mkdir -p data
# unzip train.zip, valid.zip, and test.zip so that you have:
#   data/train/{melanoma,nevus,seborrheic_keratosis}/
#   data/valid/{melanoma,nevus,seborrheic_keratosis}/
#   data/test/{melanoma,nevus,seborrheic_keratosis}/
```

3. Train both binary detectors and write `model_results.csv`:

```
python -u cancer_detector.py
```

4. Score the submission:

```
python get_results.py model_results.csv
```

That prints Category 1 (melanoma ROC-AUC), Category 2 (seborrheic keratosis ROC-AUC), and Category 3 (their mean), and writes `roc_curves.png` and `confusion_matrix.png`.

## Optional flags

```
python -u cancer_detector.py --arch densenet121 --ensemble 3
python -u cancer_detector.py --task melanoma
python -u cancer_detector.py --eval-only --output model_results.csv
```

`make_datasets.sh` is only needed if you want explicit binary folders (`melanoma_dataset/`, `sk_dataset/`). The default trainer reads `data/` directly.
