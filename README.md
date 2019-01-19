# dermatologist-ai

## Introduction ##
This uses Resnet50 to solve the Task 3 - lesion diagnosis - of the ISIC challenge. The details of the project can be found in the [PROJECT_README.md](https://github.com/rahuliyer/dermatologist-ai/blob/master/PROJECT_README.md).

## Instructions ##
1. Create the conda environment
```
conda env create -f environment.yml
```

2. Get the [training data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip), [validaton data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip), and [test data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip) and place them in `data/train`, `data/valid`, and `data/test` directories respectively.

3. Since there are two separate models - one for melanoma and one for seborrheic keratosis - and the datasets are imabalanced, run `make_datasets.sh` to create the balanced datasets for each model. The datasets are balanced by upsampling.

4. Start training by running `python -u cancer_detector.py`
