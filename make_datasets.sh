#!/bin/sh
# Optional helper that builds binary-folder copies of the official 3-class
# splits. The current trainer does not need this: it reads data/{train,valid,test}
# directly and balances classes with WeightedRandomSampler.
#
# Kept so existing scripts that pass --data-root melanoma_dataset still work.

set -e

mkdir -p melanoma_dataset/train/0 melanoma_dataset/train/1
mkdir -p melanoma_dataset/valid/0 melanoma_dataset/valid/1
mkdir -p melanoma_dataset/test/0 melanoma_dataset/test/1

cp data/train/melanoma/* melanoma_dataset/train/1/
cp data/train/nevus/* melanoma_dataset/train/0/
cp data/train/seborrheic_keratosis/* melanoma_dataset/train/0/

cp data/valid/melanoma/* melanoma_dataset/valid/1/
cp data/valid/nevus/* melanoma_dataset/valid/0/
cp data/valid/seborrheic_keratosis/* melanoma_dataset/valid/0/

cp data/test/melanoma/* melanoma_dataset/test/1/
cp data/test/nevus/* melanoma_dataset/test/0/
cp data/test/seborrheic_keratosis/* melanoma_dataset/test/0/

mkdir -p sk_dataset/train/0 sk_dataset/train/1
mkdir -p sk_dataset/valid/0 sk_dataset/valid/1
mkdir -p sk_dataset/test/0 sk_dataset/test/1

cp data/train/seborrheic_keratosis/* sk_dataset/train/1/
cp data/train/nevus/* sk_dataset/train/0/
cp data/train/melanoma/* sk_dataset/train/0/

cp data/valid/seborrheic_keratosis/* sk_dataset/valid/1/
cp data/valid/nevus/* sk_dataset/valid/0/
cp data/valid/melanoma/* sk_dataset/valid/0/

cp data/test/seborrheic_keratosis/* sk_dataset/test/1/
cp data/test/nevus/* sk_dataset/test/0/
cp data/test/melanoma/* sk_dataset/test/0/
