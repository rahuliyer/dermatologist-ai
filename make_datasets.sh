#!/bin/sh

mkdir -p melanoma_dataset/train/0
mkdir -p melanoma_dataset/train/1

mkdir -p melanoma_dataset/valid/0
mkdir -p melanoma_dataset/valid/1

mkdir -p melanoma_dataset/test/0
mkdir -p melanoma_dataset/test/1

cp data/train/melanoma/* melanoma_dataset/train/1/
cp data/train/nevus/* melanoma_dataset/train/0/
cp data/train/seborrheic_keratosis/* melanoma_dataset/train/0/

cp data/valid/melanoma/* melanoma_dataset/valid/1/
cp data/valid/nevus/* melanoma_dataset/valid/0/
cp data/valid/seborrheic_keratosis/* melanoma_dataset/valid/0/

cp data/test/melanoma/* melanoma_dataset/test/1/
cp data/test/nevus/* melanoma_dataset/test/0/
cp data/test/seborrheic_keratosis/* melanoma_dataset/test/0/

cd melanoma_dataset/train/1
files=`ls`
for f in $files
do
    cp $f "1-$f"
    cp $f "2-$f"
    cp $f "3-$f"
done
cd ../../..

mkdir -p sk_dataset/train/0
mkdir -p sk_dataset/train/1

mkdir -p sk_dataset/valid/0
mkdir -p sk_dataset/valid/1

mkdir -p sk_dataset/test/0
mkdir -p sk_dataset/test/1

cp data/train/seborrheic_keratosis/* sk_dataset/train/1/
cp data/train/nevus/* sk_dataset/train/0/
cp data/train/melanoma/* sk_dataset/train/0/

cp data/valid/seborrheic_keratosis/* sk_dataset/valid/1/
cp data/valid/nevus/* sk_dataset/valid/0/
cp data/valid/melanoma/* sk_dataset/valid/0/

cp data/test/seborrheic_keratosis/* sk_dataset/test/1/
cp data/test/nevus/* sk_dataset/test/0/
cp data/test/melanoma/* sk_dataset/test/0/

cd sk_dataset/train/1
files=`ls`
for f in $files
do
    cp $f "1-$f"
    cp $f "2-$f"
    cp $f "3-$f"
    cp $f "4-$f"
    cp $f "5-$f"
    cp $f "6-$f"
done
