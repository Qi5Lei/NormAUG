#!/bin/bash

cd ../..

DATA=/data/yhp/DG/DG

DATASET=$1
CFG=$2 # v1, v2, v3, v4
DEVICE=$3

if [ ${DATASET} == dg_pacs ]; then
    D1=art_painting
    D2=cartoon
    D3=photo
    D4=sketch
elif [ ${DATASET} == dg_officehome ]; then
    D1=art
    D2=clipart
    D3=product
    D4=real_world
elif [ ${DATASET} == dg_mini_dn ]; then
    D1=clipart
    D2=painting
    D3=real
    D4=sketch
elif [ ${DATASET} == dg_digits ]; then
    D1=mnist
    D2=mnist_m
    D3=svhn
    D4=syn
fi

TRAINER=NormAUG
NET=resnet18

for SEED in $(seq 1 3)
do
    for SETUP in $(seq 1 4)
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D2}
            S2=${D3}
            S3=${D4}
            T=${D1}
        elif [ ${SETUP} == 2 ]; then
            S1=${D1}
            S2=${D3}
            S3=${D4}
            T=${D2}
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D4}
            T=${D3}
        elif [ ${SETUP} == 4 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            T=${D4}
        fi

        CUDA_VISIBLE_DEVICES=${DEVICE} \
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --source-domains ${S1} ${S2} ${S3} \
        --target-domains ${T} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${DATASET}_${CFG}.yaml \
        --output-dir output/${DATASET}/${TRAINER}/${NET}/${CFG}/${T}/seed${SEED} \
        MODEL.BACKBONE.NAME ${NET}
    done
done