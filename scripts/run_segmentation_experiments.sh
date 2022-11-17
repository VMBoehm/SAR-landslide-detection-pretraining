#!/bin/bash

for ii in 20 2 5 10
do
    for seed in 12345 55555 54321
    do
         python train.py logger=wandb experiment=seg_hokk_hokk_pretrain_cnn.yaml model.pre_train_augmented=False datamodule.trainsize=$ii seed=$seed
         python train.py logger=wandb experiment=seg_hokk_hokk_pretrain_cnn.yaml model.pre_train_augmented=True datamodule.trainsize=$ii seed=$seed
    done
done

ii=2
for seed in 54321 56762 9867
do
    python train.py logger=wandb experiment=seg_hokk_hokk_pretrain_cnn.yaml model.pre_train_augmented=False datamodule.trainsize=$ii seed=$seed
    python train.py logger=wandb experiment=seg_hokk_hokk_pretrain_cnn.yaml model.pre_train_augmented=True datamodule.trainsize=$ii seed=$seed
done


for ii in 20 2 5 10
do
    for seed in 12345 55555 54321
    do
         python train.py logger=wandb experiment=seg_hokk_kaik_pretrain_cnn.yaml model.pre_train_augmented=False datamodule.trainsize=$ii seed=$seed
         python train.py logger=wandb experiment=seg_hokk_kaik_pretrain_cnn.yaml model.pre_train_augmented=True datamodule.trainsize=$ii seed=$seed
    done
done

ii=2
for seed in 54321 56762 9867
do
    python train.py logger=wandb experiment=seg_hokk_kaik_pretrain_cnn.yaml model.pre_train_augmented=False datamodule.trainsize=$ii seed=$seed
    python train.py logger=wandb experiment=seg_hokk_kaik_pretrain_cnn.yaml model.pre_train_augmented=True datamodule.trainsize=$ii seed=$seed
done

