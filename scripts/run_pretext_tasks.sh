#!/bin/bash
# baseline
python train.py logger=wandb experiment=Siamese_Type1_hokkaido.yaml model.unet=True model.cnn=True datamodule.balanced=True datamodule.balance_weights=False seed=12345
python train.py logger=wandb experiment=Siamese_Type1_kaikoura.yaml model.unet=True model.cnn=True datamodule.balanced=True datamodule.balance_weights=False seed=12345

