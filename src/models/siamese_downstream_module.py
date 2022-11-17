"""
Siamese Network trained to predict landlside induced change
"""
from pathlib import Path
import torch
from pytorch_lightning import LightningModule
import segmentation_models_pytorch as smp
import torchvision.models as models
import numpy as np
import math
import torch.nn as nn
from torchmetrics.functional import precision_recall
from torchmetrics import AveragePrecision,Accuracy
from src.models.siamese_module import Siamese_Type_1


def compute_metrics(preds, targets, threshold=0.5):
    #print(preds)
    prec, rec      = precision_recall(preds, targets, threshold=threshold)
    f1_score       = 2*(prec*rec)/(prec+rec)
    
    # pr_curve       = PrecisionRecallCurve(num_classes=5)
    # precision, recall, thresholds = pr_curve(preds, targets)
    
    average_precision = AveragePrecision()
    AP_score       = average_precision(preds, targets)
    accuracy       = Accuracy(threshold=threshold).cuda()
    acc            = accuracy(preds, targets)
    return f1_score, AP_score, prec, rec, acc


class Segmentation_Model(LightningModule):

    def __init__(
        self,
            input_size: list     = [2,128,128],
            embedding_size: int  = 32,
            pre_train_augmented  = False,
            pretrain_path        = None,
            pretrain_params      = {},
            encoder_depth: int   = 3,
            decoder_channels     = [128, 64, 32],
            base_lr              = 1.5e-4,
            unet                 = False,
            loss                 = 'ce'):
            
        super().__init__()

        self.input_size  = list(input_size)
        
        self.channels    = input_size[0]
        self.image_size  = input_size[1::] 
        self.base_lr     = base_lr
        self.pretrain_augmented = pre_train_augmented

        if unet:
            self.basemodel   = smp.Unet(encoder_name='resnet34', encoder_depth=encoder_depth, decoder_channels=decoder_channels, encoder_weights=None, decoder_use_batchnorm=True, in_channels=self.input_size[0], classes=embedding_size, activation='identity', aux_params=None)


        else:
            self.basemodel   = nn.Sequential(torch.nn.Conv2d(self.channels,self.channels*2, kernel_size=8,stride=1, bias=True, padding='same'),
                                torch.nn.BatchNorm2d(self.channels*2),
                                torch.nn.LeakyReLU(),
                                torch.nn.Conv2d(self.channels*2,self.channels*4, kernel_size=8,stride=1, bias=True, padding='same'),
                                torch.nn.BatchNorm2d(self.channels*4),
                                torch.nn.LeakyReLU(),
                                torch.nn.Conv2d(self.channels*4,embedding_size, kernel_size=8,stride=1, bias=True, padding='same'),
                                torch.nn.BatchNorm2d(embedding_size),
                                torch.nn.LeakyReLU()
                                )
            
        depth                = embedding_size
        
        if self.pretrain_augmented:
            depth            = embedding_size+pretrain_params['embedding_size']*2
            self.pretrained  = Siamese_Type_1(**pretrain_params)
            checkpoint       = torch.load(pretrain_path)
            self.pretrained.load_state_dict(checkpoint['state_dict'])
            self.pretrained.eval()
     
        self.cnn             = nn.Sequential(torch.nn.Conv2d(depth, depth, kernel_size=8,stride=1, bias=True, padding='same'),
                                         torch.nn.BatchNorm2d(depth),
                                         torch.nn.LeakyReLU(),
                                         torch.nn.Conv2d(depth, 1, kernel_size=8,stride=1, bias=True, padding='same'))
        self.loss                = loss
        if loss == 'ce':                        
            self.criterion       = nn.CrossEntropyLoss(reduction='none')
        elif loss=='dice':
            self.criterion       = smp.losses.DiceLoss(mode='binary')
        
        if torch.cuda.is_available():
            self.net = self.basemodel.cuda()
            self.fc  = self.cnn.cuda()
            if pre_train_augmented:
                self.pretrained = self.pretrained.cuda()                   
                for params in self.pretrained.parameters():
                    params.requires_grad = False


    def forward(self, pre, post):
                                             
        inputs = torch.cat((pre,post),1)                  
        emb = self.basemodel(inputs)
        
        if self.pretrain_augmented:
            with torch.no_grad():
                pre_emb = self.pretrained.forward_once(pre)
                pre_emb = torch.cat((pre_emb, self.pretrained.forward_once(post)),1)

            emb     = torch.cat((emb, pre_emb), 1)
        
        output = self.cnn(emb)
        if self.loss=='ce':
            output = output.view(output.size()[0], -1)
        
        return output

    def step(self, batch):
        
        pre, post, label, names, weight = batch

        names = np.asarray(names).T
        
        preds  = self.forward(pre,post)
        label  = label.view(preds.size())

        loss   = torch.mean(weight*self.criterion(preds.float(),label.float()))
        return loss, pre, post, preds, label, names
    

    def training_step(self, batch, batch_idx: int):
        loss, pre, post, preds, targets, names = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        
        f1, auprc, prec, rec, acc = compute_metrics(torch.sigmoid(preds.detach()), targets.detach())
        self.log("train/precision", prec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/recall", rec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/f1", f1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/AP", auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "inputs": torch.reshape(targets,(-1,1,128,128)),"preds": torch.reshape(preds,(-1,1,128,128))}

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx: int):
        loss, pre, post, preds, targets, names = self.step(batch)

        self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        
        f1, auprc, prec, rec, acc = compute_metrics(torch.sigmoid(preds.detach()), targets.detach())
        self.log("valid/precision", prec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("valid/recall", rec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("valid/f1", f1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("valid/AP", auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("valid/acc", acc, on_step=False, on_epoch=True, prog_bar=False)

        
        return {"loss": loss, "inputs": torch.reshape(targets,(-1,1,128,128)),"preds": torch.reshape(preds,(-1,1,128,128))}
    

    def test_step(self, batch, batch_idx: int):
        loss, pre, post, preds, targets, names = self.step(batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        
        f1, auprc, prec, rec, acc = compute_metrics(torch.sigmoid(preds.detach()), targets.detach())
        self.log("test/precision", prec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/recall", rec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/f1", f1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/AP", auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss}


    def test_epoch_end(self, outputs):
        pass


    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(),lr=self.base_lr, capturable=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=1e-4, patience=50, factor=0.9)
        
        return {'lr_scheduler':scheduler, 'optimizer':optimizer, 'monitor': 'train/loss'}
