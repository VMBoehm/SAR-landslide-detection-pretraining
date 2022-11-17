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


class Siamese_Type_1(LightningModule):

    def __init__(
        self,
            input_size: list     = [2,128,128],
            embedding_size: int  = 32,
            decoder_depth: int   = 3,
            encoder_depth: int   = 3,
            base_lr              = 1.5e-4,
            unet                 = False,
            cnn                  = True,
            decoder_channels     = [128,64,32]):
            
        super().__init__()

        self.input_size  = list(input_size)
        
        self.channels    = input_size[0]
        self.image_size  = input_size[1::] 
        self.base_lr     = base_lr
        
        #decoder_channels = [embedding_size for ii in range(decoder_depth)]
        
        dropout_rate     = 0.8
        
        if unet:
            self.net         = smp.Unet(encoder_name='resnet34', encoder_depth=encoder_depth, encoder_weights=None, decoder_use_batchnorm=True, in_channels=self.input_size[0], decoder_channels=list(decoder_channels)[:decoder_depth], classes=embedding_size, activation='identity', aux_params=None)


        else:
            self.net         = nn.Sequential(torch.nn.Conv2d(self.channels,self.channels*4, kernel_size=8,stride=1, bias=True, padding='same'),
                                torch.nn.BatchNorm2d(self.channels*4),
                                torch.nn.LeakyReLU(),
                                # torch.nn.Conv2d(self.channels*4,self.channels*4*4, kernel_size=8,stride=1, bias=True, padding='same'),
                                # torch.nn.BatchNorm2d(self.channels*4*4),
                                # torch.nn.LeakyReLU(),
                                torch.nn.Conv2d(self.channels*4,embedding_size, kernel_size=8,stride=1, bias=True, padding='same'),
                                torch.nn.BatchNorm2d(embedding_size),
                                torch.nn.LeakyReLU()
                                )
            
        size                 = np.prod(self.image_size)*embedding_size*2  
        if cnn:   
            self.cnn         = nn.Sequential(torch.nn.Conv2d(embedding_size*2, 1, kernel_size=8 ,stride=1, bias=True, padding='valid'),
                                             torch.nn.BatchNorm2d(1),
                                             torch.nn.LeakyReLU())
            size             = 1*128**2#torch.nn.LeakyReLU()
        
        self.fc          = nn.Sequential(nn.Linear(size, 8),
                                         nn.BatchNorm1d(8),
                                         nn.LeakyReLU(inplace=True),
                                         nn.Dropout(dropout_rate),
                                         nn.Linear(8, 1))
        
        if cnn:
            self.int_layer = True
        else:
            self.int_layer = False
        
        self.criterion   = nn.CrossEntropyLoss()
        
        if torch.cuda.is_available():
            self.net = self.net.cuda()
            #self.fc  = self.fc.cuda()
            if cnn:
                self.cnn = self.cnn.cuda()
            

    def forward_once(self, x):
        output = self.net(x)
        
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)
        
        #output = self.cnn(output)
        if self.int_layer:
            output = self.cnn(output)
        output = output.view(output.size()[0], -1)
        # pass the concatenation to the linear layers
        output = torch.mean(output, axis=-1)#self.fc(output)
        
        return output

    def step(self, batch):
        
        pre, post, label, names, weight = batch

        names = np.asarray(names).T
        
        preds  = self.forward(pre,post)
        preds  = torch.squeeze(preds)
        weight = torch.reshape(weight.float(),preds.shape)

        loss  = self.criterion(preds.float(),label.float())

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

        return {"loss": loss}

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

        
        return {"loss": loss}
    

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=1e-4, patience=200, factor=0.9)
        
        return {'lr_scheduler':scheduler, 'optimizer':optimizer, 'monitor': 'valid/loss'}