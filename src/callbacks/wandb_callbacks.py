import glob
import os
from typing import List
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
import gc
import numpy as np



from typing import List

import torch
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
from pytorch_lightning import Callback
from torch import nn
from tqdm import tqdm
from tqdm import trange


def make_figure(data):
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(data)
    fig.colorbar(im, ax=ax)
    plt.close()
    return fig

def make_plot(data1, data2=None, label1='target', label2='prediction'):
    fig, ax = plt.subplots(1, 1)
    ax.plot(data1, label=label1, ls='', marker='o')
    if np.all(data2)!=None:
        ax.plot(data2, label=label2, ls='', marker='+')
    plt.legend()
    plt.close()
    return fig

def make_scatter(preds, targets):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(targets, preds, alpha=.5)
    plt.grid()
    plt.xlabel("targets")
    plt.ylabel("predictions")
    plt.close()
    return fig

def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )
    
    
    
class LogValPredictions_MAE(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8, counts=True):
        super().__init__()
        self.num_samples = num_samples
        self.preds   = []
        self.inputs = []
        #self.names = []
        #self.masks =[]

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):

        """Gather data from single batch."""
        if self.ready:
            """Gather data from single batch."""
            self.preds.append(outputs["preds"])

            self.inputs.append(outputs["inputs"])

            #self.names.append(outputs['input_names'])
            
            #self.masks.append(outputs['mask'])
                
            
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment  = logger.experiment
            preds       = torch.cat(self.preds[:self.num_samples]).detach().cpu().numpy()
            inputs      = torch.cat(self.inputs[:self.num_samples]).detach().cpu().numpy()
            #masks        = torch.cat(self.masks[:self.num_samples]).detach().cpu().numpy()
            #masked_inputs = inputs*masks
            #recons       = preds
            #recons[masks.astype(bool)] = inputs[masks.astype(bool)]

            names      = np.concatenate(self.names[:self.num_samples])

            input_imgs        = []


            for i in range(self.num_samples):
                if inputs[i].shape[0]>1:
                    for jj in range(inputs[i].squeeze().shape[0]):
                        input_imgs.append(wandb.Image(make_figure(inputs[i].squeeze()[jj]), caption=f'{names[i,jj]} target'))
                        #input_imgs.append(wandb.Image(make_figure(masked_inputs[i].squeeze()[jj]), caption=f'{names[i,jj]} inputs'))
                        input_imgs.append(wandb.Image(make_figure(preds[i].squeeze()[jj]), caption=f'{names[i,jj]} recon'))
                        #input_imgs.append(wandb.Image(make_figure(recons[i].squeeze()[jj]), caption=f'{names[i,jj]} recons inpainted'))
                else:
                    input_imgs.append(wandb.Image(make_figure(inputs[i].squeeze()), caption=f'{names[i]} inputs'))
                    #input_imgs.append(wandb.Image(make_figure(masked_inputs[i].squeeze()), caption=f'{names[i]} inputs'))
                    input_imgs.append(wandb.Image(make_figure(preds[i].squeeze()), caption=f'{names[i]} recon'))
                    #input_imgs.append(wandb.Image(make_figure(recons[i].squeeze()), caption=f'{names[i]} recons inpainted'))


            experiment.log(
                {
                    "valid/maps": input_imgs,
                }
            )

            self.preds.clear()
            self.inputs.clear()
            self.names.clear()
            self.masks.clear()
    
    
    
class LogTrainPredictions_MAE_Downstream(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 2, counts=True):
        super().__init__()
        self.num_samples = num_samples
        self.preds   = []
        self.inputs  = []
        self.names   = []

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if self.ready:
            """Gather data from single batch."""
            self.preds.append(outputs["preds"])

            self.inputs.append(outputs["inputs"])

            #self.names.append(outputs['input_names'])


    def on_train_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            preds      = torch.cat(self.preds[:self.num_samples]).detach().cpu().numpy()
            inputs     = torch.cat(self.inputs[:self.num_samples]).detach().cpu().numpy()

            #names      = np.concatenate(self.names[:self.num_samples])

            input_imgs        = []


            for i in range(self.num_samples):
                if inputs[i].shape[0]>1:
                    for jj in range(inputs[i].squeeze().shape[0]):
                        input_imgs.append(wandb.Image(make_figure(inputs[i].squeeze()[jj]), caption=f'inputs'))
                        input_imgs.append(wandb.Image(make_figure(preds[i].squeeze()[jj]>0.5), caption=f'recon>0.5'))
                        input_imgs.append(wandb.Image(make_figure(preds[i].squeeze()[jj]), caption=f'recon'))
                else:
                    input_imgs.append(wandb.Image(make_figure(inputs[i].squeeze()), caption=f'original'))
                    input_imgs.append(wandb.Image(make_figure(preds[i].squeeze()>0.5), caption=f'recon>0.5')) 
                    input_imgs.append(wandb.Image(make_figure(preds[i].squeeze()), caption=f'recon'))                


            experiment.log(
                {
                    "train/maps": input_imgs,
                }
            )

            self.preds.clear()
            self.inputs.clear()
            self.names.clear()

    
class LogValPredictions_MAE_Downstream(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8, counts=True):
        super().__init__()
        self.num_samples = num_samples
        self.preds   = []
        self.inputs = []
        self.names = []

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):

        """Gather data from single batch."""
        if self.ready:
            """Gather data from single batch."""
            self.preds.append(outputs["preds"])

            self.inputs.append(outputs["inputs"])

            #self.names.append(outputs['input_names'])
            
                
            
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            preds      = torch.cat(self.preds[:self.num_samples]).detach().cpu().numpy()
            inputs     = torch.cat(self.inputs[:self.num_samples]).detach().cpu().numpy()

            #names      = np.concatenate(self.names[:self.num_samples])

            input_imgs        = []

            for i in range(self.num_samples):
                if inputs[i].shape[0]>1:
                    for jj in range(inputs[i].squeeze().shape[0]):
                        input_imgs.append(wandb.Image(make_figure(inputs[i].squeeze()[jj]), caption=f'inputs'))
                        input_imgs.append(wandb.Image(make_figure(preds[i].squeeze()[jj]>0.5), caption=f'recon>0.5'))
                        input_imgs.append(wandb.Image(make_figure(preds[i].squeeze()[jj]), caption=f'recon'))
                else:
                    input_imgs.append(wandb.Image(make_figure(inputs[i].squeeze()), caption=f'original'))
                    input_imgs.append(wandb.Image(make_figure(preds[i].squeeze()>0.5), caption=f'recon>0.5')) 
                    input_imgs.append(wandb.Image(make_figure(preds[i].squeeze()), caption=f'recon'))                  


            experiment.log(
                {
                    "valid/maps": input_imgs,
                }
            )

            self.preds.clear()
            self.inputs.clear()
            self.names.clear()
    
    
    
class LogTrainPredictions_MAE(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 2, counts=True):
        super().__init__()
        self.num_samples = num_samples
        self.preds   = []
        self.inputs  = []
        self.names   = []
        self.masks   = []

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if self.ready:
            """Gather data from single batch."""
            self.preds.append(outputs["preds"])

            self.inputs.append(outputs["inputs"])

            self.names.append(outputs['input_names'])
            self.masks.append(outputs['mask'])


    def on_train_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            preds      = torch.cat(self.preds[:self.num_samples]).detach().cpu().numpy()
            inputs     = torch.cat(self.inputs[:self.num_samples]).detach().cpu().numpy()
            masks      = torch.cat(self.masks[:self.num_samples]).detach().cpu().numpy()
            masked_inputs = inputs*masks

            names      = np.concatenate(self.names[:self.num_samples])

            input_imgs        = []


            for i in range(self.num_samples):
                if inputs[i].shape[0]>1:
                    for jj in range(inputs[i].squeeze().shape[0]):
                        input_imgs.append(wandb.Image(make_figure(inputs[i].squeeze()[jj]), caption=f'{names[i,jj]} original'))
                        input_imgs.append(wandb.Image(make_figure(masked_inputs[i].squeeze()[jj]), caption=f'{names[i,jj]} inputs'))
                        input_imgs.append(wandb.Image(make_figure(preds[i].squeeze()[jj]), caption=f'{names[i,jj]} recon'))
                else:
                    input_imgs.append(wandb.Image(make_figure(inputs[i].squeeze()), caption=f'{names[i]} original'))
                    input_imgs.append(wandb.Image(make_figure(masked_inputs[i].squeeze()), caption=f'{names[i]} inputs'))
                    input_imgs.append(wandb.Image(make_figure(preds[i].squeeze()), caption=f'{names[i]} recon'))                


            experiment.log(
                {
                    "train/maps": input_imgs,
                }
            )

            self.preds.clear()
            self.inputs.clear()
            self.names.clear()
            self.masks.clear()



class LogTrainPredictions(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 2, counts=True):
        super().__init__()
        self.num_samples = num_samples
        self.counts = counts
        self.ready = True
        self.preds = []
        self.targets = []
        self.inputs = []
        self.other_inputs = []
        self.other_names = []
        self.hists = []
        self.counts_true = []
        self.counts_pred = []
        self.names = []

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])
            self.inputs.append(outputs["inputs"])
            ### yeah.... don't ask. Feel free to clean this up. 
            try:
                self.other_inputs.append(outputs["other_inputs"])
                self.other_names.append(outputs["other_names"])
                self.other=True
            except:
                self.other=False
            
            if self.counts:
                self.counts_true.append(outputs["true_count"])
                self.counts_pred.append(outputs["pred_count"])
            self.names.append(outputs['input_names'])
                

    def on_train_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            preds      = torch.cat(self.preds[:self.num_samples]).detach().cpu().numpy()
            targets    = torch.cat(self.targets[:self.num_samples]).detach().cpu().numpy()
            inputs     = torch.cat(self.inputs[:self.num_samples]).detach().cpu().numpy()
            
            try:
                count_t    = torch.cat(self.counts_true).detach().cpu().numpy()
                count_p    = torch.cat(self.counts_pred).detach().cpu().numpy()
            except:
                try:
                    count_t    = self.counts_true.detach().cpu().numpy()
                    count_p    = self.counts_pred.detach().cpu().numpy()
                except:
                    try:
                        count_t    = np.concatenate(self.counts_true)
                        count_p    = np.concatenate(self.counts_pred) 
                    except:
                        if self.counts:
                            count_t    = self.counts_true
                            count_p    = self.counts_pred
                        else:
                            pass
                        
                
            names      = np.concatenate(self.names[:self.num_samples])
            if self.other:
                other_names      = np.concatenate(self.other_names[:self.num_samples])
                other_inputs     = torch.cat(self.other_inputs[:self.num_samples]).detach().cpu().numpy()
                
            output_imgs       = []
            input_imgs        = []
            count_imgs        = []
            
            for i in range(self.num_samples):

                output_imgs.append(wandb.Image(make_figure(targets[i].squeeze()), caption=f'Target {i}'))
                if len(preds.shape)>2:
                    output_imgs.append(wandb.Image(make_figure(preds[i].squeeze()>0.5), caption=f'Prediction {i}>0.5'))
                    output_imgs.append(wandb.Image(make_figure(preds[i].squeeze()), caption=f'Prediction {i}'))

                for jj in range(inputs[i].squeeze().shape[0]):
                    input_imgs.append(wandb.Image(make_figure(inputs[i].squeeze()[jj]), caption=f'{names[i,jj]}'))
                    if self.other:
                        input_imgs.append(wandb.Image(make_figure(other_inputs[i].squeeze()[jj]), caption=f'{other_names[i,jj]}'))
                input_imgs.append(wandb.Image(make_figure(targets[i].squeeze()), caption=f'Target {i}'))

            if self.counts:
                count_imgs.append(wandb.Image(make_scatter(count_p, count_t)))

            
            # log the images as wandb Image
            if self.counts:
                experiment.log(
                    {
                        "train/target-prediction maps": output_imgs,
                        "train/input maps": input_imgs,
                        "train/counts": count_imgs
                    }
                )
            else:
                experiment.log(
                    {
                        "train/target-prediction maps": output_imgs,
                        "train/input maps": input_imgs
                    }
                )                
            self.preds.clear()
            self.targets.clear()
            self.inputs.clear()
            self.other_inputs.clear()
            self.names.clear()
            self.other_names.clear()
            if self.counts: 
                self.counts_pred.clear()
                self.counts_true.clear()


class LogValPredictions(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8, counts=True):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True
        self.preds = []
        self.targets = []
        self.counts = counts
        if self.counts:
            self.counts_true = []
            self.counts_pred = []

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])
            if self.counts:
                self.counts_true.append(outputs["true_count"])
                self.counts_pred.append(outputs["pred_count"])
            
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            preds      = torch.cat(self.preds[:self.num_samples]).detach().cpu().numpy()
            targets    = torch.cat(self.targets[:self.num_samples]).detach().cpu().numpy()

            try:
                count_t    = torch.cat(self.counts_true).detach().cpu().numpy()
                count_p    = torch.cat(self.counts_pred).detach().cpu().numpy()
            except:
                try:
                    count_t    = self.counts_true.detach().cpu().numpy()
                    count_p    = self.counts_pred.detach().cpu().numpy()
                except:
                    try:
                        count_t    = np.concatenate(self.counts_true)
                        count_p    = np.concatenate(self.counts_pred) 
                    except:
                        if self.counts:
                            count_t    = self.counts_true
                            count_p    = self.counts_pred
                        else:
                            pass

                        
            imgs       = []
  
            for i in range(self.num_samples):

                imgs.append(wandb.Image(make_figure(targets[i].squeeze()), caption=f'Target {i}'))
                if len(preds.shape)>2:
                    imgs.append(wandb.Image(make_figure(preds[i].squeeze()>0.5), caption=f'Prediction {i}>0.5'))
                    imgs.append(wandb.Image(make_figure(preds[i].squeeze()), caption=f'Prediction {i}'))
            if self.counts:
                imgs.append(wandb.Image(make_scatter(count_p, count_t)))

            # log the images as wandb Image
            
            experiment.log(
                {
                    "Validation/target-prediction maps": imgs
                }
            )
            self.preds.clear()
            self.targets.clear()
            if self.counts: 
                self.counts_pred.clear()
                self.counts_true.clear()