from pathlib import Path
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import transforms
from src.datamodules.components import transforms
import numpy as np
import pickle
import warnings
from src.datamodules.components import chips


event_dates = {}
event_dates['hokkaido']  = {'event_start_date':'20180905', 'event_end_date':'20180907'}
event_dates['indonesia'] = {'event_start_date':'20220224', 'event_end_date':'20220225'}
event_dates['kaikoura']  = {'event_start_date':'20161113', 'event_end_date':'20161114'}

def identity(x, dummy):
    return x

class Siamese_Landslide_Dataloader(Dataset):
    """
    Torch Dataset class for Siamese Change Detection Training
    
    """
    def __init__(self, 
                 data_dir, 
                 dict_dir, 
                 split, 
                 channels,
                 setting,
                 num_time_steps,
                 time_step_summary=None,
                 input_transforms=None,
                 trainsize=-1):
        
        """
        attributes:
            data_dir: data directory
            dict_dir: data summary statistics directory
            split:    one of 'train','valid', 'test'
            channels: list of strings, which channels to return
            setting: 'pretraining' or 'downstream'
            num_time_steps: int, number of time steps before and after the event
            balance_weights: weights for imbalanced dataset
            time_step_summary: numpy function name for combining time steps, None for not combining
            input_transforms: list of input transforms
        """
        
        self.datasets  = data_dir.keys()
        self.trainsize = trainsize
        self.split    = split
        assert split in ['train', 'valid', 'test']
        
        self.setting  = setting
        assert setting in ['pretraining','downstream']
        
        # decide which 
        if self.split=='train':
            if self.setting=='pretraining':
                self.split='train_pre'
            elif self.setting=='downstream':
                self.split='train_down'
                
        # get file numbers for split
        self.dataset_dicts = {}
        self.length   = 0
        for name in self.datasets:
            self.dataset_dicts[name]  = {}
            self.dataset_dicts[name]['file_nums'] = pickle.load(open(dict_dir/'split_dict_balanced_siamese_{}.pkl'.format(name), 'rb'))[self.split]
            self.dataset_dicts[name]['length']    = len(self.dataset_dicts[name]['file_nums'])
            self.dataset_dicts[name]['start_date'], self.dataset_dicts[name]['end_date'] = pickle.load(open(dict_dir/'timestep_dict_siamese_{}.pkl'.format(name),'rb'))[str(num_time_steps)]
            self.length+=self.dataset_dicts[name]['length']
            self.dataset_dicts[name]['start_date'], self.dataset_dicts[name]['end_date']
        if self.split=='train_down':
            if len(self.datasets)==1:
                if self.trainsize!=-1:
                    inds = np.random.choice(np.arange(self.length),self.trainsize, replace=False)
                    self.dataset_dicts[name]['file_nums'] = self.dataset_dicts[name]['file_nums'][inds]
                    self.dataset_dicts[name]['length'] = self.trainsize
                    self.length = self.trainsize
        
        self.input_transforms = input_transforms
        self.channels         = list(channels)


        if time_step_summary:
            self.summary          = getattr(np,time_step_summary)
        else:
            self.summary          = False
        
        self.chipsets = {}
        for name in self.datasets:
            self.chipsets[name] = chips.Chipset(data_dir[name])
        
        print(self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        count=0
        for name_ in self.datasets:
            bound = self.dataset_dicts[name_]['length']+count
            if (index<bound) and (index>=count):
                chip  = self.chipsets[name_].get_chip(self.dataset_dicts[name_]['file_nums'][index-count])
                name  = name_
            count+=bound

        # add reactivated landslides?
        if self.setting=='pretraining':
            label       = int(np.sum(np.nan_to_num(chip.sel([f'landslides']).data))>0)
        else:
            label       = np.nan_to_num(chip.sel([f'landslides']).data).astype(int)
            
        input_chip  = chip.sel(self.channels)
        
        pre         = input_chip.sel(input_chip.get_time_varnames(),
                              start_date=self.dataset_dicts[name]['start_date'],end_date=event_dates[name]['event_start_date'])
        if self.summary:
            pre = pre.apply_across_time(self.summary)
        else:
            pre = pre.apply_across_time(identity)
            
        pos         = input_chip.sel(input_chip.get_time_varnames(),
                            start_date=event_dates[name]['event_end_date'], end_date=self.dataset_dicts[name]['end_date'])
        if self.summary:
            pos= pos.apply_across_time(self.summary)
        else:
            pos= pos.apply_across_time(identity)
            
        names      = pre.get_varnames()
        
        if self.input_transforms[name]:
            pre = self.apply_transforms(pre,name).data.astype(np.float32)
            pos = self.apply_transforms(pos,name).data.astype(np.float32)

        return pre, pos, label, names, 1.

    
    def apply_transforms(self,input_chip,name):        
        chans = []
        for ch in input_chip.get_varnames():
            chan    = input_chip.sel([ch])
            ch_base = ch.split('__')[0]
            if 'dem' not in ch_base:
                for trafo in list(self.input_transforms[name]):
                    if isinstance(trafo, transforms.Log_transform):
                        if ch_base in ['vh', 'vv','topophase.cor_0']:
                            chan = chan.apply(ch,trafo,{'channel':ch_base})
                            ch+='_log'
                        else:
                            pass
                    else:
                        chan=chan.apply(ch,trafo,{'channel':ch_base})
            chans.append(chan)
        input_chip = chan.concat_static(chans)
        return input_chip
    
    
class Siamese_Landslide_Datamodule(LightningDataModule):

    def __init__(
            self,
            data_dir: str      = "data/",
            dict_dir: str      = "data_dicts/",
            batch_size: int    = 16,
            num_workers: int   = 1,
            pin_memory: bool   = False,
            input_channels     = ['vh', 'vv'],
            time_step_summary  = 'mean',
            input_transforms: list = ['Log_transform', 'Standardize'],
            num_time_steps     = 5,
            setting            = 'pretraining',
            datasets           = ['hokkaido','kaikoura'],
            trainsize          = -1
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        data_dirs = {}
        datasets  = list(datasets)
        for name in datasets:
            data_dirs[name] = Path(data_dir)/"landslides-{}".format(name)/'chips_128x128'
            
        dict_dir = Path(dict_dir)
        
        self.input_channels = list(input_channels)
        self.num_channels   = len(self.input_channels)
        ### see hokkaido_siamese_landslide_prep.ipynb
        self.stats_dicts     = {}
        for name in datasets:
            self.stats_dicts[name] = pickle.load(open(dict_dir/'stats_dict_{}.pkl'.format(name), 'rb'))
            
        input_transforms = list(input_transforms)
        if input_transforms:
            input_transforms_dict = {}
            for name_ in datasets:
                input_transforms_dict[name_] = [getattr(transforms, name)(self.stats_dicts[name_]) for name in input_transforms]
        
        self.data_train: Optional[Dataset] = Siamese_Landslide_Dataloader(data_dir=data_dirs, dict_dir=dict_dir, split="train", 
                channels=input_channels, setting=setting,num_time_steps=num_time_steps,
                time_step_summary=time_step_summary, input_transforms=input_transforms_dict, trainsize=trainsize)
        self.data_val: Optional[Dataset]   = Siamese_Landslide_Dataloader(data_dir=data_dirs, dict_dir=dict_dir, split="valid", 
                channels=input_channels, setting=setting,num_time_steps=num_time_steps,
                time_step_summary=time_step_summary, input_transforms=input_transforms_dict, trainsize=trainsize)
        self.data_test: Optional[Dataset]  = Siamese_Landslide_Dataloader(data_dir=data_dirs, dict_dir=dict_dir, split="test", 
                channels=input_channels, setting=setting,num_time_steps=num_time_steps,
                time_step_summary=time_step_summary, input_transforms=input_transforms_dict, trainsize=trainsize)
        
    @property
    def num_classes(self) -> int:
        return 1

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True,
            prefetch_factor=8
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size*8,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=8
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size*8,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=8
        )
    
    
###-------------------------------------------------------------------------------------------------------------------###
