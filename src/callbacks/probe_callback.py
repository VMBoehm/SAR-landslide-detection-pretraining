from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_warn
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torchmetrics.functional import precision_recall

from src.models.components import networks
import segmentation_models_pytorch as smp



class SSLOnlineEvaluator(Callback):  # pragma: no cover
    """Attaches a MLP for fine-tuning using the standard self-supervised protocol.
    Example::
        # your datamodule must have 2 attributes
        dm = DataModule()
        dm.num_classes = ... # the num of classes in the datamodule
        dm.name = ... # name of the datamodule (e.g. ImageNet, STL10, CIFAR10)
        # your model must have 1 attribute
        model = Model()
        model.z_dim = ... # the representation dim
        online_eval = SSLOnlineEvaluator(
            z_dim=model.z_dim
        )
    """

    def __init__(
        self,
        in_channels: int  = 4,
        out_channels: int = 1,
        optimizer = None, 
        dataset = None,
        activation = 'Sigmoid'
    ):
        """
        Args:
            z_dim: Representation dimension
            drop_p: Dropout probability
            hidden_dim: Hidden dimension for the fine-tune MLP
        """
        super().__init__()

        self.optimizer  = optimizer
        
        self.dataset    = dataset

        self.in_channels  = in_channels 
        self.out_channels = out_channels

        self.criterion = smp.losses.DiceLoss(mode="binary",from_logits=False)
        self.activation = activation
        self._recovered_callback_state: Optional[Dict[str, Any]] = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.dataset = trainer.datamodule.name

    def on_pretrain_routine_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # must move to device after setup, as during setup, pl_module is still on cpu
        self.online_evaluator = networks.DecoderHead(self.in_channels,self.out_channels,final_activation=self.activation).to(pl_module.device)

        # switch fo PL compatibility reasons
        accel = (
            trainer.accelerator_connector
            if hasattr(trainer, "accelerator_connector")
            else trainer._accelerator_connector
        )
        if accel.is_distributed:
            if accel._strategy_flag=='ddp':
                from torch.nn.parallel import DistributedDataParallel as DDP

                self.online_evaluator = DDP(self.online_evaluator, device_ids=[pl_module.device])
            elif accel._strategy_flag=='dp':
                from torch.nn.parallel import DataParallel as DP

                self.online_evaluator = DP(self.online_evaluator, device_ids=[pl_module.device])
            # elif accel._strategy_flag=='ddp_spawn':
            #     from torch.nn.parallel import DDPSpawnStrategy as DPS
            #     self.online_evaluator = DPS(self.online_evaluator, device_ids=[pl_module.device])
            else:
                rank_zero_warn(
                    "Does not support this type of distributed accelerator. The online evaluator will not sync."
                )

        self.optimizer = torch.optim.Adam(self.online_evaluator.parameters(), lr=1e-4)

        if self._recovered_callback_state is not None:
            self.online_evaluator.load_state_dict(self._recovered_callback_state["state_dict"])
            self.optimizer.load_state_dict(self._recovered_callback_state["optimizer_state"])

    def to_device(self, batch: Sequence, device: Union[str, torch.device]) -> Tuple[Tensor, Tensor]:
        # adapted
        inputs, _, _, y = batch

        # last input is for online eval
        x = inputs#[-1]
        x = x.to(device)
        y = y.to(device)

        return x, y

    # adapted
    def shared_step(
        self,
        pl_module: LightningModule,
        batch: Sequence,
    ):
        with torch.no_grad():
            with set_training(pl_module, False):
                x, y = self.to_device(batch, pl_module.device)
                representations = pl_module(x)
        
        # forward pass
        preds     = self.online_evaluator(representations)  # type: ignore[operator]

        loss      = self.criterion(preds, y)

        prec, rec = precision_recall(preds.view(-1), y.view(-1))

        return loss, prec, rec

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        loss, prec, rec = self.shared_step(pl_module, batch)

        # update finetune weights
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        #adapted
        pl_module.log("train/online_prec", prec, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("train/online_recall", rec, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("train/online_f1", 2*(rec*prec)/(rec+prec), on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("train/online_loss", loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        loss, prec, rec = self.shared_step(pl_module, batch)
        #adapted
        pl_module.log("valid/online_prec", prec, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("valid/online_recall", rec, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("valid/online_f1", 2*(rec*prec)/(rec+prec), on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("valid/online_loss", loss, on_step=True, on_epoch=False)

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> dict:
        return {"state_dict": self.online_evaluator.state_dict(), "optimizer_state": self.optimizer.state_dict()}

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, callback_state: Dict[str, Any]) -> None:
        self._recovered_callback_state = callback_state

        
class SSLOnlineEvaluator_bottleneck(Callback):  # pragma: no cover
    """Attaches a MLP for fine-tuning using the standard self-supervised protocol.
    Example::
        # your datamodule must have 2 attributes
        dm = DataModule()
        dm.num_classes = ... # the num of classes in the datamodule
        dm.name = ... # name of the datamodule (e.g. ImageNet, STL10, CIFAR10)
        # your model must have 1 attribute
        model = Model()
        model.z_dim = ... # the representation dim
        online_eval = SSLOnlineEvaluator(
            z_dim=model.z_dim
        )
    """

    def __init__(
        self,
        in_channels: int  = 4,
        out_channels: int = 1,
        optimizer = None, 
        dataset = None,
        activation = 'Sigmoid'
    ):
        """
        Args:
            z_dim: Representation dimension
            drop_p: Dropout probability
            hidden_dim: Hidden dimension for the fine-tune MLP
        """
        super().__init__()

        self.optimizer  = optimizer
        
        self.dataset    = dataset

        self.in_channels  = in_channels 
        self.out_channels = out_channels

        self.criterion = smp.losses.DiceLoss(mode="binary",from_logits=False)
        self.activation = activation
        self._recovered_callback_state: Optional[Dict[str, Any]] = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.dataset = trainer.datamodule.name

    def on_pretrain_routine_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # must move to device after setup, as during setup, pl_module is still on cpu
        self.online_evaluator = networks.Combined_Upsample_CNN_decoder(self.in_channels,self.out_channels,final_activation=self.activation).to(pl_module.device)

        # switch fo PL compatibility reasons
        accel = (
            trainer.accelerator_connector
            if hasattr(trainer, "accelerator_connector")
            else trainer._accelerator_connector
        )
        if accel.is_distributed:
            if accel._strategy_flag=='ddp':
                from torch.nn.parallel import DistributedDataParallel as DDP

                self.online_evaluator = DDP(self.online_evaluator, device_ids=[pl_module.device])
            elif accel._strategy_flag=='dp':
                from torch.nn.parallel import DataParallel as DP

                self.online_evaluator = DP(self.online_evaluator, device_ids=[pl_module.device])
            # elif accel._strategy_flag=='ddp_spawn':
            #     from torch.nn.parallel import DDPSpawnStrategy as DPS
            #     self.online_evaluator = DPS(self.online_evaluator, device_ids=[pl_module.device])
            else:
                rank_zero_warn(
                    "Does not support this type of distributed accelerator. The online evaluator will not sync."
                )

        self.optimizer = torch.optim.Adam(self.online_evaluator.parameters(), lr=1e-4)

        if self._recovered_callback_state is not None:
            self.online_evaluator.load_state_dict(self._recovered_callback_state["state_dict"])
            self.optimizer.load_state_dict(self._recovered_callback_state["optimizer_state"])

    def to_device(self, batch: Sequence, device: Union[str, torch.device]) -> Tuple[Tensor, Tensor]:
        # adapted
        inputs, _, _, y = batch

        # last input is for online eval
        x = inputs#[-1]
        x = x.to(device)
        y = y.to(device)

        return x, y

    # adapted
    def shared_step(
        self,
        pl_module: LightningModule,
        batch: Sequence,
    ):
        with torch.no_grad():
            with set_training(pl_module, False):
                x, y = self.to_device(batch, pl_module.device)
                representations = pl_module.encoder(x, torch.ones(x.shape).cuda())
        
        # forward pass

        preds     = self.online_evaluator(representations)  # type: ignore[operator]

        loss      = self.criterion(preds, y)

        prec, rec = precision_recall(preds.view(-1), y.view(-1))

        return loss, prec, rec

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        loss, prec, rec = self.shared_step(pl_module, batch)

        # update finetune weights
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        #adapted
        pl_module.log("train/bottleneck_online_prec", prec, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("train/bottleneck_online_recall", rec, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("train/bottleneck_online_f1", 2*(rec*prec)/(rec+prec), on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("train/bottleneck_online_loss", loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        loss, prec, rec = self.shared_step(pl_module, batch)
        #adapted
        pl_module.log("valid/bottleneck_online_prec", prec, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("valid/bottleneck_online_recall", rec, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("valid/bottleneck_online_f1", 2*(rec*prec)/(rec+prec), on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("valid/bottleneck_online_loss", loss, on_step=True, on_epoch=False)

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> dict:
        return {"state_dict": self.online_evaluator.state_dict(), "optimizer_state": self.optimizer.state_dict()}

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, callback_state: Dict[str, Any]) -> None:
        self._recovered_callback_state = callback_state

@contextmanager
def set_training(module: nn.Module, mode: bool):
    """Context manager to set training mode.
    When exit, recover the original training mode.
    Args:
        module: module to set training mode
        mode: whether to set training mode (True) or evaluation mode (False).
    """
    original_mode = module.training

    try:
        module.train(mode)
        yield module
    finally:
        module.train(original_mode)