# Training in 256Hz data and 4s
import os
import math
from typing import Any, Dict, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from functools import partial
import numpy as np
import random
import copy
import torchvision
from pytorch_lightning import loggers as pl_loggers


from utils import WarmupCosineSchedule, CosineWDSchedule, grad_logger
from modeling_pretraining import EEGTransformer, EEGTransformerPredictor, EEGTransformerReconstructor, apply_mask
from configs import *
#-- use channels for model

# use_channels_names = [      'FP1', 'FPZ', 'FP2',
#                                'AF3', 'AF4',
#             'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
#         'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
#             'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
#         'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
#              'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
#                       'PO7', 'PO3', 'POZ',  'PO4', 'PO8',
#                                'O1', 'OZ', 'O2', ]
use_channels_names = config.eeg_column_names


def compute_f1(tp, fp, fn):
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calculate_f1(positive_losses, negative_losses, threshold):
    num_pos = len(positive_losses)
    num_neg = len(negative_losses)

    tp = np.sum(positive_losses > threshold)
    fn = num_pos - tp
    fp = np.sum(negative_losses > threshold)
    tn = num_neg - fp

    f1 = compute_f1(tp, fp, fn)
    return (tp, fn, fp, tn), f1


def find_best_f1(positive_losses, negative_losses, step=0.01):
    """
    Finds the threshold that maximizes the F1 score.

    Args:
        positive_losses (np.ndarray): Losses for positive samples.
        negative_losses (np.ndarray): Losses for negative samples.
        step (float): Step size for threshold search.

    Returns:
        best_threshold (float): Threshold with highest F1 score.
        best_f1 (float): Corresponding F1 score.
        best_result (tuple): (TP, FN, FP, TN) at best threshold.
        all_thresholds (np.ndarray): All thresholds tested.
        all_f1s (np.ndarray): Corresponding F1 scores.
    """
    min_threshold = np.mean([np.min(positive_losses), np.min(negative_losses)])
    max_threshold = np.mean([np.max(positive_losses), np.max(negative_losses)])
    thresholds = np.arange(min_threshold, max_threshold + step, step)

    f1s = np.zeros_like(thresholds)
    results = []

    for i, threshold in enumerate(thresholds):
        (tp, fn, fp, tn), f1 = calculate_f1(positive_losses,
                                            negative_losses, threshold)

        result = (tp, fn, fp, tn)
        results.append(result)

        f1s[i] = f1

    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    best_result = results[best_idx]

    return best_threshold, best_f1, best_result, thresholds, f1s


class LitEEGPT(pl.LightningModule):

    def __init__(self,
                 models_configs=None,
                 positive_valid_loader=None,
                 negative_valid_loader=None,
                 positive_test_loader=None,
                 negative_test_loader=None,
                 USE_LOSS_A=True, USE_LN=True, USE_SKIP=True):
        super().__init__()
        self.USE_LOSS_A = USE_LOSS_A
        self.USE_LN     = USE_LN
        self.USE_SKIP   = USE_SKIP

        self.positive_valid_loader = positive_valid_loader
        self.negative_valid_loader = negative_valid_loader
        self.positive_test_loader  = positive_test_loader
        self.negative_test_loader  = negative_test_loader

        encoder = EEGTransformer(
            # img_size=[58, 256*4],
            img_size=[config.num_channels, config.seq_len],
            patch_size=32*2,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['encoder'])

        predictor = EEGTransformerPredictor(
            num_patches=encoder.num_patches,
            use_part_pred=True,################
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['predictor'])

        reconstructor = EEGTransformerReconstructor(
            num_patches=encoder.num_patches,
            patch_size=32*2,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['reconstructor'])

        target_encoder = copy.deepcopy(encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False

        self.encoder        = encoder
        self.target_encoder = target_encoder
        self.predictor      = predictor
        self.reconstructor  = reconstructor
        self.chans_id       = encoder.prepare_chan_ids(use_channels_names)

        self.loss_fn        = torch.nn.MSELoss()
        self.pred_loss_fn   = torch.nn.MSELoss(reduction='none')

        self.is_sanity=True

        self.train_loss1_epoch = []
        self.train_loss2_epoch = []
        self.train_loss_epoch = []

        self.valid_loss1_epoch = []
        self.valid_loss2_epoch = []
        self.valid_loss_epoch = []

        self.valid_preds = []
        self.valid_targets = []
        self.test_preds = []
        self.test_targets = []

    def make_masks(self, num_patchs, mC_x=5, p_n_y=0.5, p_c_y=0.2):

        C, N = num_patchs

        while True:
            mask_x = []# mN, mC
            mask_y = []
            mask_y_bx = []
            for i in range(N):
                c_idx = torch.randperm(C) + i*C
                if random.random()>p_n_y:
                    mask_x.append(c_idx[:mC_x])
                    mask_y_bx.append(c_idx[mC_x:])
                else:
                    mask_y.append(c_idx)
            if len(mask_x)==0: continue
            if len(mask_y_bx)==0: continue
            mask_y_bx = torch.cat(mask_y_bx, dim=0)
            mask_y_bx = mask_y_bx[torch.rand(mask_y_bx.shape)<p_c_y]
            if len(mask_y_bx)==0: continue
            break

        return torch.stack(mask_x, dim=0), torch.cat(mask_y+[mask_y_bx], dim=0)

    def forward_target(self, x, mask_y):
        with torch.no_grad():
            h = self.target_encoder(x, self.chans_id.to(x))
            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            C, N = self.encoder.num_patches
            assert x.shape[-1]%N==0 and x.shape[-2]%C == 0
            block_size_c, block_size_n = x.shape[-2]//C, x.shape[-1]//N
            x = x.view(x.shape[0], C, block_size_c, N, block_size_n)
            # 将维度重新排列以使分块沿着通道轴和空间轴
            x = x.permute(0, 3, 1, 2, 4).contiguous() # B, N, C, bc, bn
            x = x.view(x.shape[0], C, N, block_size_c * block_size_n)
            y = apply_mask(mask_y.to(x.device), x)
            if self.USE_LN:
                y = F.layer_norm(y, (y.size(-1),))
            return h, y

    def forward_context(self, x, mask_x, mask_y):
        z = self.encoder(x, self.chans_id.to(x), mask_x=mask_x)
        z, comb_z = self.predictor(z, mask_x=mask_x)
        if not self.USE_SKIP:
            comb_z = z
        r = self.reconstructor(comb_z, self.chans_id.to(x), mask_y=mask_y)
        return z, r

    def training_step(self, batch, batch_idx):
        x, _ = batch
        mask_x, mask_y = self.make_masks(self.encoder.num_patches)
        h, y = self.forward_target(x, mask_y)
        z, r = self.forward_context(x, mask_x, mask_y)
        loss1 = self.loss_fn(h, z)
        loss2 = self.loss_fn(y, r)
        if self.USE_LOSS_A:
            loss  = loss1 + loss2
        else:
            loss  = loss2

        # -- Contrast
        self.log('train_loss1', loss1, on_epoch=True, on_step=False, sync_dist=True)
        # -- Reconstruct
        self.log('train_loss2', loss2, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_loss' , loss , on_epoch=True, on_step=False, sync_dist=True)

        self.train_loss1_epoch.append(loss1.clone().detach())
        self.train_loss2_epoch.append(loss2.clone().detach())
        self.train_loss_epoch.append(loss.clone().detach())

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        mask_x, mask_y = self.make_masks(self.encoder.num_patches)
        h, y = self.forward_target(x, mask_y)
        z, r = self.forward_context(x, mask_x, mask_y)
        loss1 = self.loss_fn(h, z)
        loss2 = self.loss_fn(y, r)
        if self.USE_LOSS_A:
            loss  = loss1 + loss2
        else:
            loss  = loss2

        # -- Contrast
        self.log('valid_loss1', loss1, on_epoch=True, on_step=False, sync_dist=True)
        # -- Reconstruct
        self.log('valid_loss2', loss2, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_loss' , loss , on_epoch=True, on_step=False, sync_dist=True)

        self.valid_loss1_epoch.append(loss1.clone().detach())
        self.valid_loss2_epoch.append(loss2.clone().detach())
        self.valid_loss_epoch.append(loss.clone().detach())

        return loss

    def predict_loss(self, batch):
        x, _ = batch
        x = x.to(config.device)
        mask_x, mask_y = self.make_masks(self.encoder.num_patches)
        h, y = self.forward_target(x, mask_y)
        z, r = self.forward_context(x, mask_x, mask_y)
        loss1 = self.pred_loss_fn(h, z)
        loss2 = self.pred_loss_fn(y, r)

        loss1 = loss1.view(loss1.shape[0], -1).mean(dim=1).cpu().numpy()
        loss2 = loss2.view(loss2.shape[0], -1).mean(dim=1).cpu().numpy()

        if self.USE_LOSS_A:
            loss  = loss1 + loss2
        else:
            loss  = loss2

        return loss, loss1, loss2

    @torch.no_grad()
    def evaluate(self, positive_loader, negative_loader, threshold=None):
        positive_pred_loss = []
        negative_pred_loss = []


        for batch in positive_loader:
            loss, loss1, loss2 = self.predict_loss(batch)
            positive_pred_loss.append(loss2)

        for batch in negative_loader:
            loss, loss1, loss2 = self.predict_loss(batch)
            negative_pred_loss.append(loss2)

        positive_pred_losses = np.concat(positive_pred_loss)
        negative_pred_losses = np.concat(negative_pred_loss)

        if threshold:
            f1 = calculate_f1(positive_pred_losses, negative_pred_losses, threshold)
            return threshold, f1

        (best_threshold, best_f1, best_result, thresholds, f1s) = find_best_f1(positive_pred_losses,
                                                                               negative_pred_losses)
        return best_threshold, test_f1


    def on_train_batch_start(self, batch: Any, batch_idx: int):
        self.wd_scheduler.step()

        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        grad_stats = grad_logger(self.encoder.named_parameters())
        self.log('grad_stats.first_layer', grad_stats.first_layer, on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.last_layer', grad_stats.last_layer, on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.min', grad_stats.min, on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.max', grad_stats.max, on_epoch=True, on_step=False, sync_dist=True)

        # momentum update of target encoder
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_train_epoch_start(self) -> None:
        self.train_loss1_epoch = []
        self.train_loss2_epoch = []
        self.train_loss_epoch = []

        return super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        train_loss1_epoch = torch.mean(torch.stack(self.train_loss1_epoch))
        train_loss2_epoch= torch.mean(torch.stack(self.train_loss2_epoch))
        train_loss_epoch= torch.mean(torch.stack(self.train_loss_epoch))

        self.log_dict({
            "train/Loss": train_loss_epoch,
            "train/L1":   train_loss1_epoch,
            "train/L2":   train_loss2_epoch,
        }, prog_bar=True)

        self.train_loss1_epoch.clear()
        self.train_loss2_epoch.clear()
        self.train_loss_epoch.clear()

        return super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        self.valid_loss1_epoch = []
        self.valid_loss2_epoch = []
        self.valid_loss_epoch = []

        self.valid_preds = []
        self.valid_targets = []

        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()

         # self.positive_valid_loader and self.negative_valid_loader
        valid_threshold, valid_f1 = self.evaluate(self.positive_valid_loader,
                                                  self.negative_valid_loader)
        test_threshold, test_f1  = self.evaluate(self.positive_test_loader,
                                                 self.negative_test_loader,
                                                 threshold=valid_threshold)

        valid_loss1_epoch= torch.mean(torch.stack(self.valid_loss1_epoch))
        valid_loss2_epoch= torch.mean(torch.stack(self.valid_loss2_epoch))
        valid_loss_epoch= torch.mean(torch.stack(self.valid_loss_epoch))

        self.log_dict({
            "val/F1":     valid_f1,
            "test/F1":    test_f1,
            "val/Loss":   valid_loss_epoch,
            "val/L1":     valid_loss1_epoch,
            "val/L2":     valid_loss2_epoch,
        }, prog_bar=True)

        self.log("val_F1", valid_f1)

        self.valid_loss1_epoch.clear()
        self.valid_loss2_epoch.clear()
        self.valid_loss_epoch.clear()

        self.valid_preds.clear()
        self.valid_targets.clear()

        return super().on_validation_epoch_end()


    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        res = super().on_load_checkpoint(checkpoint)

        self.configure_optimizers()
        return res

    def configure_optimizers(self):

        param_groups = [
            {
                'params': (p for n, p in self.encoder.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.predictor.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.reconstructor.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.encoder.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in self.predictor.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in self.reconstructor.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=6e-5)

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch,
                                                           epochs=max_epochs,
                                                           div_factor = 2,
                                                           final_div_factor=8,
                                                           pct_start = 0.2 ,
                                                           )
        # lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'epoch',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val/loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
        self.wd_scheduler = CosineWDSchedule(
                            optimizer,
                            ref_wd=1e-6,
                            final_wd=1e-6,
                            T_max=int(max_epochs*steps_per_epoch))
        ema = [0.996,1.0]
        self.momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(steps_per_epoch*max_epochs)
                          for i in range(int(steps_per_epoch*max_epochs)+1))
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )


#-- modeling
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

