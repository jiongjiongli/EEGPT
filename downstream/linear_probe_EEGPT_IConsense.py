import json
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace
import re
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


config_dict = dict(
    seed = 17,
    data_phase = "Exam",
    root_dir_path=r"E:/data/eeg_data",
    kaggle_label_root_dir_path=r"/kaggle/input/eeg-labels",
    kaggle_eeg_root_dir_path=r"/kaggle/input/eeglab-output-data",
    input_csv_dir_name="eeg_labels",
    input_eeg_csv_dir_name="eeglab_output_data",
    # output_dir_name="TODO",
    data_cache_file_name = "all_data.pkl",
    class_name = "TagHandMenuPumpTime",
    label_names = ["Unkown", "Aha"],
    eeg_column_names=["FP1", "FP2", "C3", "C4", "P7", "P8", "O1", "O2", "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4"],

    subject_ids_file_name = "subject_ids.json",
    time_param_column_index = 1,
    valid_percent = 0.1,
    test_percent = 0.1,

    seq_len=1024,
    segment_len=128 // 4,

    num_channels=16,
    emb_size=40,
    depth=6,
    num_classes=2,
    conv_kernel_size=25,
    pool_stride=15,
    pool_kernel_size=75,

    att_num_heads=10,
    forward_expansion=4,
    drop_p=0.5,
    forward_drop_p=0.5,

    linear_reduce_factor=8,

    # CPU/GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),

    batch_size = 64,
    epochs=300,
    log_every_n_steps=10,

    model_log_dir = "./logs",
    checkpoint_path = "",
    kaggle_model_log_dir = "/kaggle/working/logs",

    # Validate
    validate_interval = 100,
)

config = SimpleNamespace(**config_dict)


import random
import os
import torch
from torch import nn
import pytorch_lightning as pl

from functools import partial
import numpy as np
import random
import os
import tqdm
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(config.seed)

from Modules.models.EEGPT_mcae import EEGTransformer

from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
from utils_eval import get_metrics

use_channels_names = config.eeg_column_names

class LitEEGPTCausal(pl.LightningModule):

    def __init__(self, load_path="../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"):
        super().__init__()
        self.chans_num = config.num_channels
        # init model
        target_encoder = EEGTransformer(
            img_size=[config.num_channels, config.seq_len],
            patch_size=32*2,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.target_encoder = target_encoder
        self.chans_id       = target_encoder.prepare_chan_ids(use_channels_names)

        # -- load checkpoint
        pretrain_ckpt = torch.load(load_path)

        target_encoder_stat = {}
        for k,v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]]=v

        self.target_encoder.load_state_dict(target_encoder_stat)
        self.chan_conv       = Conv1dWithConstraint(config.num_channels, self.chans_num, 1, max_norm=1)
        self.linear_probe1   =   LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2   =   LinearWithConstraint(16*16, config.num_classes, max_norm=0.25)

        self.drop           = torch.nn.Dropout(p=0.50)

        self.loss_fn        = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity=True

    def forward(self, x):

        x = self.chan_conv(x)

        self.target_encoder.eval()

        z = self.target_encoder(x, self.chans_id.to(x))

        h = z.flatten(2)

        h = self.linear_probe1(self.drop(h))

        h = h.flatten(1)

        h = self.linear_probe2(h)

        return x, h

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y = F.one_hot(y.long(), num_classes=config.num_classes).float()

        label = y

        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==torch.argmax(label, dim=-1))*1.0).mean()
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)

        return loss


    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()
    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()

        label, y_score = [], []
        for x,y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        print(label.shape, y_score.shape)

        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, False)

        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()

        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False)

        self.running_scores["valid"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))
        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            list(self.chan_conv.parameters())+
            list(self.linear_probe1.parameters())+
            list(self.linear_probe2.parameters()),
            weight_decay=0.01)#

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }

        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )


from utils import *
data_path = "../datasets/downstream/Data/BCIC_2a_0_38HZ"
import math
# used seed: 7
seed_torch(8)
for i in range(1,10):
    all_subjects = [i]
    all_datas = []
    train_dataset,valid_dataset,test_dataset = get_data(i,data_path,1,is_few_EA = True, target_sample=1024)

    global max_epochs
    global steps_per_epoch
    global max_lr

    batch_size=64

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, num_workers=0, shuffle=False)

    max_epochs = 100
    steps_per_epoch = math.ceil(len(train_loader) )
    max_lr = 4e-4

    # init model
    model = LitEEGPTCausal(load_path=config.load_path)

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]

    trainer = pl.Trainer(accelerator='cuda',
                         devices=[0,],
                         max_epochs=max_epochs,
                         callbacks=callbacks,
                         enable_checkpointing=False,
                         logger=[pl_loggers.TensorBoardLogger('./logs/', name="EEGPT_BCIC2A_tb", version=f"subject{i}"),
                                 pl_loggers.CSVLogger('./logs/', name="EEGPT_BCIC2A_csv")])

    trainer.fit(model, train_loader, test_loader, ckpt_path='last')
