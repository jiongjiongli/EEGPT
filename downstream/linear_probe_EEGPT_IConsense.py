import json
import math
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace
import re
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
from sklearn.model_selection import StratifiedKFold


from iconsense_finetune_dataset import SeqDatasetGenerator, get_dir_path

from Modules.models.EEGPT_mcae import EEGTransformer

from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
# from utils_eval import get_metrics

torch.set_float32_matmul_precision('high')


config_dict = dict(
    data_phase = "Exam",
    seed = 17,

    input_root_dir_path=(r"E:/data/eeg_data",
                         r"/home/iconsense/Desktop/jiongjiong_li/data/eeg_data",
                         r"/kaggle/input"),

    output_root_dir_path = (r"E:/data/eeg_data",
                            r"/home/iconsense/Desktop/jiongjiong_li/data/eeg_data",
                            r"/kaggle/working"),

    input_seq_splits_dir_name="seq_splits",

    input_eeg_dir_name=("eeglab_output_data",
                        "/kaggle/input/eeglab-output-data/eeglab_output_data"),

    output_dir_name="finetune_dataset",

    data_cache_file_name = "all_data.pkl",
    class_name = "TagHandMenuPumpTime",
    label_names = ["Unkown", "Aha"],
    eeg_column_names=["FP1", "FP2", "C3", "C4", "P7", "P8", "O1", "O2", "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4"],

    time_param_column_index = 1,
    valid_percent = 0.1,
    test_percent = 0.1,

    seq_len=1024,
    num_channels=16,
    num_classes=2,
    negative_label_index=0,
    trainval=False,
    n_splits = 5,

    task = "binary", # "binary" or "multiclass"

    # CPU/GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),

    # load_path = "/kaggle/working/EEGPT/pretrain/checkpoints/last.ckpt",
    load_path = ("/home/iconsense/Desktop/jiongjiong_li/data/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt",
                 "/kaggle/input/eegpt-pretrained-model/pytorch/default/1/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"),

    max_lr = 4e-5,

    batch_size = 64,
    epochs=50,
    n_folds = 5,
)

config = SimpleNamespace(**config_dict)

dataset_generator = SeqDatasetGenerator(config)
datasets_infos = dataset_generator.generate()


import random
import os
import torch
from torch import nn
import pytorch_lightning as pl

from functools import partial
import numpy as np
import random
import os
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

use_channels_names = config.eeg_column_names

class LitEEGPTCausal(pl.LightningModule):

    def __init__(self,
                 load_path="../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt",
                 steps_per_epoch=None,
                 task=None,
                 num_train_samples=None):
        super().__init__()
        self.chans_num = config.num_channels
        self.steps_per_epoch=steps_per_epoch
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
        load_path = get_dir_path(load_path, raise_error=False)

        if load_path:
            pretrain_ckpt = torch.load(load_path, weights_only=False)

            target_encoder_stat = {}
            for k,v in pretrain_ckpt['state_dict'].items():
                if k.startswith("target_encoder."):
                    target_encoder_stat[k[15:]]=v

            self.target_encoder.load_state_dict(target_encoder_stat)

        self.chan_conv       = Conv1dWithConstraint(config.num_channels, self.chans_num, 1, max_norm=1)
        self.linear_probe1   =   LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2   =   LinearWithConstraint(16*16, 1, max_norm=1)

        self.drop           = torch.nn.Dropout(p=0.50)

        pos_weight = torch.tensor([num_train_samples["num_negative"] / num_train_samples["num_positive"]])

        self.loss_fn        = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) if config.task == "binary" else torch.nn.CrossEntropyLoss()
        # self.running_scores = {"train":[], "val":[], "test":[]}
        self.is_sanity=True

        # Metrics (compute_on_step=False means accumulate across entire epoch)
        if task == "binary":
            # Binary classification metrics
            self.val_accuracy = torchmetrics.Accuracy(task=task)
            self.val_balanced_accuracy = torchmetrics.classification.BinaryAccuracy()
            self.val_cohen_kappa = torchmetrics.classification.BinaryCohenKappa()

            self.val_precision = torchmetrics.Precision(task=task)
            self.val_recall = torchmetrics.Recall(task=task)

            self.val_f1_macro = torchmetrics.F1Score(task=task)
            self.val_f1_micro = torchmetrics.F1Score(task=task)
            self.val_f1_weighted = torchmetrics.F1Score(task=task)

            self.val_cm = torchmetrics.ConfusionMatrix(task=task)
        else:
            assert task == "multiclass", task
            self.val_accuracy = torchmetrics.Accuracy(task=task, num_classes=config.num_classes)
            self.val_balanced_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=config.num_classes)
            self.val_cohen_kappa = torchmetrics.classification.MulticlassCohenKappa(num_classes=config.num_classes)

            self.val_precision = torchmetrics.Precision(task=task, num_classes=config.num_classes)
            self.val_recall = torchmetrics.Recall(task=task, num_classes=config.num_classes)

            self.val_f1_macro = torchmetrics.F1Score(task=task, num_classes=config.num_classes, average="macro")
            self.val_f1_micro = torchmetrics.F1Score(task=task, num_classes=config.num_classes, average="micro")
            self.val_f1_weighted = torchmetrics.F1Score(task=task, num_classes=config.num_classes, average="weighted")

            self.val_cm = torchmetrics.ConfusionMatrix(task=task, num_classes=config.num_classes)

        self.preds_epoch = []
        self.targets_epoch = []
        self.test_preds = []
        self.test_targets = []

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

        x, logit = self.forward(x)

        if config.task == "binary":
            logit = logit.squeeze(-1)
            y = y.float()
        else:
            y = F.one_hot(y.long(), num_classes=config.num_classes).float()

        label = y

        loss = self.loss_fn(logit, label)

        if config.task == "binary":
            probs = torch.sigmoid(logit)
            preds = (probs > 0.5).int()

            accuracy = ((preds == label.int())*1.0).mean()
        else:
            preds = torch.argmax(logit, dim=-1)

            accuracy = ((preds==torch.argmax(label, dim=-1))*1.0).mean()
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch

        x, logit = self.forward(x)

        if config.task == "binary":
            logit = logit.squeeze(-1)
            label = y.float()
        else:
            label = y.long()

        loss = self.loss_fn(logit, label)

        if config.task == "binary":
            probs = torch.sigmoid(logit)
            preds = (probs > 0.5).int()
        else:
            preds = torch.argmax(logit, dim=-1)

        accuracy = ((preds==label.int())*1.0).mean()
        # Logging to TensorBoard by default
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        self.log('val_acc', accuracy, on_epoch=True, on_step=False)

        # self.running_scores["valid"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))
        self.preds_epoch.append(preds.clone().detach())
        self.targets_epoch.append(label.clone().detach().int())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        x, logit = self.forward(x)

        if config.task == "binary":
            logit = logit.squeeze(-1)
            label = y.float()
        else:
            label = y.long()

        if config.task == "binary":
            probs = torch.sigmoid(logit)
            preds = (probs > 0.5).int()
        else:
            preds = torch.argmax(logit, dim=-1)

        self.test_preds.append(preds.clone().detach())
        self.test_targets.append(y.clone().detach().int())


    def on_validation_epoch_start(self) -> None:
        # self.running_scores["valid"]=[]
        # device = self.device
        # self.val_accuracy.to(device)
        # self.val_balanced_accuracy.to(device)
        # self.val_cohen_kappa.to(device)
        # self.val_f1_macro.to(device)
        # self.val_f1_micro.to(device)
        # self.val_f1_weighted.to(device)

        self.preds_epoch = []
        self.targets_epoch = []

        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()

        # label, y_score = [], []
        # for x,y in self.running_scores["valid"]:
        #     label.append(x)
        #     y_score.append(y)
        # label = torch.cat(label, dim=0)
        # y_score = torch.cat(y_score, dim=0)
        # print(label.shape, y_score.shape)

        # metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
        # results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, False)

        # for key, value in results.items():
        #     self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)

        preds = torch.cat(self.preds_epoch)
        targets = torch.cat(self.targets_epoch)

        acc = self.val_accuracy(preds, targets)
        bal_acc = self.val_balanced_accuracy(preds, targets)
        kappa = self.val_cohen_kappa(preds, targets)
        f1_macro = self.val_f1_macro(preds, targets)
        f1_micro = self.val_f1_micro(preds, targets)
        f1_weighted = self.val_f1_weighted(preds, targets)

        prec = self.val_precision(preds, targets)
        rec = self.val_recall(preds, targets)

        # Compute confusion matrix
        cm = self.val_cm(preds, targets)

        tn, fp, fn, tp = cm.flatten().tolist()
        self.log_dict({
            "val/TN": tn,
            "val/FP": fp,
            "val/FN": fn,
            "val/TP": tp,
            "val/PR": prec,
            "val/RC": rec,
            "val/ACC": acc,
            "val/F1": f1_macro,
        }, prog_bar=True)

        # self.log("val/balanced_accuracy", bal_acc)
        self.log("val/cohen_kappa", kappa)
        # self.log("val/f1_macro", f1_macro)
        # self.log("val/f1_micro", f1_micro)
        # self.log("val/f1_weighted", f1_weighted)

        self.preds_epoch.clear()
        self.targets_epoch.clear()

        return super().on_validation_epoch_end()

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_targets = []

        return super().on_test_epoch_start()

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds)
        targets = torch.cat(self.test_targets)

        # Log metrics
        acc = self.val_accuracy(preds, targets)
        bal_acc = self.val_balanced_accuracy(preds, targets)
        kappa = self.val_cohen_kappa(preds, targets)
        f1_macro = self.val_f1_macro(preds, targets)
        f1_micro = self.val_f1_micro(preds, targets)
        f1_weighted = self.val_f1_weighted(preds, targets)

        # Compute confusion matrix
        cm = self.val_cm(preds, targets)

        tn, fp, fn, tp = cm.flatten().tolist()

        prec = self.val_precision(preds, targets)
        rec = self.val_recall(preds, targets)

        self.log_dict({
            "test/TN": tn,
            "test/FP": fp,
            "test/FN": fn,
            "test/TP": tp,
            "test/PR": prec,
            "test/RC": rec,
            "test/ACC": acc,
            "test/F1": f1_macro,
        }, prog_bar=True)

        # self.log("test/balanced_accuracy", bal_acc)
        self.log("test/cohen_kappa", kappa)
        # self.log("test/f1_macro", f1_macro)
        # self.log("test/f1_micro", f1_micro)
        # self.log("test/f1_weighted", f1_weighted)

        self.test_preds.clear()
        self.test_targets.clear()

        return super().on_test_epoch_end()

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            list(self.chan_conv.parameters())+
            list(self.linear_probe1.parameters())+
            list(self.linear_probe2.parameters()),
            weight_decay=0.01)#

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.max_lr, steps_per_epoch=self.steps_per_epoch, epochs=config.epochs, pct_start=0.2)
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

for datasets_info in datasets_infos:
    fold_idx = datasets_info["fold_idx"]
    datasets = datasets_info["datasets"]
    num_split_neg_pos_samples = dataset_generator.get_datasets_stat(fold_idx,
                                                                    datasets)

    seed_torch(config.seed)

    train_loader = DataLoader(datasets["train"],
                              batch_size=config.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(datasets["val"],
                              batch_size=config.batch_size,
                              shuffle=False)
    test_loader  = DataLoader(datasets["test"],
                              batch_size=config.batch_size,
                              shuffle=False)

    steps_per_epoch = math.ceil(len(train_loader))

    # init model
    model = LitEEGPTCausal(load_path=config.load_path,
                           steps_per_epoch=steps_per_epoch,
                           task=config.task,
                           num_train_samples=num_split_neg_pos_samples["train"])

    checkpoint_cb = ModelCheckpoint(
        save_top_k=1,                      # save only the best checkpoint
        monitor='val/F1',               # metric to monitor (make sure it's logged)
        mode='min',                       # 'min' if lower is better, 'max' otherwise
        save_last=True,                   # also save the last checkpoint
        dirpath='./iconsense_checkpoints/',         # directory to save checkpoints
        filename= f"EEGPT_{fold_idx}" + '-{epoch:03d}-{val_loss:.4f}'  # naming pattern
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor, checkpoint_cb]

    trainer = pl.Trainer(accelerator='cuda',
                         devices=[0,],
                         max_epochs=config.epochs,
                         callbacks=callbacks,
                         log_every_n_steps=steps_per_epoch,
                         logger=[pl_loggers.TensorBoardLogger('./logs/', name="EEGPT_ICONSENSE_tb", version=f"fold_idx-{fold_idx}_pos_weight-max_lr-{config.max_lr:.6f}"),
                                 pl_loggers.CSVLogger('./logs/', name="EEGPT_ICONSENSE_csv")])

    trainer.fit(model, train_loader, valid_loader)
    del model

    print("Test on pretrain:")

    model = LitEEGPTCausal(load_path=config.load_path,
                           steps_per_epoch=steps_per_epoch,
                           task=config.task,
                           num_train_samples=num_split_neg_pos_samples["train"])
    trainer.test(model, dataloaders=test_loader)
    del model

    # Get best checkpoint path
    print("Test on best model checkpoint:", checkpoint_cb.best_model_path)

    model = LitEEGPTCausal(load_path=checkpoint_cb.best_model_path,
                           steps_per_epoch=steps_per_epoch,
                           task=config.task,
                           num_train_samples=num_split_neg_pos_samples["train"])
    trainer.test(model, dataloaders=test_loader)
    del model

    # Optional: get last checkpoint
    print("Test on last checkpoint:", checkpoint_cb.last_model_path)

    model = LitEEGPTCausal(load_path=checkpoint_cb.last_model_path,
                           steps_per_epoch=steps_per_epoch,
                           task=config.task,
                           num_train_samples=num_split_neg_pos_samples["train"])
    trainer.test(model, dataloaders=test_loader)
    del model
