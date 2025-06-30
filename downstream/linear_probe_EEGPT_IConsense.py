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
from pytorch_lightning.callbacks import ModelCheckpoint


config_dict = dict(
    seed = 17,
    data_phase = "Exam",

    root_dir_path=r"E:/data/eeg_data",
    kaggle_eeg_root_dir_path=r"/kaggle/input/eeglab-output-data",
    kaggle_working_dir_path=r"/kaggle/working",

    input_csv_dir_name="seq_labels",
    input_eeg_csv_dir_name="eeglab_output_data",

    pretrain_input_csv_dir_name = "pretrain",
    subject_ids_file_name = "subject_ids.json",
    # output_dir_name="TODO",

    data_cache_file_name = "all_data.pkl",
    class_name = "TagHandMenuPumpTime",
    label_names = ["Unkown", "Aha"],
    eeg_column_names=["FP1", "FP2", "C3", "C4", "P7", "P8", "O1", "O2", "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4"],

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

    load_path = "/kaggle/working/EEGPT/pretrain/checkpoints/last.ckpt",

    max_lr = 4e-4,

    batch_size = 64,
    epochs=30,

    model_log_dir = "./logs",
    kaggle_model_log_dir = "/kaggle/working/logs",

    # Validate
    validate_interval = 100,
)

config = SimpleNamespace(**config_dict)


def get_subject_id(session_dir_path):
    match_result = re.match('^OpenBCISession_Subject[_ ]([0-9]+)[_ ](.*)$', session_dir_path.stem)

    subject_id = int(match_result.group(1))
    return subject_id


class DataGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self):
        config = self.config

        root_dir_path = Path(config.root_dir_path)
        kaggle_working_dir_path = Path(config.kaggle_working_dir_path)
        kaggle_eeg_root_dir_path = Path(config.kaggle_eeg_root_dir_path)

        if kaggle_working_dir_path.exists():
            csv_dir_path = kaggle_working_dir_path / config.input_csv_dir_name
        else:
            csv_dir_path = root_dir_path / config.data_phase / config.input_csv_dir_name

        if kaggle_eeg_root_dir_path.exists():
            eeg_csv_dir_path = kaggle_eeg_root_dir_path / config.input_eeg_csv_dir_name
        else:
            eeg_csv_dir_path = root_dir_path / config.data_phase / config.input_eeg_csv_dir_name

        eeg_data = self.read_eeg(eeg_csv_dir_path)

        if kaggle_working_dir_path.exists():
            pretrain_dir_path = kaggle_working_dir_path / config.pretrain_input_csv_dir_name
        else:
            pretrain_dir_path = root_dir_path / config.data_phase / config.pretrain_input_csv_dir_name

        pretrain_file_path = pretrain_dir_path / config.subject_ids_file_name

        with open(pretrain_file_path, "r") as file_stream:
            subject_id_splits = json.load(file_stream)

        print(subject_id_splits)
        subject_id_to_split = {}

        for split, subject_ids in subject_id_splits.items():
            for subject_id in subject_ids:
                assert subject_id not in subject_id_to_split, subject_id
                subject_id_to_split[subject_id] = split

        all_data_infos = {}

        csv_session_dirs = list(csv_dir_path.glob("OpenBCISession_Subject_*"))
        csv_session_dirs.sort(key=get_subject_id)

        pbar = tqdm(csv_session_dirs)

        for csv_session_dir in pbar:
            subject_id = get_subject_id(csv_session_dir)
            csv_file_paths = list(csv_session_dir.glob("*_positive.csv"))

            assert len(csv_file_paths) == 1, csv_session_dir

            csv_file_path = csv_file_paths[-1]
            pbar.set_description(f"Processing {csv_file_path}")

            positive_events_df = pd.read_csv(csv_file_path)

            negative_csv_file_paths = list(csv_session_dir.glob("*_negative.csv"))

            assert len(negative_csv_file_paths) == 1, csv_session_dir

            negative_csv_file_path = negative_csv_file_paths[-1]

            negative_events_df = pd.read_csv(negative_csv_file_path)

            assert subject_id in subject_id_to_split
            split = subject_id_to_split[subject_id]

            assert subject_id in eeg_data
            eeg_data_df = eeg_data[subject_id]

            event_infos = [
                {
                    "event_df": positive_events_df,
                    "label_index": 1,
                },
                {
                    "event_df": negative_events_df
                    "label_index": 0,
                }
            ]

            for event_info in event_infos:
                event_df = event_info["event_df"]
                label_index = event_info["label_index"]

                # seq shape: [num_channels, seq_len]
                seqs = self.get_seqs(eeg_data_df, event_df)

                all_data_infos.setdefault(split, [])
                data_info = {"seqs": seqs, "label_index": label_index}
                all_data_infos[split].append(data_info)

        return all_data_infos

    def read_eeg(self, eeg_csv_dir_path):
        eeg_data = {}

        csv_session_dirs = list(eeg_csv_dir_path.glob("OpenBCISession_Subject_*"))
        csv_session_dirs.sort(key=get_subject_id)

        pbar = tqdm(csv_session_dirs)

        for csv_session_dir in pbar:
            subject_id = get_subject_id(csv_session_dir)
            csv_file_paths = list(csv_session_dir.glob("*_ica.csv"))

            assert len(csv_file_paths) == 1, csv_session_dir

            csv_file_path = csv_file_paths[-1]
            pbar.set_description(f"Processing {csv_file_path}")

            eeg_data_df = pd.read_csv(csv_file_path, usecols=config.eeg_column_names)

            assert subject_id not in eeg_data, f"{subject_id}"
            eeg_data[subject_id] = eeg_data_df

        return eeg_data

    def get_seqs(self, eeg_data_df, event_df):
        seqs = []

        for _, row in event_df.iterrows():
            start = row['DataStartIndex']
            end = row['DataEndIndex']
            # shape: [seq_len, num_channels]
            seq = eeg_data_df.iloc[start:end].to_numpy()
            # shape: [num_channels, seq_len]
            seq = np.transpose(seq, (0, 1))
            seqs.append(seq)

        return seqs

class IconsenseDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Shape: [num_channels, seq_len]
        sample = self.samples[idx]
        seq, label_index = sample
        return torch.tensor(seq, dtype=torch.float32), label_index


data_generator = DataGenerator(config)
all_data_infos = data_generator.generate()

datasets = []

for split, data_infos in all_data_infos.items():
    samples = []

    for data_info in data_infos:
        seqs = data_info["seqs"]
        label_index = data_info["label_index"]

        for seq in seqs:
            sample = (seq, label_index)
            samples.append(sample)

    datasets[split] = IconsenseDataset(samples)


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

    def __init__(self,
                 load_path="../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt",
                 steps_per_epoch=None):
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
        self.running_scores = {"train":[], "val":[], "test":[]}
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


data_path = "../datasets/downstream/Data/BCIC_2a_0_38HZ"
import math
# used seed: 7
seed_torch(config.seed)

train_loader = DataLoader(datasets["train"], batch_size=config.batch_size, num_workers=0, shuffle=True)
valid_loader = DataLoader(datasets["valid"], batch_size=config.batch_size, num_workers=0, shuffle=False)
test_loader  = DataLoader(datasets["test"],  batch_size=config.batch_size, num_workers=0, shuffle=False)

steps_per_epoch = math.ceil(len(train_loader) )

# init model
model = LitEEGPTCausal(load_path=config.load_path,
                       steps_per_epoch=steps_per_epoch)

checkpoint_cb = ModelCheckpoint(
    save_top_k=1,                      # save only the best checkpoint
    monitor='valid_loss',               # metric to monitor (make sure it's logged)
    mode='min',                       # 'min' if lower is better, 'max' otherwise
    save_last=True,                   # also save the last checkpoint
    dirpath='./iconsense_checkpoints/',         # directory to save checkpoints
    filename='EEGPT_{epoch:03d}-{val_loss:.4f}'  # naming pattern
)

lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
callbacks = [lr_monitor, checkpoint_cb]

trainer = pl.Trainer(accelerator='cuda',
                     devices=[0,],
                     max_epochs=config.epochs,
                     callbacks=callbacks,
                     log_every_n_steps=steps_per_epoch,
                     logger=[pl_loggers.TensorBoardLogger('./logs/', name="EEGPT_BCIC2A_tb", version=f"subject{i}"),
                             pl_loggers.CSVLogger('./logs/', name="EEGPT_BCIC2A_csv")])

trainer.fit(model, train_loader, valid_loader, ckpt_path='last')

# Get best checkpoint path
print("Best model checkpoint path:", checkpoint_cb.best_model_path)

# Optional: get last checkpoint
print("Last checkpoint path:", checkpoint_cb.last_model_path)
