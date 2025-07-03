
import torch
import torchvision
import math
import random

def load_fn(x):
    x = torch.load(x)

    window_length = 4*256
    data_length = x.shape[1]

    # Calculate the maximum starting index for the windows
    max_start_index = data_length - window_length

    # Generate random indices
    if max_start_index>0:
        index = random.randint(0, max_start_index)
        x = x[:, index:index+window_length]
    x = x.to(torch.float)
    return x


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


from iconsense_pretrain_dataset import get_sub_based_dataset, get_seq_datasets


config_dict = dict(
    data_phase = "Exam",

    seed = 17,

    # seq based:

    input_root_dir_path=(r"E:/data/eeg_data",
                         r"/home/iconsense/Desktop/jiongjiong_li/data/eeg_data",
                         r"/kaggle/input"),

    output_root_dir_path = (r"E:/data/eeg_data",
                            r"/home/iconsense/Desktop/jiongjiong_li/data/eeg_data",
                            r"/kaggle/working"),

    input_seq_splits_dir_name="seq_splits",

    input_eeg_dir_name=("eeglab_output_data",
                        "/kaggle/input/eeglab-output-data/eeglab_output_data"),

    eeg_column_names=["FP1", "FP2", "C3", "C4", "P7", "P8", "O1", "O2", "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4"],

    negative_label_index=0,
    positive_label_index=1,
    trainval=False,

    # subject based:

    root_dir_path=r"E:/data/eeg_data",

    kaggle_label_root_dir_path=r"/kaggle/input/eeg-labels",
    input_csv_dir_name="eeg_labels",

    kaggle_eeg_root_dir_path=r"/kaggle/input/eeglab-output-data",

    input_eeg_csv_dir_name="eeglab_output_data",

    kaggle_working_dir_path = "/kaggle/working",
    output_dir_name="pretrain",

    data_cache_file_name = "all_data.pkl",
    class_name = "TagHandMenuPumpTime",
    label_names = ["Unkown", "Aha"],

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

    model_log_dir = "./logs",
    kaggle_model_log_dir = "/kaggle/working/logs",

    # Validate
    validate_interval = 100,
    max_lr = 5e-4,
)

config = SimpleNamespace(**config_dict)

# datasets = get_sub_based_dataset(config)
datasets = get_seq_datasets(config)
train_dataset = datasets["train"][config.negative_label_index]
valid_dataset = datasets["val"][config.negative_label_index]
test_dataset = datasets["test"][config.negative_label_index]

positive_valid_dataset = datasets["val"][config.positive_label_index]
negative_valid_dataset = datasets["val"][config.negative_label_index]

positive_test_dataset = datasets["test"][config.positive_label_index]
negative_test_dataset = datasets["test"][config.negative_label_index]

max_epochs = config.epochs
max_lr = config.max_lr
batch_size=config.batch_size
devices=[0]


train_loader = DataLoader(train_dataset,
                          batch_size=config.batch_size,
                          shuffle=True)
valid_loader = DataLoader(valid_dataset,
                         batch_size=config.batch_size,
                         shuffle=False)
test_loader = DataLoader(test_dataset,
                         batch_size=config.batch_size,
                         shuffle=False)


positive_valid_loader = DataLoader(positive_valid_dataset,
                         batch_size=config.batch_size,
                         shuffle=False)

negative_valid_loader = DataLoader(negative_valid_dataset,
                         batch_size=config.batch_size,
                         shuffle=False)

positive_test_loader = DataLoader(positive_test_dataset,
                         batch_size=config.batch_size,
                         shuffle=False)

negative_test_loader = DataLoader(negative_test_dataset,
                         batch_size=config.batch_size,
                         shuffle=False)

steps_per_epoch = math.ceil(len(train_loader)/len(devices))

tag = "tiny1"
variant = "D"

MODELS_CONFIGS = {
    "tiny1": {
        "embed_dim":64, "embed_num":1, "depth":[2,2,4], "num_heads":4},
    "tiny2": {
        "embed_dim":64, "embed_num":4, "depth":[2,2,4], "num_heads":4},
    "tiny3": {
        "embed_dim":64, "embed_num":4, "depth":[8,8,8], "num_heads":4},
    "little": {
        "embed_dim":128, "embed_num":4, "depth":[8,8,8], "num_heads":4},
    "base1": {
        "embed_dim":256, "embed_num":1, "depth":[6,6,6], "num_heads":4},
    "base2": {
        "embed_dim":256, "embed_num":4, "depth":[8,8,8], "num_heads":4},
    "base3": {
        "embed_dim":512, "embed_num":1, "depth":[6,6,6], "num_heads":8},
    "large": {
        "embed_dim":512, "embed_num":4, "depth":[8,8,8], "num_heads":8},
}

def get_config(embed_dim=512, embed_num=4, depth=[8,8,8], num_heads=4):

    models_configs = {
            'encoder': {
                    'embed_dim': embed_dim,
                    'embed_num': embed_num,
                    'depth': depth[0],
                    'num_heads': num_heads,
                },
            'predictor': {
                    'embed_dim': embed_dim,
                    'embed_num': embed_num,
                    'predictor_embed_dim': embed_dim,
                    'depth': depth[1],
                    'num_heads': num_heads,
                },
            'reconstructor': {
                    'embed_dim': embed_dim,
                    'embed_num': embed_num,
                    'reconstructor_embed_dim': embed_dim,
                    'depth': depth[2],
                    'num_heads': num_heads,
                },
    }
    return models_configs



