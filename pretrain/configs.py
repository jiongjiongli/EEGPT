
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


import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace
import re
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

    time_param_column_index = 1,
    batch_size = 2,
    train_split_percent = 0.9,

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

    epochs=2,

    model_log_dir = "./logs",
    kaggle_model_log_dir = "/kaggle/working/logs",

    # Validate
    validate_interval = 100,
)

config = SimpleNamespace(**config_dict)



max_epochs = config.epochs
max_lr = 5e-4
batch_size=config.batch_size
devices=[0]


def get_subject_id(session_dir_path):
    match_result = re.match('^OpenBCISession_Subject[_ ]([0-9]+)[_ ](.*)$', session_dir_path.stem)

    subject_id = int(match_result.group(1))
    return subject_id


root_dir_path = Path(config.root_dir_path)
kaggle_label_root_dir_path = Path(config.kaggle_label_root_dir_path)
kaggle_eeg_root_dir_path = Path(config.kaggle_eeg_root_dir_path)

if kaggle_label_root_dir_path.exists():
    csv_dir_path = kaggle_label_root_dir_path / config.input_csv_dir_name
else:
    csv_dir_path = root_dir_path / config.data_phase / config.input_csv_dir_name

if kaggle_eeg_root_dir_path.exists():
    eeg_csv_dir_path = kaggle_eeg_root_dir_path / config.input_eeg_csv_dir_name
else:
    eeg_csv_dir_path = root_dir_path / config.data_phase / config.input_eeg_csv_dir_name

csv_session_dirs = list(eeg_csv_dir_path.glob("OpenBCISession_Subject_*"))
csv_session_dirs.sort(key=get_subject_id)

all_seqs = []

pbar = tqdm(csv_session_dirs)

for csv_session_dir in pbar:
    subject_id = get_subject_id(csv_session_dir)
    csv_file_paths = list(csv_session_dir.glob("*_ica.csv"))

    assert len(csv_file_paths) == 1, csv_session_dir

    csv_file_path = csv_file_paths[-1]
    pbar.set_description(f"Processing {csv_file_path}")

    eeg_data_df = pd.read_csv(csv_file_path, usecols=config.eeg_column_names)
    # seq_len * 2 to enable sliding window to select seq_len
    sample_seq_len = config.seq_len * 2
    num_seqs = len(eeg_data_df) // sample_seq_len
    # shape: [num_seqs * seq_len, num_channels]
    seqs = eeg_data_df.iloc[:num_seqs * sample_seq_len].to_numpy()
    # shape: [num_seqs, seq_len, num_channels]
    seqs = seqs.reshape(num_seqs, sample_seq_len, -1)
    # shape: [num_seqs, num_channels, seq_len]
    seqs = seqs.transpose(0, 2, 1)
    all_seqs.append(seqs)

# shape: [num_seqs, num_channels, seq_len]
all_seqs = np.concatenate(all_seqs, axis=0)

# shape: [num_seqs, num_channels, seq_len]
data_tensor = torch.from_numpy(all_seqs).to(torch.float)
num_train = int(config.train_split_percent * len(data_tensor))
num_val = len(data_tensor) - num_train

torch.manual_seed(config.seed)
train_data, valid_data = torch.utils.data.random_split(data_tensor,
                                                     [num_train, num_val])
train_dataset = TensorDataset(train_data)
valid_dataset  = TensorDataset(valid_data)

train_loader = DataLoader(train_dataset,
                          batch_size=config.batch_size,
                          shuffle=True)
valid_loader  = DataLoader(valid_dataset,
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



