import sys

dataset_path = "/home/iconsense/Desktop/jiongjiong_li/proj/git/EEGPT/downstream"

if dataset_path not in sys.path:
    sys.path.append(dataset_path)

import json
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace
import re
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset

from iconsense_finetune_dataset import SeqDatasetGenerator, get_dir_path


def get_subject_id(session_dir_path):
    match_result = re.match('^OpenBCISession_Subject[_ ]([0-9]+)[_ ](.*)$', session_dir_path.stem)

    subject_id = int(match_result.group(1))
    return subject_id


class WindowDataset(Dataset):
    def __init__(self, seqs):
        # shape: [num_seqs, num_channels, (seq_len * 2)]
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        # Shape: [num_channels, seq_len * 2]
        seq = self.seqs[idx]
        max_start_index = seq.shape[-1] - config.seq_len
        start = np.random.randint(0, max_start_index + 1)

        # Shape: [num_channels, seq_len]
        sliced = seq[:, start:start + config.seq_len]
        return torch.tensor(sliced, dtype=torch.float32), -1


def get_sub_based_dataset(config):
    root_dir_path = Path(config.root_dir_path)
    kaggle_label_root_dir_path = Path(config.kaggle_label_root_dir_path)
    kaggle_eeg_root_dir_path = Path(config.kaggle_eeg_root_dir_path)
    kaggle_working_dir_path = Path(config.kaggle_working_dir_path)

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

    seqs_infos = []

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
        # shape: [num_seqs * (seq_len * 2), num_channels]
        seqs = eeg_data_df.iloc[:num_seqs * sample_seq_len].to_numpy()
        # shape: [num_seqs, (seq_len * 2), num_channels]
        seqs = seqs.reshape(num_seqs, sample_seq_len, config.num_channels)
        # shape: [num_seqs, num_channels, (seq_len * 2)]
        seqs = np.transpose(seqs, (0, 2, 1))
        seqs_infos.append({"subject_id": subject_id, "seqs": seqs})

    seqs_infos_trainval, seqs_infos_test = train_test_split(seqs_infos,
                                                            test_size=config.test_percent,
                                                            random_state=config.seed)

    seqs_infos_train, finetune_seqs_infos_val = train_test_split(seqs_infos_trainval,
                                                            test_size=config.valid_percent,
                                                            random_state=config.seed)

    train_subject_ids = [seqs_info["subject_id"] for seqs_info in seqs_infos_train]
    finetune_val_subject_ids = [seqs_info["subject_id"] for seqs_info in finetune_seqs_infos_val]
    test_subject_ids = [seqs_info["subject_id"] for seqs_info in seqs_infos_test]

    subject_id_splits = {
        "train": train_subject_ids,
        "val": finetune_val_subject_ids,
        "test": test_subject_ids,
    }

    print(subject_id_splits)

    if kaggle_working_dir_path.exists():
        target_dir = kaggle_working_dir_path / config.output_dir_name
    else:
        target_dir = root_dir_path / config.data_phase / config.output_dir_name
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    target_file_name = config.subject_ids_file_name
    target_file_path = target_dir / target_file_name
    target_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_file_path, "w") as file_stream:
        json.dump(subject_id_splits, file_stream, indent=4)

    # shape: [num_seqs, num_channels, (seq_len * 2)]
    pretrain_seqs_trainval = np.concatenate([seqs_info["seqs"] for seqs_info in seqs_infos_train],
                                            axis=0)
    seqs_train, seqs_val = train_test_split(pretrain_seqs_trainval,
                                            test_size=config.valid_percent,
                                            random_state=config.seed)


    train_dataset = WindowDataset(seqs_train)
    valid_dataset = WindowDataset(seqs_val)

    datasets = {
        "prerain_train": train_dataset,
        "pretrain_val": valid_dataset,
        "finetune_seqs_infos_val": finetune_seqs_infos_val,
        "test": seqs_infos_test,
    }

    return datasets


def get_seq_datasets(config):
    dataset_generator = SeqDatasetGenerator(config)
    datasets_infos = dataset_generator.generate(pretrain=True)

    for datasets_info in datasets_infos:
        fold_idx = datasets_info["fold_idx"]
        datasets = datasets_info["datasets"]

        num_split_neg_pos_samples = dataset_generator.get_datasets_stat(fold_idx, datasets)

        train_dataset = datasets["train"]
        valid_dataset = datasets["valid"]

        pretrain_datasets = {}

        for split, dataset in datasets.items():
            if split == "test":
                pretrain_datasets[split] = dataset
            else:
                label_to_inputs = {}

                for sample in dataset:
                    input_data, label_index = sample
                    label_to_inputs.setdefault(label_index, [])
                    label_to_inputs[label_index].append(input_data)

                for label_index, seqs in label_to_inputs.items():
                    inputs = np.array(seqs)
                    labels = np.array([label_index] * len(seqs))

                    samples_info = {"inputs": inputs, "labels": labels}
                    pretrain_dataset = IconsenseSmallDataset(samples_info)
                    pretrain_datasets.setdefault(split, {})
                    pretrain_datasets[split][label_index] = pretrain_dataset

        return pretrain_datasets


def main():
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
        epochs=10,

        model_log_dir = "./logs",
        kaggle_model_log_dir = "/kaggle/working/logs",

        # Validate
        validate_interval = 100,
        max_lr = 5e-4,
    )

    config = SimpleNamespace(**config_dict)

    datasets = get_seq_datasets(config)

if __name__ == '__main__':
    main()
