import json
import os
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import shutil
from types import SimpleNamespace
import re
from datetime import datetime
import itertools
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset


def get_subject_id(session_dir_path):
    match_result = re.match('^OpenBCISession_Subject[_ ]([0-9]+)[_ ](.*)$', session_dir_path.stem)

    subject_id = int(match_result.group(1))
    return subject_id


def get_dir_path(*input_dir_paths, data_phase=None, raise_error=True):
    if len(input_dir_paths) == 1:
        parent_paths = []
        (dir_path,) = input_dir_paths
    else:
        (parent_paths, dir_path) = input_dir_paths

    if isinstance(dir_path, (list, tuple)):
        dir_paths = dir_path
    else:
        dir_paths = [dir_path]

    candidate_paths = []

    for curr_dir_path in dir_paths:
        candidate_paths.append(Path(curr_dir_path))

        for parent_path in parent_paths:
            if parent_path and Path(parent_path).exists():
                if data_phase:
                    full_path = Path(parent_path) / data_phase / curr_dir_path
                    candidate_paths.append(full_path)

                candidate_paths.append(Path(parent_path) / curr_dir_path)

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    if raise_error:
        raise ValueError(f"Not accessible path:{candidate_paths}")

    return None


class SeqDatasetGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self):
        config = self.config

        input_root_dir_path = get_dir_path(config.input_root_dir_path)
        output_root_dir_path = get_dir_path(config.output_root_dir_path)
        parent_paths = [input_root_dir_path, output_root_dir_path]

        # target_dir = output_root_dir_path / config.data_phase / config.output_dir_name
        # target_dir.mkdir(parents=True, exist_ok=True)

        seq_splits_dir_path = get_dir_path(parent_paths,
                                    config.input_seq_splits_dir_name,
                                    data_phase=config.data_phase)

        seq_splits_file_path = seq_splits_dir_path / "seq_splits.csv"

        seq_splits_df = pd.read_csv(seq_splits_file_path)
        seq_splits_groups = seq_splits_df.groupby('subject_id')

        eeg_dir_path = get_dir_path(parent_paths,
                                    config.input_eeg_dir_name,
                                    data_phase=config.data_phase)

        eeg_data = self.load_eeg_data(eeg_dir_path)

        all_sample_infos = {}

        for subject_id, subj_seq_splits_df in seq_splits_groups:
            assert subject_id in eeg_data, f"[Error] No eeg_data for {subject_id}"
            eeg_data_df = eeg_data[subject_id]

            for _, row in subj_seq_splits_df.iterrows():
                split = row["Split"]
                start = row['DataStartIndex']
                end = row['DataEndIndex']
                label_index = row["LabelIndex"]
                # shape: [seq_len, num_channels]
                seq = eeg_data_df.iloc[start:end].to_numpy()
                expected_seq_shape = (config.seq_len, config.num_channels)
                assert seq.shape == expected_seq_shape, f"{seq.shape} != {expected_seq_shape}"
                # shape: [num_channels, seq_len]
                seq = seq.T
                sample_info = {"seq": seq, "label_index": label_index}

                if config.trainval:
                    if split in ["train", "val"]:
                        split = "trainval"

                all_sample_infos.setdefault(split, [])
                all_sample_infos[split].append(sample_info)

        split_infos = {}

        for split, sample_infos in all_sample_infos.items():
            seqs = []
            label_indices = []

            for sample_info in sample_infos:
                seq = sample_info["seq"]
                label_index = sample_info["label_index"]
                seqs.append(seq)
                label_indices.append(label_index)

            inputs = np.array(seqs)
            labels = np.array(label_indices)

            split_info = {"inputs": inputs, "labels": labels}
            split_infos[split] = split_info

        if "trainval" in split_infos:
            test_split_info = split_infos["test"]
            test_dataset = IconsenseSmallDataset(test_split_info)

            trainval_split_info = split_infos["trainval"]
            trainval_inputs = trainval_split_info["inputs"]
            trainval_labels = trainval_split_info["labels"]

            skf = StratifiedKFold(n_splits=config.n_splits,
                                  shuffle=True,
                                  random_state=config.seed)

            for fold_idx, (train_index, val_index) in enumerate(skf.split(trainval_inputs, trainval_labels)):
                train_inputs = trainval_inputs[train_index]
                train_labels = trainval_labels[train_index]

                val_inputs = trainval_inputs[val_index]
                val_labels = trainval_labels[val_index]

                train_split_info = {
                    "inputs": train_inputs,
                    "labels": train_labels
                }
                train_dataset = IconsenseSmallDataset(train_split_info)

                val_split_info = {
                    "inputs": val_inputs,
                    "labels": val_labels
                }
                val_dataset = IconsenseSmallDataset(val_split_info)

                datasets = {
                    "train": train_dataset,
                    "val": val_dataset,
                    "test": test_dataset,
                }

                yield {"fold_idx": fold_idx, "datasets": datasets}

        else:
            datasets = {}

            for split, split_info in split_infos.items():
                dataset = IconsenseSmallDataset(split_info)

                datasets[split] = dataset

            yield {"fold_idx": 0, "datasets": datasets}

    def load_eeg_data(self, eeg_csv_dir_path):
        config = self.config

        eeg_data = {}

        csv_session_dirs = list(eeg_csv_dir_path.glob("OpenBCISession_Subject_*"))
        csv_session_dirs.sort(key=get_subject_id)

        pbar = tqdm(csv_session_dirs)

        for csv_session_dir in pbar:
            subject_id = get_subject_id(csv_session_dir)
            csv_file_paths = list(csv_session_dir.glob("*_ica.csv"))

            assert len(csv_file_paths) == 1, csv_session_dir

            csv_file_path = csv_file_paths[-1]
            pbar.set_description(f"Processing {subject_id}")

            eeg_data_df = pd.read_csv(csv_file_path,
                                      usecols=config.eeg_column_names)

            assert subject_id not in eeg_data, f"{subject_id}"
            eeg_data[subject_id] = eeg_data_df

        return eeg_data

    def get_datasets_stat(self, fold_idx, datasets):
        config = self.config

        print(f"Fold {fold_idx + 1} / {config.n_splits}")

        for split, dataset in datasets.items():
            print(f"{split}:{len(dataset)}")

        num_samples_dict = {}

        for split, dataset in datasets.items():
            num_samples_dict.setdefault(split, {})

            for sample in dataset:
                input_data, label_index = sample
                num_samples_dict[split].setdefault(label_index, 0)
                num_samples_dict[split][label_index] += 1

        num_split_neg_pos_samples = {}

        for split, num_split_samples in num_samples_dict.items():
            num_negative = num_split_samples[config.negative_label_index]
            num_positive = sum(num_split_samples.values()) - num_negative
            neg_pos_ratio = num_negative / num_positive

            print(f"{split} num_positive: {num_positive} num_negative: {num_negative} neg_pos_ratio: {neg_pos_ratio:.2f}")

            num_neg_pos_samples = {
                "num_negative": num_negative,
                "num_positive": num_positive
            }
            num_split_neg_pos_samples[split] = num_neg_pos_samples

        return num_split_neg_pos_samples

class IconsenseDataset(Dataset):
    def __init__(self, sample_infos):
        self.sample_infos = sample_infos

    def __len__(self):
        return len(self.sample_infos)

    def __getitem__(self, idx):
        # Shape: [num_channels, seq_len]
        sample_info = self.sample_infos[idx]
        seq = sample_info["seq"]
        label_index = sample_info["label_index"]
        return torch.tensor(seq, dtype=torch.float32), label_index

class IconsenseSmallDataset(Dataset):
    def __init__(self, sample_infos):
        self.inputs = sample_infos["inputs"]
        self.labels = sample_infos["labels"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        seq = self.inputs[idx]
        label_index = self.labels[idx]
        return torch.tensor(seq, dtype=torch.float32), label_index

def main():
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

        eeg_column_names=["FP1", "FP2", "C3", "C4", "P7", "P8", "O1", "O2", "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4"],
        seq_len = 1024,
        num_channels = 16,

        negative_label_index=0,
        trainval=True,
        n_splits = 5,
    )

    config = SimpleNamespace(**config_dict)

    dataset_generator = SeqDatasetGenerator(config)
    datasets_infos = dataset_generator.generate()

    for datasets_info in datasets_infos:
        fold_idx = datasets_info["fold_idx"]
        datasets = datasets_info["datasets"]
        num_split_neg_pos_samples = dataset_generator.get_datasets_stat(fold_idx, datasets)

if __name__ == '__main__':
    main()
