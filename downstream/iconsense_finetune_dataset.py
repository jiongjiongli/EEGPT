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
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset


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
                all_sample_infos.setdefault(split, [])
                all_sample_infos[split].append(sample_info)

        datasets = {}

        for split, sample_infos in all_sample_infos.items():
            dataset = IconsenseDataset(sample_infos)
            datasets[split] = dataset

        return datasets

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
    )

    config = SimpleNamespace(**config_dict)

    dataset_generator = SeqDatasetGenerator(config)
    datasets = dataset_generator.generate()

    for split, dataset in datasets.items():
        print(f"{split}:{len(dataset)}")


if __name__ == '__main__':
    main()
