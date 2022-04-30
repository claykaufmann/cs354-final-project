"""
this file loads the data set for pytorch
"""
import torch
import torch.utils.data as data
import numpy as np
import os
from util.constants import *
import random


SEQUENCE_START = 0


class VideoGameMusicDataset(data.Dataset):
    """
    represents the video game music dataset
    """

    def __init__(self, data_folder_path, max_sequence) -> None:
        super(VideoGameMusicDataset).__init__()

        self.root_dir = data_folder_path

        self.files = self.get_files()

        self.max_seq = max_sequence

    def get_files(self):
        """
        return a list of files within the data folder
        these are just the names, not concatenated with the root dir
        """
        # collect all files within data directory
        files = [f for f in os.listdir(self.root_dir)]

        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """
        get actual items
        """
        if torch.is_tensor(index):
            index = index.tolist()

        filename = os.path.join(self.root_dir, self.files[index])

        raw_mid = torch.tensor(np.load(filename, allow_pickle=True))

        x, tgt = process_midi(raw_mid, self.max_seq, None)

        return (x, tgt)


def process_midi(raw_mid, max_seq, random_seq):
    """
    returns a target for transformer
    """

    x = torch.full((max_seq,), TOKEN_PAD, dtype=TORCH_LABEL_TYPE)
    tgt = torch.full((max_seq,), TOKEN_PAD, dtype=TORCH_LABEL_TYPE)

    raw_len = len(raw_mid)
    full_seq = max_seq + 1  # Performing seq2seq

    if raw_len == 0:
        return x, tgt

    if raw_len < full_seq:
        x[:raw_len] = raw_mid
        tgt[: raw_len - 1] = raw_mid[1:]
        tgt[raw_len] = TOKEN_END
    else:
        # Randomly selecting a range
        if random_seq:
            end_range = raw_len - full_seq
            start = random.randint(SEQUENCE_START, end_range)

        # Always taking from the start to as far as we can
        else:
            start = SEQUENCE_START

        end = start + full_seq

        data = raw_mid[start:end]

        x = data[:max_seq]
        tgt = data[1:full_seq]

    return x, tgt
