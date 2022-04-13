"""
this file loads the data set for pytorch
"""
import torch
import torch.utils.data as data
import numpy as np
import os


class VideoGameMusicDataset(data.Dataset):
    """
    represents the video game music dataset
    """

    def __init__(self, data_folder_path) -> None:
        super(VideoGameMusicDataset).__init__()

        self.root_dir = data_folder_path

        self.files = self.get_files()

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

        filename = os.path.join(self.root_dir, self.files)

        sample = np.load(filename)

        if self.transform:
            sample = self.transform(sample)

        return sample
