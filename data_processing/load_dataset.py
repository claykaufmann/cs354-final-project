"""
this file loads the data set for pytorch
"""
import torch
import torch.nn as nn
import torch.utils.data as data
import math


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
        """
        files = []

        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """
        get actual items
        """
