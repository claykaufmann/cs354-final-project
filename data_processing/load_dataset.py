"""
this file loads the data set for pytorch
"""
import torch
import torch.nn as nn
import torch.utils.data as data


class VideoGameMusicDataset(data.IterableDataset):
    def __init__(self, start, end) -> None:
        super(VideoGameMusicDataset).__init__()

        assert end > start

        self.start = start
        self.end = end
