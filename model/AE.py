import torch
import torch.nn as nn
from typing import List


class SongAE(nn.Module):
    """
    a song autoencoder

    TODO - this is not completed
    """

    def __init__(self, layer_sizes: List[int]):
        super().__init__()

        self.layer_sizes = layer_sizes

        enc_layers = []
        for i in range(len(layer_sizes) - 1):
            size1 = layer_sizes[i]
            size2 = layer_sizes[i + 1]
            enc_layers.append(nn.Linear(size1, size2))
            enc_layers.append(nn.ReLU())

        dec_layers = []
        for i in range(len(layer_sizes) - 1, 0, -1):
            size1 = layer_sizes[i]
            size2 = layer_sizes[i - 1]
            dec_layers.append(nn.Linear(size1, size2))
            dec_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec
