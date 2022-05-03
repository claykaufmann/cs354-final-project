"""
this file loads the data set for pytorch
"""
import torch
import muspy


def get_maestro_dataset(path: str, representation: str = "pianoroll"):
    """
    a simple wrapper for collecting maestro dataset

    params:
    path: path to store the data
    representation: the way to represent the dataset
    """
    maestro = muspy.MAESTRODatasetV3(path, download_and_extract=True)

    maestro.convert()

    data = maestro.to_pytorch_dataset(representation=representation)

    train_size = int(0.85 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

    return train_data, test_data
