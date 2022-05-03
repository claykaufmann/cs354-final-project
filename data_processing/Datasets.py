"""
this file loads the data set for pytorch

Originally it had a custom designed dataset by us,
but we found muspy, and it is much simpler, so we
used that.
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
    # load the dataset
    maestro = muspy.MAESTRODatasetV3(path, download_and_extract=True)

    # convert and save music objects from dataset
    maestro.convert()

    # turn the dataset into a pytorch dataset
    data = maestro.to_pytorch_dataset(representation=representation)

    # train/test splits
    train_size = int(0.85 * len(data))
    test_size = len(data) - train_size

    # create train/test datasets
    train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

    # return datasets
    return train_data, test_data
