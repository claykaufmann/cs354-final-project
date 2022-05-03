"""
For training on the VACC
"""
from data_processing.Datasets import get_maestro_dataset, collate_fn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
from model.transformer import Transformer
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from tqdm.autonotebook import trange
from time import sleep
from functools import partial


def main():
    """
    main function
    """
    # collect params from passed in config.yml file path

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    train_data, test_data = get_maestro_dataset("data/maestro", representation="pitch")

    # Build dataloaders
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        collate_fn=partial(collate_fn, seq_len=SEQ_LEN, device=device),
        shuffle=False,
    )

    val_dataloader = DataLoader(
        dataset=test_data,
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=partial(collate_fn, seq_len=SEQ_LEN, device=device),
        shuffle=False,
    )

    # create model

    # train

    # save results

    # save plots

    # exit


main()
