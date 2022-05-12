from data_processing.Datasets import get_maestro_dataset, collate_fn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
from model.transformer import Transformer
from tqdm import tqdm
from time import sleep
from functools import partial