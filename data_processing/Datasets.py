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


def collate_fn(batch, seq_len, device):
    """
    Pads batch of variable length music sequences
    For use within a dataloader, see below

    to correctly use this function, functools.partial is required
    EX:
    ```
    from functools import partial
    dataloader = DataLoader(collate_fn=partial(collate_fn, seq_len=SEQ_LEN, device=device))
    ```

    PARAMS:
    batch: passed in automatically by pytorch
    seq_len: the sequence length to cut down to
    device: the active pytorch device

    RETURNS:
    batch: a tensor with the data. SHAPE: (batch_size, seq_len)
    lengths: all sequence lengths in the batch
    targets: a tensor with all batch targets (batch shifted by one)
    """
    # get sequence lengths
    # this isn't super necessary, but in case it is needed in the future
    lengths = torch.tensor([t.shape[0] for t in batch]).to(device)

    # set length to SEQ_LEN + 1
    batch = [torch.Tensor(t)[: seq_len + 1].squeeze().to(device) for t in batch]

    # this isn't technically needed since we cut all sequences down, but just in case
    batch = torch.nn.utils.rnn.pad_sequence(batch)

    # targets array is just main array subtracted by 1, as it is the future items
    targets = batch[:-1]

    # shorten batch by one (to sync length of batch and target)
    batch = batch[1:]

    return batch, lengths, targets
