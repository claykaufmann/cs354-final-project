"""
For training on the VACC
"""

from data_processing.Datasets import get_maestro_dataset, collate_fn
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model.transformer import Transformer
from tqdm import tqdm
from time import sleep
from functools import partial
import matplotlib.pyplot as plt

# for now, set hyperparameter constants here
# training hyperparams
EPOCHS = 25
LEARNING_RATE = 1e-3

# data hyperparams
SEQ_LEN = 2048
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 10

# model hyperparams
# NUM_TOKENS should be one of the following, depending on dataset representation:
# pitch: 128
# event: 388
NUM_TOKENS = 388
DIM_MODEL = 512
NUM_HEADS = 2
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT_P = 0.1


def train(model, optimizer, loss_fn, dataloader, device):
    """
    train the model
    """
    # set model to training mode
    model.train()

    # init loss metrics
    total_loss = 0

    with tqdm(dataloader, unit="batch") as bar:
        for batch in bar:
            X, y = batch[0], batch[2]

            # X = torch.tensor(X, dtype=torch.long, device=device)
            # y = torch.tensor(y, dtype=torch.long, device=device)

            X = batch[0].clone().detach()
            y = batch[2].clone().detach()

            X = X.long().to(device)
            y = y.long().to(device)

            # separate these inputs to stop mess ups in permutations
            y_input = y
            y_expected = y

            seq_len = y_input.size(0)
            tgt_mask = model.get_tgt_mask(seq_len).to(device)

            # get prediction
            pred = model(X, y_input, tgt_mask)

            # permute pred and y_expected to put batch first
            pred = pred.permute(1, 2, 0)
            y_expected = y.permute(1, 0)

            # calc loss
            loss = loss_fn(pred, y_expected)

            # set zero gradients
            optimizer.zero_grad()

            # backprop
            loss.backward()

            # step optimizer
            optimizer.step()

            # update loss metrics
            total_loss += loss.detach().item()

            # update progress bar
            bar.set_postfix(loss=loss.item())
            bar.update()
            sleep(0.1)

    # return the total loss (for plotting per epoch)
    return total_loss / len(dataloader)


def validation(model, loss_fn, dataloader, device):
    """
    validation loop
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as bar:
            for batch in dataloader:
                X, y = batch[0], batch[2]

                # X = torch.tensor(X, dtype=torch.long, device=device)
                # y = torch.tensor(y, dtype=torch.long, device=device)

                X = batch[0].clone().detach()
                y = batch[2].clone().detach()

                # convert tensor to long for the transformer
                X = X.long().to(device)
                y = y.long().to(device)

                # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
                y_input = y
                y_expected = y

                # Get mask to mask out the next words
                sequence_length = y_input.size(0)
                tgt_mask = model.get_tgt_mask(sequence_length).to(device)

                # Standard training except we pass in y_input and src_mask
                pred = model(X, y_input, tgt_mask)

                # Permute pred to have batch size first again
                pred = pred.permute(1, 2, 0)

                # permute y_expected
                y_expected = y_expected.permute(1, 0)

                loss = loss_fn(pred, y_expected)
                total_loss += loss.detach().item()

                # update progress
                bar.set_postfix(loss=loss.item())
                bar.update()
                sleep(0.1)

    return total_loss / len(dataloader)


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs, device, lr_scheduler):
    """
    Fit the model
    """
    train_loss_list, validation_loss_list = [], []

    print("Fitting model...")

    for epoch in range(epochs):
        print(f"---------------- EPOCH {epoch + 1} ----------------")

        train_loss = train(model, opt, loss_fn, train_dataloader, device)
        train_loss_list += [train_loss]

        validation_loss = validation(model, loss_fn, val_dataloader, device)
        validation_loss_list += [validation_loss]

        # step scheduler
        lr_scheduler.step()

        print(f"Training Loss: {train_loss}")
        print(f"Validation Loss: {validation_loss}")
        print()

    return train_loss_list, validation_loss_list


def main():
    """
    main function
    """

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    train_data, test_data = get_maestro_dataset("data/maestro", representation="event")

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

    # create model, optim, criterion
    model = Transformer(
        num_tokens=NUM_TOKENS,
        dim_model=DIM_MODEL,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout_p=DROPOUT_P,
    ).to(device)

    # create optmizer, pass in learning rate hyperparam
    opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # create loss function
    loss_fn = nn.CrossEntropyLoss()

    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, 2, 0.1)

    # train
    train_loss_list, validation_loss_list = fit(
        model, opt, loss_fn, train_dataloader, val_dataloader, EPOCHS, device, lr_scheduler
    )

    # save results
    torch.save(model.state_dict(), "./transformer_music.pth")

    # gen plots
    plot_epochs = range(1, EPOCHS + 1)
    plt.plot(plot_epochs, train_loss_list, "g", label="Training loss")
    plt.plot(plot_epochs, validation_loss_list, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./loss_plot.png")


main()
