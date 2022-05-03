import torch.nn as nn
import torch
import math
from torch import Tensor


class PositionalEncoding(nn.Module):
    """
    positional encoding module, allows us to make positional encoding on modelsx
    Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, dim_model: int, dropout_p: float, max_len: int) -> None:
        """
        constructor

        PARAMS:
        dim_model: the core models dimensionality
        dropout_p: the dropout of the model
        max_len: the max length of the encoding
        """

        super().__init__()

        # create dropout
        self.dropout = nn.Dropout(dropout_p)

        # create encoding
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        """
        forward function

        PARAMS:
        token_embedding: the embedding of tokens created by the transformer

        RETURNS: a tensor with the positional encoding SHAPE: (batch size, seq_len, dim_model)
        """

        return self.dropout(
            token_embedding + self.pos_encoding[: token_embedding.size(0), :]
        )


class Transformer(nn.Module):
    """
    Transformer model
    At its core, a wrapper aorund the core PyTorch transformer module,
    that includes positional encoding and embeding
    """

    def __init__(
        self,
        num_tokens: int,
        dim_model: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout_p: float,
        max_len=5000,
    ) -> None:
        """
        Constructor

        PARAMS:
        num_tokens: the number of tokens to use
        dim_model: the dimensionality of the model
        num_heads: the number of heads to use for multi-head attention
        num_encoder_layers: how many encoder layers to use
        num_decoder_layers: the number of decoder layers to sue
        dropout_p: the dropout percentage
        max_len: the max length of the positional encoder
        """
        super().__init__()

        # set model type, and save model dim
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # save positional encoder
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=max_len
        )

        # save embedding
        self.embedding = nn.Embedding(num_tokens, dim_model)

        # create core transformer
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )

        # final out layer
        self.out = nn.Linear(dim_model, num_tokens)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        tgt_mask=None,
        src_pad_mask=None,
        tgt_pad_mask=None,
    ) -> Tensor:
        """
        forward propagation function

        PARAMS:
        src: the src (X) tensor, shape: (src sequence length, batch size)
        tgt: the tgt (y) tensor shape: (tgt sequence length, batch size) (NOTE: src and tgt should have same length)
        tgt_mask: the tgt_mask
        src_pad_mask: if needed, adds padding
        tgt_pad_mask: same as src_pad_mask
        """
        # swap batch and sequence length, so batch is first
        src = src.permute(1, 0)
        tgt = tgt.permute(1, 0)

        # pass src/tgt through embedding and positional encoder
        # out size of above src and tgt is (batch size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # swap batch and seq length back
        # SHAPE: (seq length, batch size, dim_model)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # pass src and tgt through the transformer
        # Out size is (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )

        # pass through final FC layer
        out = self.out(transformer_out)

        # return out
        return out

    def get_tgt_mask(self, size: int) -> Tensor:
        """
        generate a tgt mask

        PARAMS:
        size: the size of the target mask
        """
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        return mask
