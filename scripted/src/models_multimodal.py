import torch.nn as nn
from src.transformer_utils import Transformer
import torch
import math


class Residual(nn.Module):
    """
    A residual block that adds the input to the output of a function.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        # Apply the function and add the input to the result
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        channels=1,
        kernel_size=5,
        patch_size=8,
        n_out=128,
        dropout_prob=0.5,
    ):
        super(ConvMixer, self).__init__()

        # Initial convolution layer
        self.net = nn.Sequential(
            nn.Conv2d(
                channels, dim, kernel_size=patch_size, stride=patch_size, bias=False
            ),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

        # Adding depth number of ConvMixer layers with dropout
        for _ in range(depth):
            self.net.append(
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(
                                dim, dim, kernel_size, groups=dim, padding="same"
                            ),
                            nn.GELU(),
                            nn.BatchNorm2d(dim),
                            nn.Dropout(dropout_prob),  # Add dropout here
                        )
                    ),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                    nn.Dropout(dropout_prob),  # Add dropout here
                )
            )

        # Projection head with dropout
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout_prob),  # Add dropout here
            nn.Linear(1024, n_out),
        )

    def forward(self, x):
        # Forward pass through the network
        x = self.net(x)
        x = self.projection(x)
        return x


class TimePositionalEncoding(nn.Module):
    def __init__(self, d_emb):
        """
        Inputs
            d_model - Hidden dimensionality.
        """
        super().__init__()
        self.d_emb = d_emb

    def forward(self, t):
        pe = torch.zeros(t.shape[0], t.shape[1], self.d_emb).to(t.device)  # (B, T, D)
        div_term = torch.exp(
            torch.arange(0, self.d_emb, 2).float() * (-math.log(10000.0) / self.d_emb)
        )[None, None, :].to(
            t.device
        )  # (1, 1, D / 2)
        t = t.unsqueeze(2)  # (B, 1, T)
        pe[:, :, 0::2] = torch.sin(t * div_term)  # (B, T, D / 2)
        pe[:, :, 1::2] = torch.cos(t * div_term)  # (B, T, D / 2)
        return pe  # (B, T, D)


class TransformerWithTimeEmbeddings(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self, n_out, nband=1, agg="mean", **kwargs):
        """
        :param n_out: Number of output emedding.
        :param kwargs: Arguments for Transformer.
        """
        super().__init__()

        self.agg = agg
        self.nband = nband
        self.embedding_mag = nn.Linear(in_features=1, out_features=kwargs["emb"])
        self.embedding_t = TimePositionalEncoding(kwargs["emb"])
        self.transformer = Transformer(**kwargs)

        if nband > 1:
            self.band_emb = nn.Embedding(nband, kwargs["emb"])

        self.projection = nn.Linear(kwargs["emb"], n_out)

        # If using attention, initialize a learnable query vector
        if self.agg == "attn":
            self.query = nn.Parameter(torch.rand(kwargs["emb"]))

    def forward(self, x, t, mask=None):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        t = t - t[:, 0].unsqueeze(1)
        t_emb = self.embedding_t(t)
        x = self.embedding_mag(x) + t_emb

        # learned embeddings for multibands
        if self.nband > 1:
            onehot = (
                torch.linspace(0, self.nband - 1, self.nband)
                .type(torch.LongTensor)
                .repeat_interleave(x.shape[1] // self.nband)
            )
            onehot = onehot.to(t.device)  # (T,)
            b_emb = (
                self.band_emb(onehot).unsqueeze(0).repeat((x.shape[0], 1, 1))
            )  # (T, D) -> (B, T, D)
            x = x + b_emb

        x = self.transformer(x, mask)  # (B, T, D)

        # Zero out the masked values
        x = x * mask[:, :, None]

        if self.agg == "mean":
            x = x.sum(dim=1) / mask.sum(dim=1)[:, None]
        elif self.agg == "max":
            x = x.max(dim=1)[0]
        elif self.agg == "attn":
            q = self.query.unsqueeze(0).repeat(
                x.shape[0], 1, 1
            )  # Duplicate the query across the batch dimension
            k = v = x
            x, _ = nn.MultiheadAttention(
                embed_dim=128, num_heads=2, dropout=0.0, batch_first=True
            )(q, k, v)
            x = x.squeeze(1)  # (B, 1, D) -> (B, D)

        x = self.projection(x)
        return x
