import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    """
    Canonical implementation of multi-head self attention.
    """

    def __init__(self, emb, heads=2):
        """
        :param emb:
        :param heads:
        """

        super().__init__()

        assert (
            emb % heads == 0
        ), f"Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})"

        self.emb = emb
        self.heads = heads

        # We will break the embedding into `heads` chunks and feed each to a different attention head

        self.tokeys = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues = nn.Linear(emb, emb, bias=False)

        self.unifyheads = nn.Linear(emb, emb)

    def forward(self, x, mask=None):
        b, t, e = x.size()
        h = self.heads
        assert (
            e == self.emb
        ), f"Input embedding dim ({e}) should match layer embedding dim ({self.emb})"

        s = e // h

        keys = self.tokeys(x)
        queries = self.toqueries(x)
        values = self.tovalues(x)

        keys = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values = values.view(b, t, h, s)

        # -- We first compute the k/q/v's on the whole embedding vectors, and then split into the different heads.
        #    See the following video for an explanation: https://youtu.be/KmAISyVvE1Y

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        if mask is not None:
            # expand the mask to match tensor dims
            mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand(b, h, 1, t).reshape(b * h, 1, t)

            # replace the False values with -inf
            dot = dot.masked_fill(~mask, float("-1e7"))

        dot = F.softmax(dot, dim=2)

        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, ff_hidden_mult=6, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads)

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb),
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attended = self.attention(x, mask=mask)
        x = self.norm1(attended + x)
        x = self.do(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        x = self.do(x)
        return x


class Transformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self, emb, heads, depth, ff_hidden_mult=4, dropout=0.0):
        """
        :param emb: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param ff_hidden_mult: Hidden layer dimension in feedforward network, as a fraction of `emb`
        """
        super().__init__()

        self.tblocks = nn.ModuleList(
            [
                TransformerBlock(
                    emb=emb, heads=heads, ff_hidden_mult=ff_hidden_mult, dropout=dropout
                )
                for _ in range(depth)
            ]
        )
        self.do = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """

        x = self.do(x)
        for tblock in self.tblocks:
            x = tblock(x, mask)

        return x


class TimePositionalEncoding(nn.Module):
    def __init__(self, d_emb, norm=10000.0):
        """
        Inputs
            d_model - Hidden dimensionality.
        """
        super().__init__()
        self.d_emb = d_emb
        self.norm = norm

    def forward(self, t):
        pe = torch.zeros(t.shape[0], t.shape[1], self.d_emb).to(t.device)  # (B, T, D)
        div_term = torch.exp(
            torch.arange(0, self.d_emb, 2).float() * (-math.log(self.norm) / self.d_emb)
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

    def __init__(self, n_out, nband=1, agg="mean", time_norm=10000.0, **kwargs):
        """
        :param n_out: Number of output embedding.
        :param kwargs: Arguments for Transformer.
        """
        super().__init__()

        self.agg = agg
        self.nband = nband
        self.embedding_mag = nn.Linear(in_features=1, out_features=kwargs["emb"])
        self.embedding_t = TimePositionalEncoding(kwargs["emb"], time_norm)
        self.transformer = Transformer(**kwargs)

        if nband > 1:
            self.band_emb = nn.Embedding(nband, kwargs["emb"])

        self.projection = nn.Linear(kwargs["emb"], n_out)

        # If using attention, initialize a learnable query vector
        if self.agg == "attn":
            self.query = nn.Parameter(torch.rand(kwargs["emb"]))
            self.agg_attn = nn.MultiheadAttention(
                embed_dim=kwargs["emb"], num_heads=2, dropout=0.0, batch_first=True
            )

    def forward(self, x, t, mask=None):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        # Add time embeddings
        t_emb = self.embedding_t(t)
        x = self.embedding_mag(x) + t_emb

        # learned embeddings for multibands
        if self.nband > 1:
            # first half of the array is band 0, second half is band 1, etc.
            # creates one-hot encoding for bands
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
            x, _ = self.agg_attn(q, k, v)
            x = x.squeeze(1)  # (B, 1, D) -> (B, D)
        if self.agg == "pretraining":
            return x

        x = self.projection(x)

        return x
