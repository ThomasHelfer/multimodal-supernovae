# The standard CLIP architecture uses a bidirection (symmetric between modalities, e.g. image and text) version of the so-called SimCLR loss to compute alignment between image and light curve representations.
# $$\mathcal{L}_\mathrm{CLIP}=-\frac{1}{2|\mathcal{B}|} \sum_{i=1}^{|\mathcal{B}|}\left(\log \frac{e^{t\,x_i \cdot y_i}}{\sum_{j=1}^{|\mathcal{B}|} e^{t\,x_i \cdot y_j}}+\log \frac{e^{t\,x_i \cdot y_i}}{\sum_{j=1}^{|\mathcal{B}|} e^{t\,x_j \cdot y_i}}\right)$$
#
# The standard CLIP loss can be quite unstable due to the small number of positive pairs and large number of negative pairs in a batch. It can also often require very large batch sizes to work well. There are many proposed ways of overcoming this, e.g. see https://lilianweng.github.io/posts/2021-05-31-contrastive/ for some approaches.
#
# In addition to theh softmax-based loss, we'll also try a sigmoid loss, from https://arxiv.org/abs/2303.15343:
# $$\mathcal{L}_\mathrm{SigLIP}=-\frac{1}{|\mathcal{B}|} \sum_{i=1}^{|\mathcal{B}|} \sum_{j=1}^{|\mathcal{B}|} \log \frac{1}{1+e^{z_{i j}\left(-t\, {x}_i \cdot {y}_j+b\right)}}$$
# where $x_i$ and $y_j$ are the normalized image and light curve representations, respectively, and $z_{ij}$ is a binary indicator of whether the image and light curve are a match or not.

import torch
import torch.nn as nn


def clip_loss(
    image_embeddings,
    text_embeddings,
    logit_scale=1.0,
    logit_bias=0.0,
    image_encoder=None,
    lightcurve_encoder=None,
    printing=False,
):
    logit_scale = logit_scale.exp()

    logits = (text_embeddings @ image_embeddings.T) * logit_scale + logit_bias

    images_loss = nn.LogSoftmax(dim=1)(logits)
    texts_loss = nn.LogSoftmax(dim=0)(logits)

    images_loss = -images_loss.diag()
    texts_loss = -texts_loss.diag()

    n = min(len(image_embeddings), len(text_embeddings))

    images_loss = images_loss.sum() / n
    texts_loss = texts_loss.sum() / n

    loss = (images_loss + texts_loss) / 2
    return loss


def sigmoid_loss(image_embeds, text_embeds, logit_scale=1.0, logit_bias=2.73):
    """Sigmoid-based CLIP loss, from https://arxiv.org/abs/2303.15343"""

    logit_scale = logit_scale.exp()

    bs = text_embeds.shape[0]

    labels = 2 * torch.eye(bs) - torch.ones((bs, bs))
    labels = labels.to(text_embeds.device)

    logits = -text_embeds @ image_embeds.t() * logit_scale + logit_bias
    logits = logits.to(torch.float64)

    loss = -torch.mean(torch.log(torch.sigmoid(-labels * logits)))

    return loss