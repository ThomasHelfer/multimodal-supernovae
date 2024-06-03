# Standard library imports
import math
import os
import sys

# Third-party imports
import pytorch_lightning as pl
import torch
import torch.nn as nn
from ruamel.yaml import YAML
import wandb
from torchmetrics.classification import MulticlassFBetaScore

# Local application imports
from src.loss import sigmoid_loss_multimodal, clip_loss_multimodal
from src.transformer_utils import TransformerWithTimeEmbeddings
from src.utils import get_AUC

# Typing imports
from typing import Any, Dict, Optional, Tuple, List


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


class LightCurveImageCLIP(pl.LightningModule):
    def __init__(
        self,
        enc_dim: int = 128,
        logit_scale: float = 10.0,
        nband: int = 1,
        transformer_kwargs: Dict[str, int] = {
            "n_out": 128,
            "emb": 256,
            "heads": 2,
            "depth": 8,
            "time_norm": 10000.0,
        },
        transformer_spectral_kwargs: Dict[str, int] = {
            "n_out": 128,
            "emb": 256,
            "heads": 2,
            "depth": 8,
            "time_norm": 10000.0,
        },
        conv_kwargs: Dict[str, int] = {
            "dim": 32,
            "depth": 8,
            "channels": 3,
            "kernel_size": 5,
            "patch_size": 10,
            "n_out": 128,
        },
        combinations: List[str] = ["host_galaxy", "spectral"],
        optimizer_kwargs: Dict = {},
        lr: float = 1e-4,
        loss: str = "sigmoid",
        regression: bool = False,
        classification: bool = False,
        n_classes: int = 5,
    ):
        """
        Initialize the LightCurveImageCLIP module.

        Args:
        enc_dim (int): Dimension of the encoder.
        logit_scale (float): Initial scale for the logits.
        nband (int): Number of bands.
        transformer_kwargs (Dict[str, int]): Keyword arguments for the transformer encoder.
        conv_kwargs (Dict[str, int]): Keyword arguments for the convolutional encoder.
        optimizer_kwargs (Dict): Keyword arguments for the optimizer.
        combination (array): containing
        lr (float): Learning rate.
        loss (str): Loss function type, either "sigmoid" or "softmax".
        """
        super().__init__()
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs
        self.enc_dim = enc_dim
        self.combinations = set(combinations)
        self.regression = regression
        self.classification = classification
        if self.classification:
            self.n_classes = n_classes

        # Parameters
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(logit_scale)), requires_grad=True
        )
        self.logit_bias = nn.Parameter(torch.tensor(-10.0), requires_grad=True)

        # Encoders
        if "lightcurve" in self.combinations:
            # lightcuve typically has two bands
            self.lightcurve_encoder = TransformerWithTimeEmbeddings(
                nband=nband, **transformer_kwargs
            )
            self.lightcurve_projection = nn.Linear(transformer_kwargs["n_out"], enc_dim)

        if "spectral" in self.combinations:
            # Spectral data does not need the nband variable
            self.spectral_encoder = TransformerWithTimeEmbeddings(
                nband=1, **transformer_spectral_kwargs
            )
            self.spectral_projection = nn.Linear(
                transformer_spectral_kwargs["n_out"], enc_dim
            )

        if "host_galaxy" in self.combinations:
            self.image_encoder = ConvMixer(**conv_kwargs)
            self.image_projection = nn.Linear(conv_kwargs["n_out"], enc_dim)

        self.loss = loss
        self.linear_out = 1  # for regression
        if self.classification:
            self.linear_out = self.n_classes

        if self.regression or self.classification:
            self.linear = nn.Linear(enc_dim * len(self.combinations), self.linear_out)

    def forward(
        self,
        x_img: torch.Tensor,
        x_lc: torch.Tensor,
        t_lc: torch.Tensor,
        mask_lc: Optional[torch.Tensor],
        x_sp: torch.Tensor,
        t_sp: torch.Tensor,
        mask_sp: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
        x_img (torch.Tensor): Input tensor for images.
        x_lc (torch.Tensor): Input tensor for light curves.
        t_lc (torch.Tensor): Time tensor for light curves.
        mask_lc (Optional[torch.Tensor]): Mask tensor for light curves.
        x_sp (torch.Tensor): Input tensor with spectral info
        t_sp (torch.Tensor): frequency tensor with spectral info
        Returns:
        List[torch.Tensor] : Array of embeddings.
        """
        if self.regression or self.classification:
            x = []
            if "host_galaxy" in self.combinations:
                x_img = self.image_encoder(x_img)
                x_img = self.image_projection(x_img)
                x.append(x_img)
            if "lightcurve" in self.combinations:
                x_lc = x_lc[..., None]  # Add channel dimension
                x_lc = self.lightcurve_encoder(x_lc, t_lc, mask_lc)
                x_lc = self.lightcurve_projection(x_lc)
                x.append(x_lc)
            if "spectral" in self.combinations:
                x_sp = x_sp[..., None]  # Add channel dimension
                x_sp = self.spectral_encoder(x_sp, t_sp, mask_sp)
                x_sp = self.spectral_projection(x_sp)
                x.append(x_sp)

            x = torch.cat(x, dim=-1)
            x = self.linear(x)
            return x
        else:
            x = []
            if "host_galaxy" in self.combinations:
                x.append(self.image_embeddings_with_projection(x_img))
            if "lightcurve" in self.combinations:
                x.append(
                    self.lightcurve_embeddings_with_projection(x_lc, t_lc, mask_lc)
                )
            if "spectral" in self.combinations:
                x.append(self.spectral_embeddings_with_projection(x_sp, t_sp, mask_sp))
            return x

    def image_embeddings_with_projection(self, x_img):
        """Convenience function to get image embeddings with projection"""
        x_img = self.image_encoder(x_img)
        x_img = self.image_projection(x_img)
        return x_img / x_img.norm(dim=-1, keepdim=True)

    def lightcurve_embeddings_with_projection(self, x_lc, t_lc, mask_lc=None):
        """Convenience function to get light curve embeddings with projection"""
        x_lc = x_lc[..., None]  # Add channel dimension
        x_lc = self.lightcurve_encoder(x_lc, t_lc, mask_lc)
        x_lc = self.lightcurve_projection(x_lc)
        return x_lc / x_lc.norm(dim=-1, keepdim=True)

    def spectral_embeddings_with_projection(self, x_lc, t_lc, mask_lc=None):
        """Convenience function to get spectral curve embeddings with projection"""
        x_lc = x_lc[..., None]  # Add channel dimension
        x_lc = self.spectral_encoder(x_lc, t_lc, mask_lc)
        x_lc = self.spectral_projection(x_lc)
        return x_lc / x_lc.norm(dim=-1, keepdim=True)

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(
            self.parameters(), lr=self.lr, **self.optimizer_kwargs
        )
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        (
            x_img,
            x_lc,
            t_lc,
            mask_lc,
            x_sp,
            t_sp,
            mask_sp,
            redshift,
            classification,
        ) = batch
        x = self(x_img, x_lc, t_lc, mask_lc, x_sp, t_sp, mask_sp)

        if self.regression:
            loss = nn.MSELoss()(x.squeeze(), redshift)

            # Store the predictions and true values for R2 calculation
            self.y_pred.append(x.flatten())
            self.y_true.append(redshift)

        elif self.classification:
            # matching the (rough) class breakdown of ZTF BTS
            if self.n_classes == 5:
                class_weights = (
                    torch.tensor([0.3, 0.08, 1.0, 0.01, 0.2]).to(x.device).float()
                )
            elif self.n_classes == 3:
                class_weights = torch.tensor([0.33, 0.06, 1.0]).to(x.device).float()
            else:
                # if we can't figure out the classification don't reweight
                class_weights = torch.ones(self.n_classes).to(x.device).float()

            loss = nn.CrossEntropyLoss(weight=class_weights)(
                x.squeeze(), classification.long()
            )

            self.y_pred.append(x)
            self.y_true.append(classification)

        elif self.loss == "sigmoid":
            loss = sigmoid_loss_multimodal(x, self.logit_scale, self.logit_bias).mean()
        elif self.loss == "softmax":
            loss = clip_loss_multimodal(
                x,
                self.logit_scale,
                self.logit_bias,
            ).mean()

        self.log(
            "train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )
        return loss

    def on_train_epoch_start(self):
        if self.regression or self.classification:
            # Initialize empty lists to store predictions and true values
            self.y_true = []
            self.y_pred = []

    def on_train_epoch_end(self) -> None:
        if self.regression:
            # Compute R2
            y_true = torch.cat(self.y_true, dim=0)
            y_pred = torch.cat(self.y_pred, dim=0)
            r2 = (
                1
                - (y_true - y_pred).pow(2).sum() / (y_true - y_true.mean()).pow(2).sum()
            )
            self.log(
                "R2_train",
                r2.cpu(),
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                logger=True,
            )

        elif self.classification:
            # Compute F2 score (weighted harmonic mean between precision and recall)
            y_true = torch.cat(self.y_true, dim=0)
            y_pred = torch.cat(self.y_pred, dim=0)

            y_pred = torch.argmax(y_pred, dim=1)

            y_pred = y_pred.int()  # Ensure it's an integer tensor
            y_true = y_true.int()  # Ensure it's an integer tensor

            # beta is the weighting between precision and recall. For now use beta=1, equal weight.
            f1 = MulticlassFBetaScore(beta=1.0, num_classes=self.n_classes).to(
                y_pred.device
            )(y_pred.int(), y_true)
            self.log(
                "f1_train",
                f1.cpu(),
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                logger=True,
            )

    def on_validation_start(self) -> None:
        """
        Called at the beginning of the validation loop.
        """
        # Initialize an empty list to store embeddings
        self.embs_list = [[] for i in range(len(self.combinations))]
        if self.regression or self.classification:
            self.y_pred_val = []
            self.y_true_val = []

    def validation_step(self, batch, batch_idx):
        (
            x_img,
            x_lc,
            t_lc,
            mask_lc,
            x_sp,
            t_sp,
            mask_sp,
            redshift,
            classification,
        ) = batch
        x = self(x_img, x_lc, t_lc, mask_lc, x_sp, t_sp, mask_sp)

        if self.regression:
            loss = nn.MSELoss()(x.squeeze(), redshift)

            # Store the predictions and true values for R2 calculation
            self.y_pred_val.append(x.flatten())
            self.y_true_val.append(redshift)

        elif self.classification:
            # matching the (rough) class breakdown of ZTF BTS - lower for more common classes
            if self.n_classes == 5:
                class_weights = (
                    torch.tensor([0.3, 0.08, 1.0, 0.01, 0.2]).to(x.device).float()
                )
            elif self.n_classes == 3:
                class_weights = torch.tensor([0.33, 0.06, 1.0]).to(x.device).float()
            else:
                # if we can't figure out the classification don't reweight
                class_weights = torch.ones(self.n_classes).to(x.device).float()

            loss = nn.CrossEntropyLoss(weight=class_weights)(
                x.squeeze(), classification.long()
            )

            self.y_pred_val.append(x)
            self.y_true_val.append(classification)

        elif self.loss == "sigmoid":
            for i in range(len(x)):
                self.embs_list[i].append(x[i])
            loss = sigmoid_loss_multimodal(x, self.logit_scale, self.logit_bias).mean()
        elif self.loss == "softmax":
            for i in range(len(x)):
                self.embs_list[i].append(x[i])
            loss = clip_loss_multimodal(
                x,
                self.logit_scale,
                self.logit_bias,
            ).mean()
        self.log(
            "val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of the validation epoch.
        """

        if self.regression:
            # Compute R2 Value if regression
            y_true = torch.cat(self.y_true_val, dim=0)
            y_pred = torch.cat(self.y_pred_val, dim=0)
            r2 = (
                1
                - (y_true - y_pred).pow(2).sum() / (y_true - y_true.mean()).pow(2).sum()
            )
            self.log(
                "R2_val",
                r2.cpu(),
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                logger=True,
            )
        elif self.classification:
            # Compute R2 Value if regression
            y_true = torch.cat(self.y_true_val, dim=0)
            y_pred = torch.cat(self.y_pred_val, dim=0)

            y_pred = torch.argmax(y_pred, dim=1)

            y_pred = y_pred.int()  # Ensure it's an integer tensor
            y_true = y_true.int()  # Ensure it's an integer tensor

            f1 = MulticlassFBetaScore(beta=1.0, num_classes=self.n_classes).to(
                y_pred.device
            )(y_pred, y_true)
            self.log(
                "f1_val",
                f1.cpu(),
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                logger=True,
            )
        else:
            # Concatenate all embeddings into single tensors
            for i in range(len(self.embs_list)):
                self.embs_list[i] = torch.cat(self.embs_list[i], dim=0)

            if len(self.combinations) == 2:
                self.log(
                    f"AUC_val",
                    get_AUC(self.embs_list[0], self.embs_list[1]),
                    on_epoch=True,
                    on_step=False,
                    prog_bar=True,
                    logger=True,
                )
            else:
                count = 1
                for i in range(len(self.combinations) - 1):
                    for j in range(i + 1, len(self.combinations)):
                        self.log(
                            f"AUC_val{count}",
                            get_AUC(self.embs_list[i], self.embs_list[j]),
                            on_epoch=True,
                            on_step=False,
                            prog_bar=True,
                            logger=True,
                        )
                        count += 1

            # Delete the embeddings to free up memory
            self.embs_list = None


def initialize_model(
    path: str, combinations: Optional[Any] = None, regression: Optional[Any] = None
) -> LightCurveImageCLIP:
    '''
    Initialize the model with the configuration parameters from the config file stored in path.
    
    Args:
        path (str): Path to the checkpoint file (.ckpt) or the config file (.yaml).
        combinations (Optional[Any]): Combination parameters for the model. If not provided, it will be loaded from the sweep configuration.
        regression (Optional[Bool]): Regression bolean, if True the model is a regression model. If not provided, it will be loaded from the sweep configuration.
    '''
    # Load the sweep configuration file
    config_dir = os.path.dirname(path)
    sweep_config_dir = os.path.dirname(config_dir)
    cfg_extra_args: Dict[str, Any] = YAML(typ="safe").load(
        open(f"{sweep_config_dir}/sweep_config.yaml")
    )["extra_args"]

    if combinations is None:
        combinations = cfg_extra_args["combinations"]
    if regression is None:
        regression = cfg_extra_args.get("regression", False)

    # Load the main configuration file
    cfg: Dict[str, Any] = YAML(typ="safe").load(open(f"{config_dir}/config.yaml"))

    # Setting parameters for the transformer
    transformer_kwargs = {
        "n_out": cfg["n_out"],
        "emb": cfg["emb"],
        "heads": cfg["heads"],
        "depth": cfg["transformer_depth"],
        "dropout": cfg["dropout"],
        "time_norm": cfg["time_norm"],
        "agg": cfg["agg"],
    }

    # Setting parameters for the spectral transformer
    transformer_spectral_kwargs = {
        "n_out": cfg["n_out"],
        "emb": cfg["emb_spectral"],
        "heads": cfg["heads"],
        "depth": cfg["transformer_depth_spectral"],
        "dropout": cfg["dropout"],
        "time_norm": cfg["time_norm_spectral"],
        "agg": cfg["agg_spectral"],
    }

    # Setting parameters for the convolutional model
    conv_kwargs = {
        "dim": cfg.get("cnn_dim", 32),
        "depth": cfg.get("cnn_depth", 2),
        "channels": cfg.get("cnn_channels", 3),
        "kernel_size": cfg.get("cnn_kernel_size", 5),
        "patch_size": cfg.get("cnn_patch_size", 10),
        "n_out": cfg["n_out"],
        "dropout_prob": cfg["dropout"],
    }
    # Create the model instance
    model = LightCurveImageCLIP(
        logit_scale=cfg["logit_scale"],
        lr=cfg["lr"],
        nband=2,
        loss="softmax",
        transformer_kwargs=transformer_kwargs,
        transformer_spectral_kwargs=transformer_spectral_kwargs,
        conv_kwargs=conv_kwargs,
        optimizer_kwargs={},
        combinations=combinations,
        regression=regression,
    )

    return model, combinations, regression, cfg, cfg_extra_args
    

def load_model(
    path: str, path_statedict: Optional[str] = None, combinations: Optional[Any] = None, regression: Optional[Any] = None
) -> LightCurveImageCLIP:
    """
    Load a trained LightCurveImageCLIP model from a checkpoint file.

    Args:
        path (str): Path to the checkpoint file (.ckpt) or the config file (.yaml).
        path_statedict (Optional[str]): Path to the model state dictionary file (.ckpt). If not provided, it will be loaded from path.
        combinations (Optional[Any]): Combination parameters for the model. If not provided, it will be loaded from the sweep configuration.
        regression (Optional[Bool]): Regression bolean, if True the model is a regression model. If not provided, it will be loaded from the sweep configuration.

    Returns:
        LightCurveImageCLIP: The loaded and configured model.
        combinations: Combination parameters for the model.
        regression:
        cfg: dictonary containing yaml
    """
    model, combinations, regression, cfg, cfg_extra_args = initialize_model(
        path, combinations, regression
    )

    # Set the model to the appropriate device (CPU/GPU)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    path_ckpt = path_statedict if path_statedict else path

    assert path_ckpt.endswith(".ckpt"), "The checkpoint file must have a .ckpt extension."

    # Load the model state from the checkpoint file
    model.load_state_dict(
        torch.load(path_ckpt, map_location=torch.device("cpu"))["state_dict"]
    )

    # Set the model to evaluation mode
    model.eval()

    return model, combinations, regression, cfg, cfg_extra_args


def load_pretrain_lc_model(
    pretrain_lc_path: Optional[str], clip_model: nn.Module, freeze_backbone_lc: bool
) -> None:
    """
    Loads a pretrained lightcurve model from a specified path, modifies its state dictionary,
    loads it into a specific component of a given model, and optionally freezes all
    parameters except for specified ones in the model's encoder.

    Args:
    pretrain_lc_path (Optional[str]): Path to the pretrained model's state dict file. If None, no loading is done.
    clip_model (nn.Module): The main model which contains the encoder to be loaded and optionally frozen.
    freeze_backbone_lc (bool): If True, freezes all parameters in the encoder except for 'projection.weight' and 'projection.bias'.

    Returns:
    None
    """
    # Loading up pretrained models
    if pretrain_lc_path:
        pre = torch.load(pretrain_lc_path)
        # Preparing data to be processed by the model
        new_dict = {
            k.replace("net.", ""): v
            for k, v in pre["state_dict"].items()
            if "net." in k
        }
        # Writing the new state dict for the encoder
        clip_model.lightcurve_encoder.load_state_dict(new_dict)

        # Freezing pretrained backbone if required
        if freeze_backbone_lc:
            for name, param in clip_model.lightcurve_encoder.named_parameters():
                if name not in ["projection.weight", "projection.bias"]:
                    param.requires_grad = False
                else:
                    param.requires_grad = True



def load_pretrain_clip_model(
    pretrain_path: Optional[str], clip_model: nn.Module, freeze_backbone: bool
) -> None:
    """
    Loads a pretrained lightcurve model from a specified path, modifies its state dictionary,
    loads it into a specific component of a given model, and optionally freezes all
    parameters except for specified ones in the model's encoder.

    Args:
    pretrain_lc_path (Optional[str]): Path to the pretrained model's state dict file. If None, no loading is done.
    clip_model (nn.Module): The main model which contains the encoder to be loaded and optionally frozen.
    freeze_backbone_lc (bool): If True, freezes all parameters in the encoder except for 'projection.weight' and 'projection.bias'.

    Returns:
    None
    """
    # Loading up pretrained models
    if pretrain_path:
        pre = torch.load(pretrain_path)
        clip_model.load_state_dict(pre["state_dict"])

        # Freezing pretrained backbone if required
        if freeze_backbone:
            for name, param in clip_model.lightcurve_encoder.named_parameters():
                if name not in ["projection.weight", "projection.bias"]:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            for name, param in clip_model.spectral_encoder.named_parameters():
                if name not in ["projection.weight", "projection.bias"]:
                    param.requires_grad = False
                else:
                    param.requires_grad = True


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(self.dropout))
        for i in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(self.dropout))
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class ClipMLP(nn.Module):
    def __init__(self, clip_model, mlp_kwargs, optimizer_kwargs, lr, 
                 combinations=['lightcurve'], 
                 regression=True, 
                 classification=False, 
                 n_classes=5):
        super(ClipMLP, self).__init__()
        
        enc_dim = 0 
        if 'lightcurve' in combinations: enc_dim += clip_model.lightcurve_encoder.projection.out_features
        if 'spectral' in combinations: enc_dim += clip_model.spectral_encoder.projection.out_features

        mlp_kwargs['input_dim'] == enc_dim 

        self.clip_model = clip_model
        self.mlp_model = MLP(**mlp_kwargs)
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.combinations = combinations
        self.regression = regression
        self.classification = classification
        self.n_classes = n_classes

    def forward(self, x_lc=None, t_lc=None, mask_lc=None, x_sp=None, t_sp=None, mask_sp=None):
        x = [] 
        if 'lightcurve' in self.combinations:
            x.append(self.clip_model.lightcurve_embeddings_with_projection(x_lc, t_lc, mask_lc)) 
        if 'spectral' in self.combinations:
            x.append(self.clip_model.spectral_embeddings_with_projection(x_sp, t_sp, mask_sp)) 

        x = torch.cat(x, dim=-1)
        x = self.mlp_model(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(
            self.parameters(), lr=self.lr, **self.optimizer_kwargs
        )
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        (
            x_img,
            x_lc,
            t_lc,
            mask_lc,
            x_sp,
            t_sp,
            mask_sp,
            redshift,
            classification,
        ) = batch
        x = self(x_lc, t_lc, mask_lc, x_sp, t_sp, mask_sp)

        if self.regression:
            loss = nn.MSELoss()(x.squeeze(), redshift)

            # Store the predictions and true values for R2 calculation
            self.y_pred.append(x.flatten())
            self.y_true.append(redshift)

        elif self.classification:
            # matching the (rough) class breakdown of ZTF BTS
            if self.n_classes == 5:
                class_weights = (
                    torch.tensor([0.3, 0.08, 1.0, 0.01, 0.2]).to(x.device).float()
                )
            elif self.n_classes == 3:
                class_weights = torch.tensor([0.33, 0.06, 1.0]).to(x.device).float()
            else:
                # if we can't figure out the classification don't reweight
                class_weights = torch.ones(self.n_classes).to(x.device).float()

            loss = nn.CrossEntropyLoss(weight=class_weights)(
                x.squeeze(), classification.long()
            )

            self.y_pred.append(x)
            self.y_true.append(classification)

        self.log(
            "train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )
        return loss

    def on_train_epoch_start(self):
        if self.regression or self.classification:
            # Initialize empty lists to store predictions and true values
            self.y_true = []
            self.y_pred = []

    def on_train_epoch_end(self) -> None:
        if self.regression:
            # Compute R2
            y_true = torch.cat(self.y_true, dim=0)
            y_pred = torch.cat(self.y_pred, dim=0)
            r2 = (
                1 - (y_true - y_pred).pow(2).sum() / (y_true - y_true.mean()).pow(2).sum()
            )
            self.log(
                "R2_train",
                r2.cpu(),
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                logger=True,
            )

        elif self.classification:
            # Compute F2 score (weighted harmonic mean between precision and recall)
            y_true = torch.cat(self.y_true, dim=0)
            y_pred = torch.cat(self.y_pred, dim=0)

            y_pred = torch.argmax(y_pred, dim=1)

            y_pred = y_pred.int()  # Ensure it's an integer tensor
            y_true = y_true.int()  # Ensure it's an integer tensor

            # beta is the weighting between precision and recall. For now use beta=1, equal weight.
            f1 = MulticlassFBetaScore(beta=1.0, num_classes=self.n_classes).to(
                y_pred.device
            )(y_pred.int(), y_true)
            self.log(
                "f1_train",
                f1.cpu(),
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                logger=True,
            )

    def on_validation_start(self) -> None:
        """
        Called at the beginning of the validation loop.
        """
        # Initialize an empty list to store embeddings
        if self.regression or self.classification:
            self.y_pred_val = []
            self.y_true_val = []

    def validation_step(self, batch, batch_idx):
        (
            x_img,
            x_lc,
            t_lc,
            mask_lc,
            x_sp,
            t_sp,
            mask_sp,
            redshift,
            classification,
        ) = batch
        x = self(x_lc, t_lc, mask_lc, x_sp, t_sp, mask_sp)

        if self.regression:
            loss = nn.MSELoss()(x.squeeze(), redshift)

            # Store the predictions and true values for R2 calculation
            self.y_pred_val.append(x.flatten())
            self.y_true_val.append(redshift)

        elif self.classification:
            # matching the (rough) class breakdown of ZTF BTS - lower for more common classes
            if self.n_classes == 5:
                class_weights = (
                    torch.tensor([0.3, 0.08, 1.0, 0.01, 0.2]).to(x.device).float()
                )
            elif self.n_classes == 3:
                class_weights = torch.tensor([0.33, 0.06, 1.0]).to(x.device).float()
            else:
                # if we can't figure out the classification don't reweight
                class_weights = torch.ones(self.n_classes).to(x.device).float()

            loss = nn.CrossEntropyLoss(weight=class_weights)(
                x.squeeze(), classification.long()
            )

            self.y_pred_val.append(x)
            self.y_true_val.append(classification)

        self.log(
            "val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of the validation epoch.
        """

        if self.regression:
            # Compute R2 Value if regression
            y_true = torch.cat(self.y_true_val, dim=0)
            y_pred = torch.cat(self.y_pred_val, dim=0)
            r2 = (
                1
                - (y_true - y_pred).pow(2).sum() / (y_true - y_true.mean()).pow(2).sum()
            )
            self.log(
                "R2_val",
                r2.cpu(),
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                logger=True,
            )
        elif self.classification:
            # Compute R2 Value if regression
            y_true = torch.cat(self.y_true_val, dim=0)
            y_pred = torch.cat(self.y_pred_val, dim=0)

            y_pred = torch.argmax(y_pred, dim=1)

            y_pred = y_pred.int()  # Ensure it's an integer tensor
            y_true = y_true.int()  # Ensure it's an integer tensor

            f1 = MulticlassFBetaScore(beta=1.0, num_classes=self.n_classes).to(
                y_pred.device
            )(y_pred, y_true)
            self.log(
                "f1_val",
                f1.cpu(),
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                logger=True,
            )