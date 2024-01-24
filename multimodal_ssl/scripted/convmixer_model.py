import torch.nn as nn


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
        print(dropout_prob)

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
