import torch.nn as nn


class Residual(nn.Module):
    """
    A residual block that adds the input to the output of a given function.

    Args:
        fn (nn.Module): The function to apply to the input.

    Returns
    -------
        torch.Tensor: The output tensor after applying the function and adding the input.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        """Forward pass of the layer."""
        return self.fn(x, **kwargs) + x


def ConvBlock(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride=1,
    dilation=1,
    bias=True,
    padding="same",
):
    """
    Creates a convolutional block consisting of batch normalization, 1D convolution, and GELU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int, optional): Stride value for the convolutional operation. Defaults to 1.
        dilation (int, optional): Dilation value for the convolutional operation. Defaults to 1.
        bias (bool, optional): Whether to include bias in the convolutional operation. Defaults to True.
        padding (str, optional): Padding mode for the convolutional operation. Defaults to "same".

    Returns
    -------
        nn.Sequential: A sequential module consisting of batch normalization, 1D convolution, and GELU activation.
    """
    return nn.Sequential(
        nn.BatchNorm1d(in_channels),
        nn.GELU(),
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=bias,
        ),
    )


class ConvTower(nn.Module):
    """
    ConvTower is a class that represents a tower of convolutional layers followed by max pooling.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        layers (int, optional): Number of convolutional layers in the tower. Defaults to 5.
        stride (int, optional): Stride value for the convolutional layers. Defaults to 1.
        dilation (int, optional): Dilation value for the convolutional layers. Defaults to 1.
        bias (bool, optional): Whether to include bias in the convolutional layers. Defaults to True.
        padding (str, optional): Padding mode for the convolutional layers. Defaults to "same".
        pooling_size (int, optional): Size of the max pooling window. Defaults to 2.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        layers=5,
        stride=1,
        dilation=1,
        bias=True,
        padding="same",
        pooling_size=2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                nn.Sequential(
                    ConvBlock(in_channels, out_channels, kernel_size, stride, dilation, bias, padding),
                    Residual(ConvBlock(out_channels, out_channels, 1, stride, dilation, bias, padding)),
                    nn.MaxPool1d(pooling_size),
                )
            )

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """Forward pass of the ConvTower."""
        return self.layers(x)
