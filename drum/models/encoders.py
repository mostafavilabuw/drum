import torch.nn as nn

from .layers import ConvBlock, ConvTower, Residual


class BasicEncoder(nn.Module):
    """
    A basic encoder class.

    This class serves as a base class for implementing different encoder architectures.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes
    ----------
        None

    Methods
    -------
        encode: Abstract method to be implemented by subclasses.
        forward: Forward pass method.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def encode(self):
        """
        Abstract method to be implemented by subclasses.

        This method should implement the encoding logic specific to each encoder architecture.

        Returns
        -------
            NotImplementedError: This method should be overridden by subclasses.

        """
        print("Implement the basic encoder here.")
        return NotImplementedError

    def forward(self, x):
        """
        Forward pass method.

        Args:
            x: Input tensor.

        Returns
        -------
            Tensor: Encoded representation of the input tensor.

        """
        return self.encode(x)


class SeqEncoder(BasicEncoder):
    """
    Sequence Encoder class.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        first_kernel_size (int, optional): Size of the first convolutional kernel. Defaults to 15.
        kernel_size (int, optional): Size of the convolutional kernels in the ConvTower. Defaults to 5.
        stride (int, optional): Stride value for the convolutional layers. Defaults to 1.
        dilation (int, optional): Dilation value for the convolutional layers. Defaults to 1.
        bias (bool, optional): Whether to include bias in the convolutional layers. Defaults to False.
        padding (str, optional): Padding mode for the convolutional layers. Defaults to "same".
        pooling_size (int, optional): Size of the max pooling kernel. Defaults to 2.
        layers (int, optional): Number of layers in the ConvTower. Defaults to 5.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        first_kernel_size=15,
        kernel_size=5,
        stride=1,
        dilation=1,
        bias=False,
        padding="same",
        pooling_size=2,
        layers=5,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels, first_kernel_size, stride, dilation, bias, padding),
            Residual(ConvBlock(out_channels, out_channels, 1, stride, dilation, bias, padding)),
            nn.MaxPool1d(pooling_size),
        )

        self.conv_tower = ConvTower(
            out_channels, out_channels, kernel_size, layers - 1, stride, dilation, bias, padding, pooling_size
        )

    def encode(self, x):
        """
        Encodes the input sequence.

        Args:
            x (torch.Tensor): Input sequence tensor.

        Returns
        -------
            torch.Tensor: Encoded sequence tensor.
        """
        x = self.conv(x)
        x = self.conv_tower(x)
        return x

    def forward(self, x):
        """
        Forward pass of the SeqEncoder.

        Args:
            x (torch.Tensor): Input sequence tensor.

        Returns
        -------
            torch.Tensor: Encoded sequence tensor.
        """
        return self.encode(x)


def ChromatinEncoder():
    """_summary_

    The ChromatinEncoder class to encode the chromatin data.
    """
    return NotImplementedError


def GeneExpressionEncoder():
    """_summary_

    The GeneExpressionEncoder class to encode the gene expression data.
    """
    return NotImplementedError


def HistoneModificationEncoder():
    """_summary_

    The HistoneModificationEncoder class to encode the histone modification data.
    """
    return NotImplementedError


def ChromatinDecoder():
    """_summary_

    The ChromatinDecoder class to decode the chromatin data.
    """
    return NotImplementedError


def GeneExpressionDecoder():
    """_summary_

    The GeneExpressionDecoder class to decode the gene expression data.
    """
    return NotImplementedError


def HistoneModificationDecoder():
    """_summary_

    The HistoneModificationDecoder class to decode the histone modification data.
    """
    return NotImplementedError


def MultiModalFusion():
    """_summary_

    The MultiModalFusion class to fuse the multimodal data.
    """
    return NotImplementedError
