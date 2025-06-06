from typing import Union

import torch.nn as nn


class Residual(nn.Module):
    """
    A residual block that adds the input to the output of a given function.

    Args:
        fn (nn.Module): The function to apply to the input.
    """

    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Forward pass of the residual layer.

        Args:
            x (torch.Tensor): Input tensor
            **kwargs: Additional arguments to pass to the function

        Returns
        -------
            torch.Tensor: Output tensor with residual connection
        """
        return self.fn(x, **kwargs) + x


def ConvBlock(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    bias: bool = True,
    padding: Union[str, int] = "same",
) -> nn.Sequential:
    """
    Creates a convolutional block consisting of batch normalization, GELU activation, and 1D convolution.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolutional kernel
        stride: Stride value for the convolutional operation (default: 1)
        dilation: Dilation value for the convolutional operation (default: 1)
        bias: Whether to include bias in the convolutional operation (default: True)
        padding: Padding mode for the convolutional operation (default: "same")

    Returns
    -------
        A sequential module consisting of batch normalization, GELU activation, and 1D convolution
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
    A tower of convolutional layers followed by max pooling.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolutional kernel
        layers: Number of convolutional layers in the tower (default: 5)
        stride: Stride value for the convolutional layers (default: 1)
        dilation: Dilation value for the convolutional layers (default: 1)
        bias: Whether to include bias in the convolutional layers (default: True)
        padding: Padding mode for the convolutional layers (default: "same")
        pooling_size: Size of the max pooling window (default: 2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        layers: int = 5,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        padding: Union[str, int] = "same",
        pooling_size: int = 2,
        pooling_method: str = "max",
    ):
        super().__init__()

        # Create a list of convolutional blocks with residual connections
        conv_blocks = []
        for _ in range(layers):
            conv_blocks.append(
                nn.Sequential(
                    ConvBlock(in_channels, out_channels, kernel_size, stride, dilation, bias, padding),
                    Residual(ConvBlock(out_channels, out_channels, 1, stride, dilation, bias, padding)),
                    nn.MaxPool1d(pooling_size) if pooling_method == "max" else nn.AvgPool1d(pooling_size),
                )
            )

        # Combine all blocks into a sequential module
        self.layers = nn.Sequential(*conv_blocks)

    def forward(self, x):
        """
        Forward pass of the ConvTower.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, sequence_length]

        Returns
        -------
            torch.Tensor: Output tensor with reduced sequence length due to pooling
        """
        return self.layers(x)


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for attending over sequence and ATAC data."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """
        Forward pass through the cross-attention layer.

        Args:
            query (torch.Tensor): Query tensor of shape [batch_size, seq_len, embed_dim]
            key_value (torch.Tensor): Key and value tensor of shape [batch_size, seq_len, embed_dim]

        Returns
        -------
            torch.Tensor: Attended output tensor of shape [batch_size, seq_len, embed_dim]
        """
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        query = self.norm1(query + self.dropout(attn_output))
        ffn_output = self.ffn(query)
        query = self.norm2(query + self.dropout(ffn_output))
        return query
