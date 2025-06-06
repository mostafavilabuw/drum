# No need to import Dict from typing anymore
import torch
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
        super().__init__()

    def encode(self, x):
        """
        Abstract method to be implemented by subclasses.

        This method should implement the encoding logic specific to each encoder architecture.

        Args:
            x (torch.Tensor): Input tensor to encode.

        Returns
        -------
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the encode method")

    def forward(self, x):
        """
        Forward pass method.

        Args:
            x (torch.Tensor): Input tensor.

        Returns
        -------
            torch.Tensor: Encoded representation of the input tensor.
        """
        return self.encode(x)


class SeqEncoder(BasicEncoder):
    """
    Sequence Encoder class for processing DNA sequences.

    This encoder uses a series of convolutional blocks and a convolutional tower
    to extract features from sequence data.

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
        in_channels: int,
        out_channels: int,
        first_kernel_size: int = 15,
        kernel_size: int = 5,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        padding: str = "same",
        pooling_size: int = 2,
        layers: int = 5,
        pooling_method: str = "max",
    ):
        super().__init__()
        # Initial convolutional block with max pooling
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels, first_kernel_size, stride, dilation, bias, padding),
            Residual(ConvBlock(out_channels, out_channels, 1, stride, dilation, bias, padding)),
            nn.MaxPool1d(pooling_size) if pooling_method == "max" else nn.AvgPool1d(pooling_size),
        )

        # Convolutional tower for deep feature extraction
        self.conv_tower = ConvTower(
            out_channels,
            out_channels,
            kernel_size,
            layers - 1,
            stride,
            dilation,
            bias,
            padding,
            pooling_size,
            pooling_method,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input sequence.

        Args:
            x (torch.Tensor): Input sequence tensor of shape [batch_size, in_channels, sequence_length]

        Returns
        -------
            torch.Tensor: Encoded sequence tensor of shape [batch_size, out_channels, encoded_length]
        """
        x = self.conv(x)
        x = self.conv_tower(x)
        return x


class ChromatinEncoder(BasicEncoder):
    """
    Chromatin Encoder for processing chromatin accessibility data (e.g., ATAC-seq).

    This encoder is designed to extract features from chromatin accessibility signals
    across genomic regions.

    Args:
        in_channels (int): Number of input channels (typically 1 for ATAC-seq data)
        out_channels (int): Number of output feature channels
        kernel_size (int, optional): Size of the convolutional kernels. Defaults to 15.
        layers (int, optional): Number of convolutional layers. Defaults to 5.
        pooling_size (int, optional): Size of the max pooling kernel. Defaults to 2.
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.
        **kwargs: Additional keyword arguments for the convolutional layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        layers: int = 5,
        pooling_size: int = 2,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__()

        # Initial convolutional block
        self.initial_conv = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size, **kwargs),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(pooling_size),
        )

        # Deep feature extraction using ConvTower
        self.conv_tower = ConvTower(
            out_channels, out_channels, kernel_size=5, layers=layers - 1, pooling_size=pooling_size, **kwargs
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes chromatin accessibility data.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, sequence_length]

        Returns
        -------
            torch.Tensor: Encoded chromatin features
        """
        x = self.initial_conv(x)
        x = self.conv_tower(x)
        return x


class GeneExpressionEncoder(BasicEncoder):
    """
    Gene Expression Encoder for processing gene expression data.

    This encoder is designed to extract features from gene expression data,
    which can be used for various downstream tasks.

    Args:
        input_dim (int): Number of input genes
        hidden_dims (list, optional): List of hidden dimensions. Defaults to [512, 256, 128].
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.
        activation (nn.Module, optional): Activation function. Defaults to nn.ReLU().
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        dropout_rate: float = 0.2,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        # Use default if None is provided
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        layers = []
        prev_dim = input_dim

        # Create encoder layers
        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, hidden_dim), activation, nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout_rate)]
            )
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes gene expression data.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]

        Returns
        -------
            torch.Tensor: Encoded gene expression features of shape [batch_size, output_dim]
        """
        return self.encoder(x)


class HistoneModificationEncoder(BasicEncoder):
    """
    Histone Modification Encoder for processing histone mark data (e.g., ChIP-seq).

    This encoder is designed to extract features from multiple histone modification signals
    across genomic regions.

    Args:
        in_channels (int): Number of input channels (number of histone marks)
        out_channels (int): Number of output feature channels
        kernel_size (int, optional): Size of the convolutional kernels. Defaults to 15.
        layers (int, optional): Number of convolutional layers. Defaults to 4.
        pooling_size (int, optional): Size of the max pooling kernel. Defaults to 2.
        **kwargs: Additional keyword arguments for the convolutional layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        layers: int = 4,
        pooling_size: int = 2,
        **kwargs,
    ):
        super().__init__()

        # Initial conv layer to process multiple histone marks in parallel
        self.initial_conv = ConvBlock(in_channels, out_channels, kernel_size, **kwargs)

        # Process each histone mark separately then combine
        self.mark_specific_convs = nn.ModuleList(
            [ConvBlock(1, out_channels // in_channels, kernel_size // 2, **kwargs) for _ in range(in_channels)]
        )

        # Feature fusion layer
        self.fusion_conv = ConvBlock(
            out_channels + (out_channels // in_channels) * in_channels, out_channels, 1, **kwargs
        )

        # Deep feature extraction
        self.conv_tower = ConvTower(
            out_channels, out_channels, kernel_size=5, layers=layers, pooling_size=pooling_size, **kwargs
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes histone modification data.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, sequence_length]

        Returns
        -------
            torch.Tensor: Encoded histone modification features
        """
        # Global processing of all marks together
        global_features = self.initial_conv(x)

        # Process each histone mark separately
        mark_features = []
        for i in range(x.size(1)):  # For each histone mark channel
            mark = x[:, i : i + 1]  # Extract single channel
            mark_features.append(self.mark_specific_convs[i](mark))

        # Concatenate global and mark-specific features
        mark_features = torch.cat(mark_features, dim=1)
        combined = torch.cat([global_features, mark_features], dim=1)

        # Fuse features and apply conv tower
        x = self.fusion_conv(combined)
        x = self.conv_tower(x)

        return x


class MultiModalFusion(nn.Module):
    """
    Multi-Modal Fusion module for combining features from different modalities.

    This module implements various fusion strategies for integrating features
    from multiple data modalities.

    Args:
        input_dims (dict): Dictionary of input dimensions for each modality
        output_dim (int): Dimension of the fused output
        fusion_method (str): Fusion method to use ('concat', 'attention', or 'gated')
        hidden_dim (int, optional): Hidden dimension for attention/gating. Defaults to 256.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.2.
    """

    def __init__(
        self,
        input_dims: dict,
        output_dim: int,
        fusion_method: str = "attention",
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.modalities = list(input_dims.keys())
        self.fusion_method = fusion_method

        # Create projection layers for each modality
        self.projections = nn.ModuleDict(
            {
                modality: nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate))
                for modality, dim in input_dims.items()
            }
        )

        if fusion_method == "concat":
            # Simple concatenation followed by projection
            self.fusion_layer = nn.Sequential(
                nn.Linear(hidden_dim * len(input_dims), output_dim), nn.ReLU(), nn.Dropout(dropout_rate)
            )

        elif fusion_method == "attention":
            # Attention-based fusion
            self.query_layers = nn.ModuleDict(
                {modality: nn.Linear(hidden_dim, hidden_dim) for modality in input_dims.keys()}
            )
            self.key_layers = nn.ModuleDict(
                {modality: nn.Linear(hidden_dim, hidden_dim) for modality in input_dims.keys()}
            )
            self.value_layers = nn.ModuleDict(
                {modality: nn.Linear(hidden_dim, hidden_dim) for modality in input_dims.keys()}
            )
            self.attention_scale = hidden_dim**0.5
            self.output_layer = nn.Linear(hidden_dim, output_dim)

        elif fusion_method == "gated":
            # Gated fusion
            self.gate_layers = nn.ModuleDict(
                {
                    modality: nn.Sequential(nn.Linear(hidden_dim * len(input_dims), 1), nn.Sigmoid())
                    for modality in input_dims.keys()
                }
            )
            self.output_layer = nn.Linear(hidden_dim, output_dim)

        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for multi-modal fusion.

        Args:
            inputs (dict[str, torch.Tensor]): Dictionary of input tensors for each modality

        Returns
        -------
            torch.Tensor: Fused multi-modal features
        """
        # Project each modality
        projected = {modality: self.projections[modality](inputs[modality]) for modality in self.modalities}

        if self.fusion_method == "concat":
            # Concatenate all modalities and project
            concatenated = torch.cat([projected[m] for m in self.modalities], dim=1)
            return self.fusion_layer(concatenated)

        elif self.fusion_method == "attention":
            # Compute attention weights between modalities
            fused_features = 0
            for target_modality in self.modalities:
                queries = self.query_layers[target_modality](projected[target_modality])

                # Compute attention from this modality to all others
                attention_weights = []
                for source_modality in self.modalities:
                    keys = self.key_layers[source_modality](projected[source_modality])
                    scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.attention_scale
                    attention_weights.append(torch.softmax(scores, dim=-1))

                # Apply attention weights to values
                attended_values = 0
                for i, source_modality in enumerate(self.modalities):
                    values = self.value_layers[source_modality](projected[source_modality])
                    attended_values += torch.matmul(attention_weights[i], values)

                fused_features += attended_values

            return self.output_layer(fused_features)

        elif self.fusion_method == "gated":
            # Concatenate for gate computation
            concatenated = torch.cat([projected[m] for m in self.modalities], dim=1)

            # Compute gated combination
            gated_features = 0
            for modality in self.modalities:
                gate = self.gate_layers[modality](concatenated)
                gated_features += gate * projected[modality]

            return self.output_layer(gated_features)
