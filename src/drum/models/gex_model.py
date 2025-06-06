# filepath: /homes/gws/tuxm/Project/drum-dev/src/drum/models/gex_model.py
import torch
import torch.nn as nn

from .base import BaseDrumModel
from .encoders import SeqEncoder
from .layers import CrossAttentionLayer


class GatedDrumGEX(BaseDrumModel):
    """A gated DRUM model for gene expression prediction."""

    def __init__(
        self,
        seq_len,
        first_kernel_size=15,
        kernel_size=5,
        num_kernel=256,
        conv_layers=6,
        pooling_size=2,
        lr=1e-3,
        weight_decay=1e-5,
        loss="mse",
        log_input=True,
        **kwargs,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, loss=loss, log_input=log_input)
        self.save_hyperparameters()

        self.DNAEncoder = SeqEncoder(
            4,
            num_kernel,
            first_kernel_size,
            kernel_size,
            pooling_size=pooling_size,
            layers=conv_layers,
            **kwargs,
        )

        # view chromatin accessibility as a sequence of 1D
        self.atac_encoder = SeqEncoder(
            1,
            num_kernel,
            first_kernel_size,
            kernel_size,
            pooling_size=pooling_size,
            layers=conv_layers,
            **kwargs,
        )

        self.outdim = num_kernel * (seq_len // pooling_size**conv_layers)
        self.fc_output = nn.Sequential(nn.Linear(self.outdim, 1))

    def forward(self, seq, atac):
        """Forward pass of the model."""
        x_seq = self.DNAEncoder(seq)
        x_atac = self.atac_encoder(atac)
        x = x_seq * x_atac

        x = x.flatten(1)
        x = self.fc_output(x)

        x = x.squeeze(1)
        return x

    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        seq_idx, atac_idx, seq, atac, gex = batch

        y_hat = self.forward(seq, atac)
        loss = self.compute_loss(y_hat, gex)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        """Validation step for the model."""
        seq_idx, atac_idx, seq, atac, gex = batch

        y_hat = self.forward(seq, atac)
        loss = self.compute_loss(y_hat, gex)

        self.log("val_loss", loss)

        self.validation_step_outputs.append((seq_idx, atac_idx, y_hat, gex))
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Prediction step for the model."""
        seq_idx, atac_idx, seq, atac, gex = batch

        y_hat = self.forward(seq, atac)
        return y_hat

    def test_step(self, batch, batch_idx):
        """Test step for the model."""
        seq_idx, atac_idx, seq, atac, gex = batch

        y_hat = self.forward(seq, atac)
        loss = self.compute_loss(y_hat, gex)
        self.log("test_loss", loss)

        self.test_step_outputs.append((seq_idx, atac_idx, y_hat, gex))
        return loss

    def on_validation_epoch_end(self):
        """Calculate and log metrics at the end of each validation epoch."""
        if not self.validation_step_outputs:
            print("Validation step outputs are empty. Skipping metric calculation.")
            return
        self._aggregate_and_log_metrics(self.validation_step_outputs, phase="validation")
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        """Calculate and log metrics at the end of each test epoch."""
        if not self.test_step_outputs:
            print("Test step outputs are empty. Skipping metric calculation.")
            return
        self._aggregate_and_log_metrics(self.test_step_outputs, phase="test")
        self.test_step_outputs.clear()


class ProbDecoupleGatedDrumGEX(GatedDrumGEX):
    """A probabilistic gated DRUM model for gene expression prediction with decoupled outputs."""

    def __init__(
        self,
        seq_len,
        first_kernel_size=15,
        kernel_size=5,
        num_kernel=256,
        conv_layers=6,
        pooling_size=2,
        lr=1e-3,
        h_dim=128,
        h_layers=1,
        weight_decay=1e-5,
        dropout=0.2,
        loss="PoissonNLL",
        **kwargs,
    ):
        super().__init__(
            seq_len,
            first_kernel_size=first_kernel_size,
            kernel_size=kernel_size,
            num_kernel=num_kernel,
            conv_layers=conv_layers,
            pooling_size=pooling_size,
            lr=lr,
            weight_decay=weight_decay,
            loss=loss,
            **kwargs,
        )
        self.save_hyperparameters()

        self.DNAEncoder = SeqEncoder(
            4,
            num_kernel,
            first_kernel_size,
            kernel_size,
            pooling_size=pooling_size,
            layers=conv_layers,
            **kwargs,
        )

        # taking
        self.atac_encoder = SeqEncoder(
            1,
            num_kernel,
            first_kernel_size,
            kernel_size,
            pooling_size=pooling_size,
            layers=conv_layers,
            **kwargs,
        )

        self.outdim = num_kernel * (seq_len // pooling_size**conv_layers)
        outdim = self.outdim

        self.fc_seq = nn.ModuleList()
        self.fc_atac = nn.ModuleList()

        for _ in range(h_layers):
            self.fc_seq.append(nn.Sequential(nn.Linear(outdim, h_dim), nn.ReLU(), nn.Dropout(dropout)))
            outdim = h_dim
            self.fc_atac.append(nn.Sequential(nn.Linear(outdim, h_dim), nn.ReLU(), nn.Dropout(dropout)))
            outdim = h_dim

        self.fc_seq.append(nn.Sequential(nn.Linear(outdim, 1), nn.ReLU()))
        self.fc_atac.append(nn.Sequential(nn.Linear(outdim, 1), nn.ReLU()))

        if loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss == "PoissonNLL":
            self.loss_fn = nn.PoissonNLLLoss(log_input=False)

    def forward(self, seq, atac):
        """Forward pass of the model."""
        x_seq = self.DNAEncoder(seq)
        x_atac = self.atac_encoder(atac)
        x_seq_atac = x_seq * x_atac

        x_seq = x_seq.flatten(1)
        for fc in self.fc_seq:
            x_seq = fc(x_seq)

        x_seq_atac = x_seq_atac.flatten(1)
        for fc in self.fc_atac:
            x_seq_atac = fc(x_seq_atac)

        x = x_seq.squeeze(1) + x_seq_atac.squeeze(1)
        return x

    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        seq_idx, atac_idx, seq, atac, gex = batch

        y_hat = self.forward(seq, atac)
        loss = self.loss_fn(y_hat, gex.exp())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        seq_idx, atac_idx, seq, atac, gex = batch

        y_hat = self.forward(seq, atac)
        loss = self.loss_fn(y_hat, gex.exp())
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step for the model."""
        seq_idx, atac_idx, seq, atac, gex = batch

        y_hat = self.forward(seq, atac)
        loss = self.loss_fn(y_hat, gex)
        self.log("test_loss", loss)
        return loss


class DecoupleGatedDrumGEX(GatedDrumGEX):
    """A gated DRUM model for gene expression prediction with decoupled outputs."""

    def __init__(
        self,
        seq_len,
        first_kernel_size=15,
        kernel_size=5,
        num_kernel=256,
        conv_layers=6,
        pooling_size=2,
        lr=1e-3,
        h_dim=256,
        h_layers=1,
        weight_decay=1e-5,
        dropout=0.2,
        loss="mse",
        log_input=False,
        **kwargs,
    ):
        super().__init__(
            seq_len,
            first_kernel_size=first_kernel_size,
            kernel_size=kernel_size,
            num_kernel=num_kernel,
            conv_layers=conv_layers,
            pooling_size=pooling_size,
            lr=lr,
            weight_decay=weight_decay,
            loss=loss,
            log_input=log_input,
            **kwargs,
        )
        self.save_hyperparameters()

        self.DNAEncoder = SeqEncoder(
            4,
            num_kernel,
            first_kernel_size,
            kernel_size,
            pooling_size=pooling_size,
            layers=conv_layers,
            **kwargs,
        )

        # taking
        self.atac_encoder = SeqEncoder(
            1,
            num_kernel,
            first_kernel_size,
            kernel_size,
            pooling_size=pooling_size,
            layers=conv_layers,
            **kwargs,
        )

        self.outdim = num_kernel * (seq_len // pooling_size**conv_layers)
        outdim = self.outdim

        self.fc_seq = nn.ModuleList()
        self.fc_atac = nn.ModuleList()

        # Create sequential layers for sequence and ATAC data processing
        for _ in range(h_layers):
            self.fc_seq.append(nn.Sequential(nn.Linear(outdim, h_dim), nn.ReLU(), nn.Dropout(dropout)))
            outdim = h_dim
            self.fc_atac.append(nn.Sequential(nn.Linear(outdim, h_dim), nn.ReLU(), nn.Dropout(dropout)))
            outdim = h_dim

        # Output layers for sequence and ATAC data
        self.fc_seq.append(nn.Sequential(nn.Linear(outdim, 1), nn.ReLU()))
        self.fc_atac.append(nn.Sequential(nn.Linear(outdim, 1), nn.ReLU()))

        # Loss function initialization
        if loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss == "PoissonNLL":
            self.loss_fn = nn.PoissonNLLLoss(log_input=False)

    def forward(self, seq, atac):
        """Forward pass of the model."""
        # Extract features from sequence and ATAC data
        x_seq = self.DNAEncoder(seq)
        x_seq = x_seq.flatten(1)  # Flatten the output from the DNA encoder

        x_atac = self.atac_encoder(atac)
        x_atac = x_atac.flatten(1)  # Flatten the output from the ATAC encoder

        # Process sequence features
        for fc in self.fc_seq:
            x_seq = fc(x_seq)

        # Process ATAC features
        for fc in self.fc_atac:
            x_atac = fc(x_atac)

        # Combine the processed features
        x = x_seq.squeeze(1) + x_atac.squeeze(1)
        return x

    def compute_loss(self, y_hat, y):
        """Compute the loss for the model."""
        loss_type = self.hparams.loss
        log_input = self.hparams.log_input

        # Handle MSE loss with log transformation if needed
        if loss_type == "mse":
            if log_input:
                return self.loss_fn(y_hat, torch.log1p(y))
            else:
                return self.loss_fn(y_hat, y)
        # Handle PoissonNLL loss
        elif loss_type == "PoissonNLL":
            if log_input:
                y_hat = torch.exp(y_hat)
            return self.loss_fn(y_hat, y)
        # Default behavior
        return self.loss_fn(y_hat, y)


class DrumMultiTaskGEX(BaseDrumModel):
    """A multi-task gated DRUM model for gene expression prediction across multiple cell states."""

    def __init__(
        self,
        seq_len,
        first_kernel_size=15,
        kernel_size=5,
        num_kernel=256,
        conv_layers=6,
        pooling_size=2,
        lr=1e-3,
        weight_decay=1e-5,
        loss="mse",
        h_dim=256,
        h_layers=1,
        log_input=False,
        num_tasks=10,
        **kwargs,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, loss=loss, log_input=log_input)
        self.save_hyperparameters()

        self.DNAEncoder = SeqEncoder(
            4,
            num_kernel,
            first_kernel_size,
            kernel_size,
            pooling_size=pooling_size,
            layers=conv_layers,
            **kwargs,
        )

        self.outdim = num_kernel * (seq_len // pooling_size**conv_layers)
        outdim = self.outdim
        self.fc_output = nn.ModuleList()

        for _ in range(h_layers):
            self.fc_output.append(nn.Sequential(nn.Linear(outdim, h_dim), nn.ReLU(), nn.Dropout(0.2)))
            outdim = h_dim

        self.fc_output.append(nn.Sequential(nn.Linear(outdim, num_tasks), nn.ReLU()))
        # Output layer adapted for multi-task predictions (one output per task/cell state)

        # Define loss based on input argument
        if loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss == "PoissonNLL":
            self.loss_fn = nn.PoissonNLLLoss(log_input=False)

    def forward(self, seq):
        """Forward pass of the model."""
        x_seq = self.DNAEncoder(seq)
        x = x_seq.flatten(1)
        for fc in self.fc_output:
            x = fc(x)

        return x  # Output shape: (batch_size, num_tasks)

    def compute_loss(self, y_hat, y):
        """Compute the loss for the model."""
        loss_type = self.hparams.loss
        log_input = self.hparams.log_input

        if loss_type == "PoissonNLL":
            if log_input:
                y_hat = torch.exp(y_hat)  # Convert from log space if needed
            return self.loss_fn(y_hat, y)
        elif loss_type == "mse":
            if not log_input:
                y_hat = torch.exp(y_hat) - 1  # Reverse the log1p transform if needed
            return self.loss_fn(y_hat, y)
        return self.loss_fn(y_hat, y)

    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        seq_idx, seq, gex = batch

        y_hat = self.forward(seq)  # Predict gene expression for all tasks
        loss = self.compute_loss(y_hat, gex)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        """Validation step for the model."""
        seq_idx, seq, gex = batch

        y_hat = self.forward(seq)
        loss = self.compute_loss(y_hat, gex)

        self.log("val_loss", loss)

        self.validation_step_outputs.append((seq_idx, y_hat, gex))
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Prediction step for the model."""
        seq_idx, seq, gex = batch

        y_hat = self.forward(seq)
        return y_hat

    def test_step(self, batch, batch_idx):
        """Test step for the model."""
        seq_idx, seq, gex = batch

        y_hat = self.forward(seq)
        loss = self.compute_loss(y_hat, gex)
        self.log("test_loss", loss)

        self.test_step_outputs.append((seq_idx, y_hat, gex))
        return loss

    def on_test_epoch_end(self):
        """Calculate and log metrics at the end of each test epoch."""
        self._aggregate_and_log_metrics(self.test_step_outputs, phase="test")
        self.test_step_outputs.clear()


class MeanCellDrumGEX(GatedDrumGEX):
    """A gated DRUM model that separately predicts mean gene expression and cell-specific differences.

    DNA sequence is used to predict the mean gene expression across all cells,
    while ATAC data is used to predict cell-specific differences from the mean.
    """

    def __init__(
        self,
        seq_len,
        first_kernel_size=15,
        kernel_size=5,
        num_kernel=256,
        conv_layers=6,
        pooling_size=2,
        lr=1e-3,
        h_dim=256,
        h_layers=1,
        weight_decay=1e-5,
        dropout=0.2,
        loss="mse",
        log_input=True,
        mean_loss_weight=1.0,
        diff_loss_weight=1.0,
        **kwargs,
    ):
        super().__init__(
            seq_len,
            first_kernel_size=first_kernel_size,
            kernel_size=kernel_size,
            num_kernel=num_kernel,
            conv_layers=conv_layers,
            pooling_size=pooling_size,
            lr=lr,
            weight_decay=weight_decay,
            loss=loss,
            log_input=log_input,
            **kwargs,
        )
        self.save_hyperparameters()

        # DNA encoder for mean expression prediction
        self.DNAEncoder = SeqEncoder(
            4,
            num_kernel,
            first_kernel_size,
            kernel_size,
            pooling_size=pooling_size,
            layers=conv_layers,
            **kwargs,
        )

        # ATAC encoder for cell-specific differences
        self.atac_encoder = SeqEncoder(
            1,
            num_kernel,
            first_kernel_size,
            kernel_size,
            pooling_size=pooling_size,
            layers=conv_layers,
            **kwargs,
        )

        self.outdim = num_kernel * (seq_len // pooling_size**conv_layers)
        outdim = self.outdim

        # Network for DNA (mean expression) prediction
        self.fc_mean = nn.ModuleList()
        for _ in range(h_layers):
            self.fc_mean.append(nn.Sequential(nn.Linear(outdim, h_dim), nn.ReLU(), nn.Dropout(dropout)))
            outdim = h_dim
        self.fc_mean.append(nn.Sequential(nn.Linear(outdim, 1)))

        # Network for ATAC (cell-specific differences) prediction
        outdim = self.outdim
        self.fc_diff = nn.ModuleList()
        for _ in range(h_layers):
            self.fc_diff.append(nn.Sequential(nn.Linear(outdim, h_dim), nn.ReLU(), nn.Dropout(dropout)))
            outdim = h_dim
        self.fc_diff.append(nn.Sequential(nn.Linear(outdim, 1)))

        # Loss weights
        self.mean_loss_weight = mean_loss_weight
        self.diff_loss_weight = diff_loss_weight

    def forward(self, seq, atac):
        """Forward pass of the model.

        Args:
            seq: DNA sequence data
            atac: ATAC-seq data
            mean_expr: Optional mean expression value (used during training)

        Returns
        -------
            tuple: (final prediction, mean prediction, diff prediction)
        """
        # Extract features from sequence and ATAC data
        x_seq = self.DNAEncoder(seq)
        x_atac = self.atac_encoder(atac)

        # Process sequence features for mean prediction
        x_seq_flat = x_seq.flatten(1)
        x_mean = x_seq_flat
        for fc in self.fc_mean:
            x_mean = fc(x_mean)
        x_mean = x_mean.squeeze(1)

        # Process ATAC features for cell-specific difference prediction
        x_atac_flat = x_atac.flatten(1)
        x_diff = x_atac_flat
        for fc in self.fc_diff:
            x_diff = fc(x_diff)
        x_diff = x_diff.squeeze(1)

        # Final prediction is mean + difference
        x_final = x_mean + x_diff

        return x_final, x_mean, x_diff

    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        seq_idx, atac_idx, seq, atac, gex, mean_expr = batch

        # Forward pass
        y_hat_final, y_hat_mean, y_hat_diff = self.forward(seq, atac)

        # Calculate target difference (actual - mean)
        diff_target = gex - mean_expr

        # Compute losses for mean and difference predictions
        mean_loss = self.compute_loss(y_hat_mean, mean_expr)
        diff_loss = self.compute_loss(y_hat_diff, diff_target)
        total_loss = self.compute_loss(y_hat_final, gex)

        # Weighted loss
        combined_loss = self.mean_loss_weight * mean_loss + self.diff_loss_weight * diff_loss

        # Log losses
        self.log("train_total_loss", total_loss)
        self.log("train_mean_loss", mean_loss)
        self.log("train_diff_loss", diff_loss)
        self.log("train_combined_loss", combined_loss)

        return combined_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        """Validation step for the model."""
        seq_idx, atac_idx, seq, atac, gex, mean_expr = batch

        # Forward pass
        y_hat_final, y_hat_mean, y_hat_diff = self.forward(seq, atac)

        # Calculate target difference
        diff_target = gex - mean_expr

        # Compute losses
        mean_loss = self.compute_loss(y_hat_mean, mean_expr)
        diff_loss = self.compute_loss(y_hat_diff, diff_target)
        total_loss = self.compute_loss(y_hat_final, gex)

        combined_loss = self.mean_loss_weight * mean_loss + self.diff_loss_weight * diff_loss

        # Log losses
        self.log("val_total_loss", total_loss)
        self.log("val_mean_loss", mean_loss)
        self.log("val_diff_loss", diff_loss)
        self.log("val_combined_loss", combined_loss)

        # Store outputs for correlation calculation - format compatible with parent class
        self.validation_step_outputs.append((seq_idx, atac_idx, y_hat_final, gex))

        return combined_loss

    def test_step(self, batch, batch_idx):
        """Test step for the model."""
        seq_idx, atac_idx, seq, atac, gex, mean_expr = batch

        # Forward pass
        y_hat_final, y_hat_mean, y_hat_diff = self.forward(seq, atac)

        # Calculate target difference
        diff_target = gex - mean_expr

        # Compute losses
        mean_loss = self.compute_loss(y_hat_mean, mean_expr)
        diff_loss = self.compute_loss(y_hat_diff, diff_target)
        total_loss = self.compute_loss(y_hat_final, gex)

        combined_loss = self.mean_loss_weight * mean_loss + self.diff_loss_weight * diff_loss

        # Log losses
        self.log("test_total_loss", total_loss)
        self.log("test_mean_loss", mean_loss)
        self.log("test_diff_loss", diff_loss)
        self.log("test_combined_loss", combined_loss)

        # Use compatible format with parent class for correlation calculation
        self.test_step_outputs.append((seq_idx, atac_idx, y_hat_final, gex))

        return combined_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Prediction step for the model."""
        seq_idx, atac_idx, seq, atac, gex, mean_expr = batch

        y_hat_final, y_hat_mean, y_hat_diff = self.forward(seq, atac)
        return y_hat_final


class BaselineMeanCellDrumGEX(MeanCellDrumGEX):
    """A simple baseline model for gene expression prediction.

    This baseline model:
    1. Predicts zero for mean gene expression across all genes
    2. Uses the sum of ATAC signal (log1p transformed) as the cell-specific prediction

    This serves as a simple comparison point for more complex models.
    """

    def __init__(
        self,
        seq_len,
        first_kernel_size=15,
        kernel_size=5,
        num_kernel=256,
        conv_layers=6,
        pooling_size=2,
        lr=1e-3,
        weight_decay=1e-5,
        loss="mse",
        log_input=True,
        mean_loss_weight=0.0,  # Set to zero by default since we're not using mean prediction
        diff_loss_weight=1.0,
        **kwargs,
    ):
        # Call the parent constructor
        super().__init__(
            seq_len,
            first_kernel_size=first_kernel_size,
            kernel_size=kernel_size,
            num_kernel=num_kernel,
            conv_layers=conv_layers,
            pooling_size=pooling_size,
            lr=lr,
            weight_decay=weight_decay,
            loss=loss,
            log_input=log_input,
            mean_loss_weight=mean_loss_weight,
            diff_loss_weight=diff_loss_weight,
            **kwargs,
        )
        # No need to redefine hyperparameters or outputs lists as they're already set in parent

    def forward(self, seq, atac):
        """Forward pass of the baseline model.

        Args:
            seq: DNA sequence data (not used in this baseline)
            atac: ATAC-seq data

        Returns
        -------
            tuple: (final prediction, mean prediction, diff prediction)
        """
        # Mean prediction is always zero
        mean_x = self.DNAEncoder(seq) + self.atac_encoder(atac)
        mean_x = mean_x.flatten(1)

        for fc in self.fc_mean:
            mean_x = fc(mean_x)
        mean_pred = mean_x.squeeze(1)

        # Cell-specific prediction is sum of ATAC signal (log1p transformed)
        # Sum across all dimensions except batch dimension (dim 0)
        diff_pred = torch.log1p(torch.sum(atac, dim=(1, 2)))

        # Final prediction is the sum of diff prediction and mean prediction
        final_pred = diff_pred + mean_pred  # Updated to include mean prediction

        return final_pred, mean_pred, diff_pred


class CrossAttentionDrumGEX(GatedDrumGEX):  # Inherits from GatedDrumGEX
    """Cross-attention DRUM model for gene expression prediction."""

    def __init__(
        self,
        seq_len,
        # SeqEncoder params (can be same as parent or specialized)
        first_kernel_size=15,
        kernel_size=5,
        num_kernel=256,  # This will be embed_dim for attention
        conv_layers=6,
        pooling_size=2,
        # Attention specific params
        num_attention_heads=8,
        num_attention_layers=2,
        attention_dropout=0.1,
        # MLP head params
        h_dim=128,
        h_layers=1,  # Renamed from parent's h_layers to avoid confusion if parent used it differently
        mlp_dropout=0.2,  # Renamed from parent's dropout
        # Common training params (passed to super)
        lr=1e-3,
        weight_decay=1e-5,
        loss="mse",
        log_input=True,
        **kwargs,  # For any other SeqEncoder specific params
    ):
        # Call parent's __init__
        super().__init__(
            seq_len=seq_len,
            first_kernel_size=first_kernel_size,
            kernel_size=kernel_size,
            num_kernel=num_kernel,
            conv_layers=conv_layers,
            pooling_size=pooling_size,
            lr=lr,
            weight_decay=weight_decay,
            loss=loss,
            log_input=log_input,
            **kwargs,
        )

        # Save hyperparameters
        self.save_hyperparameters()

        # Define encoders (overriding parent's)
        self.DNAEncoder = SeqEncoder(
            4, num_kernel, first_kernel_size, kernel_size, pooling_size=pooling_size, layers=conv_layers, **kwargs
        )
        self.atac_encoder = SeqEncoder(
            1, num_kernel, first_kernel_size, kernel_size, pooling_size=pooling_size, layers=conv_layers, **kwargs
        )

        # Calculate encoded sequence length after all the convolution and pooling layers
        _encoded_seq_len = seq_len
        for _ in range(conv_layers):
            _encoded_seq_len = _encoded_seq_len // pooling_size
        self.encoded_seq_len = _encoded_seq_len

        # Cross-attention layers
        self.atac_q_dna_kv_attention = nn.ModuleList(
            [
                CrossAttentionLayer(num_kernel, num_attention_heads, attention_dropout)
                for _ in range(num_attention_layers)
            ]
        )

        # FC Output head specific to this model
        fc_in_dim = num_kernel
        fc_layers_list = []
        for _ in range(h_layers):
            fc_layers_list.append(nn.Sequential(nn.Linear(fc_in_dim, h_dim), nn.ReLU(), nn.Dropout(mlp_dropout)))
            fc_in_dim = h_dim
        fc_layers_list.append(nn.Linear(fc_in_dim, 1))
        self.fc_output = nn.Sequential(*fc_layers_list)

    def forward(self, seq, atac):
        """Forward pass of the cross-attention model."""
        x_seq = self.DNAEncoder(seq)
        x_atac = self.atac_encoder(atac)

        # Change from shape (batch, channels, seq_len) to (batch, seq_len, channels)
        # to match the expected input shape for attention layers
        x_seq = x_seq.permute(0, 2, 1)
        x_atac = x_atac.permute(0, 2, 1)

        # Apply cross-attention: ATAC queries DNA
        contextualized_atac = x_atac
        for layer in self.atac_q_dna_kv_attention:
            contextualized_atac = layer(contextualized_atac, x_seq)

        # Global pooling over the sequence dimension
        x = contextualized_atac.mean(dim=1)

        # Pass through the output layers
        x = self.fc_output(x)
        x = x.squeeze(1)
        return x

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Prediction step for the model."""
        if isinstance(batch, list) or isinstance(batch, tuple):
            seq_idx, atac_idx, seq, atac, gex = batch
        elif isinstance(batch, dict):
            seq = batch["seq"]
            atac = batch["atac"]
        else:
            raise ValueError("Unsupported batch format")

        return self.forward(seq, atac)


class DualCrossAttentionDrumGEX(GatedDrumGEX):
    """Dual cross-attention DRUM model for gene expression prediction."""

    def __init__(
        self,
        seq_len,
        # SeqEncoder params
        first_kernel_size=15,
        kernel_size=5,
        num_kernel=256,  # Embed_dim for attention
        conv_layers=6,
        pooling_size=2,
        # Cross-Attention params (shared for both directions, or specify separately)
        num_attention_heads=8,
        num_attention_layers=2,
        attention_dropout=0.1,
        # MLP head params
        h_dim=128,
        h_layers=1,
        mlp_dropout=0.2,
        # Common training params
        lr=1e-3,
        weight_decay=1e-5,
        loss="mse",
        log_input=True,
        **kwargs,  # For other SeqEncoder or GatedDrumGEX params
    ):
        super().__init__(
            seq_len=seq_len,
            first_kernel_size=first_kernel_size,
            kernel_size=kernel_size,
            num_kernel=num_kernel,
            conv_layers=conv_layers,
            pooling_size=pooling_size,
            lr=lr,
            weight_decay=weight_decay,
            loss=loss,
            log_input=log_input,
            **kwargs,
        )
        self.save_hyperparameters()

        # Encoders (overwriting from GatedDrumGEX if necessary)
        self.DNAEncoder = SeqEncoder(
            4, num_kernel, first_kernel_size, kernel_size, pooling_size=pooling_size, layers=conv_layers, **kwargs
        )
        self.atac_encoder = SeqEncoder(
            1, num_kernel, first_kernel_size, kernel_size, pooling_size=pooling_size, layers=conv_layers, **kwargs
        )

        # Pathway 1: DNA queries ATAC context
        self.dna_query_atac_kv_attention = nn.ModuleList(
            [
                CrossAttentionLayer(
                    self.hparams.num_kernel, self.hparams.num_attention_heads, self.hparams.attention_dropout
                )
                for _ in range(self.hparams.num_attention_layers)
            ]
        )

        # Pathway 2: ATAC queries DNA context
        self.atac_query_dna_kv_attention = nn.ModuleList(
            [
                CrossAttentionLayer(
                    self.hparams.num_kernel, self.hparams.num_attention_heads, self.hparams.attention_dropout
                )
                for _ in range(self.hparams.num_attention_layers)
            ]
        )

        # FC Output head
        # Input to MLP will be concatenation of the two pathways' outputs after pooling
        fc_in_dim = self.hparams.num_kernel * 2
        fc_layers_list = []
        current_dim = fc_in_dim
        for _ in range(self.hparams.h_layers):
            fc_layers_list.append(
                nn.Sequential(
                    nn.Linear(current_dim, self.hparams.h_dim), nn.ReLU(), nn.Dropout(self.hparams.mlp_dropout)
                )
            )
            current_dim = self.hparams.h_dim
        fc_layers_list.append(nn.Linear(current_dim, 1))
        self.fc_output = nn.Sequential(*fc_layers_list)

    def forward(self, seq, atac):
        """Forward pass of the dual cross-attention model."""
        # 1. Initial Convolutional Encoding
        x_seq_encoded = self.DNAEncoder(seq)
        x_atac_encoded = self.atac_encoder(atac)

        # 2. Permute for Attention: (batch, features, length) -> (batch, length, features)
        x_seq = x_seq_encoded.permute(0, 2, 1)
        x_atac = x_atac_encoded.permute(0, 2, 1)

        # 3. Pathway 1: DNA queries ATAC context
        seq_contextualized_by_atac = x_seq  # Initial query for this pathway
        for layer in self.dna_query_atac_kv_attention:
            seq_contextualized_by_atac = layer(seq_contextualized_by_atac, x_atac)

        # 4. Pathway 2: ATAC queries DNA context
        atac_contextualized_by_dna = x_atac  # Initial query for this pathway
        for layer in self.atac_query_dna_kv_attention:
            atac_contextualized_by_dna = layer(atac_contextualized_by_dna, x_seq)

        # 5. Pooling
        # Pool over the sequence dimension
        pooled_seq_context = seq_contextualized_by_atac.mean(dim=1)  # (batch, num_kernel)
        pooled_atac_context = atac_contextualized_by_dna.mean(dim=1)  # (batch, num_kernel)

        # 6. Concatenate pooled representations
        combined_representation = torch.cat((pooled_seq_context, pooled_atac_context), dim=1)
        # Shape: (batch, num_kernel * 2)

        # 7. MLP Head
        output = self.fc_output(combined_representation)  # Shape: (batch, 1)
        output = output.squeeze(1)  # Shape: (batch,)
        return output


class FiveChannelDrumGEX(GatedDrumGEX):
    """A simple baseline DRUM model for gene expression prediction.

    This model takes 5 input channels (4 from DNA sequence, 1 from ATAC) and processes them
    through a single encoder, rather than using separate encoders for DNA and ATAC.
    """

    def __init__(
        self,
        seq_len,
        first_kernel_size=15,
        kernel_size=5,
        num_kernel=256,
        conv_layers=6,
        pooling_size=2,
        lr=1e-3,
        weight_decay=1e-5,
        loss="mse",
        log_input=True,
        h_dim=128,
        h_layers=1,
        dropout=0.2,
        **kwargs,
    ):
        # Initialize the parent class
        super().__init__(
            seq_len=seq_len,
            first_kernel_size=first_kernel_size,
            kernel_size=kernel_size,
            num_kernel=num_kernel,
            conv_layers=conv_layers,
            pooling_size=pooling_size,
            lr=lr,
            weight_decay=weight_decay,
            loss=loss,
            log_input=log_input,
            **kwargs,
        )

        # Save hyperparameters
        self.save_hyperparameters()

        # Replace the separate DNA and ATAC encoders with a single 5-channel encoder
        self.combined_encoder = SeqEncoder(
            5,
            num_kernel,
            first_kernel_size,
            kernel_size,
            pooling_size=pooling_size,
            layers=conv_layers,
            **kwargs,
        )

        # Recalculate output dimension
        self.outdim = num_kernel * (seq_len // pooling_size**conv_layers)

        # Replace the output layer with a more complex MLP if needed
        if h_layers > 0:
            layers = []
            current_dim = self.outdim
            for _ in range(h_layers):
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                current_dim = h_dim
            layers.append(nn.Linear(current_dim, 1))
            self.fc_output = nn.Sequential(*layers)
        else:
            self.fc_output = nn.Sequential(nn.Linear(self.outdim, 1))

    def forward(self, seq, atac):
        """Forward pass of the model.

        Args:
            seq: DNA sequence tensor of shape (batch_size, 4, seq_len)
            atac: ATAC tensor of shape (batch_size, 1, seq_len)

        Returns
        -------
            Predicted gene expression values
        """
        # Concatenate DNA and ATAC along the channel dimension
        x = torch.cat([seq, atac], dim=1)  # (batch_size, 5, seq_len)

        # Pass through the combined encoder
        x = self.combined_encoder(x)

        # Flatten and pass through MLP
        x = x.flatten(1)
        x = self.fc_output(x)

        # Squeeze output to match expected shape
        x = x.squeeze(1)

        return x
