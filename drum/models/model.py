import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb

from .encoders import SeqEncoder


class GatedDrumGEX(pl.LightningModule):
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
        log_input="True",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

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

        self.fc_output = nn.Sequential(nn.Linear(self.outdim, 1), nn.ReLU())

        if loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss == "PoissonNLL":
            self.loss_fn = nn.PoissonNLLLoss(
                log_input=False
            )  # make sure the predict is the counts if using poisson loss

        # only support mse log_input=True and poisson loss

    def forward(self, seq, atac):
        """Forward pass of the model."""
        x_seq = self.DNAEncoder(seq)
        x_atac = self.atac_encoder(atac)
        x = x_seq * x_atac

        x = x.flatten(1)
        x = self.fc_output(x)

        x = x.squeeze(1)
        return x

    def compute_loss(self, y_hat, y):
        """Compute the loss for the model."""
        loss_type = self.hparams.loss
        log_input = self.hparams.log_input

        # Handle the Poisson Negative Log-Likelihood loss
        if loss_type == "PoissonNLL":
            if log_input:
                # Convert log-space predictions back to counts for loss computation
                y = y.exp() - 1
            # Compute the PoissonNLL loss
            return self.loss_fn(y_hat, y)

        # Handle Mean Squared Error (MSE) loss
        elif loss_type == "mse":
            if not log_input:
                # Convert both predictions and targets to log-space for MSE
                y = y.log1p()
                y_hat = y_hat.log1p()
            # Compute the MSE loss
            return self.loss_fn(y_hat, y)

        # For other loss types, compute directly
        return self.loss_fn(y_hat, y)

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

    def configure_optimizers(self):
        """Configures the optimizer for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

    def on_validation_epoch_end(self):
        """Calculate and log metrics at the end of each validation epoch."""
        self._aggregate_and_log_metrics(self.validation_step_outputs, phase="validation")
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        """Calculate and log metrics at the end of each test epoch."""
        self._aggregate_and_log_metrics(self.test_step_outputs, phase="test")
        self.test_step_outputs.clear()

    def _aggregate_and_log_metrics(self, step_outputs, phase):
        """Aggregate data from step outputs and log metrics."""
        all_seq_ids, all_cell_states, all_actuals, all_preds = self._aggregate_step_outputs(step_outputs)
        seq_corrs_df, cell_state_corrs_df = self._calculate_correlations(
            all_seq_ids, all_cell_states, all_actuals, all_preds
        )
        self._log_metrics(seq_corrs_df, cell_state_corrs_df, phase)

    def _aggregate_step_outputs(self, step_outputs):
        """Aggregate data from the step outputs."""
        unique_pairs = {}
        all_actuals = []
        all_preds = []

        for seq_idx, atac_idx, y_hat, gex in step_outputs:
            for i in range(len(seq_idx)):
                pair = (seq_idx[i].item(), atac_idx[i].item())
                if pair not in unique_pairs:
                    unique_pairs[pair] = True
                    all_actuals.append(gex[i].item())
                    all_preds.append(y_hat[i].item())

        seq_ids, cell_states = zip(*unique_pairs.keys())

        if not self.hparams.log_input:
            all_actuals = np.log1p(np.array(all_actuals))
            all_preds = np.log1p(np.array(all_preds))

        return (
            np.array(seq_ids),
            np.array(cell_states),
            np.array(all_actuals),
            np.array(all_preds),
        )

    def _calculate_correlations(self, all_seq_ids, all_cell_states, all_actuals, all_preds):
        """Calculate Pearson correlations for sequences and cell states."""
        seq_correlations = self._calculate_per_correlations(all_seq_ids, all_actuals, all_preds)
        cell_state_correlations = self._calculate_per_correlations(all_cell_states, all_actuals, all_preds)

        seq_corrs_df = pd.DataFrame(seq_correlations, columns=["SeqID", "Correlation"])
        cell_state_corrs_df = pd.DataFrame(cell_state_correlations, columns=["CellStateID", "Correlation"])

        return seq_corrs_df, cell_state_corrs_df

    def _calculate_per_correlations(self, ids, actuals, preds):
        """Calculate Pearson correlations for unique identifiers."""
        correlations = []
        unique_ids = np.unique(ids)

        for uid in unique_ids:
            mask = ids == uid
            if np.std(actuals[mask]) == 0 or np.std(preds[mask]) == 0:
                correlations.append((uid, np.nan))
                continue

            corr = np.corrcoef(actuals[mask], preds[mask])[0, 1]
            correlations.append((uid, corr))

        return correlations

    def _log_metrics(self, seq_corrs_df, cell_state_corrs_df, phase):
        """Log the computed metrics to WandB. including image of histogram of correlations."""
        median_seq_corr = seq_corrs_df["Correlation"].median()
        median_cell_state_corr = cell_state_corrs_df["Correlation"].median()

        self.log(f"{phase}_median_seq_correlation", median_seq_corr)
        self.log(f"{phase}_median_cell_state_correlation", median_cell_state_corr)

        wandb_logger = self.logger.experiment

        # log table of correlations
        wandb_logger.log({f"{phase}_seq_correlations": wandb.Table(dataframe=seq_corrs_df)})
        wandb_logger.log({f"{phase}_cell_state_correlations": wandb.Table(dataframe=cell_state_corrs_df)})

        # Sequence correlations histogram
        plt.figure()
        plt.hist(seq_corrs_df["Correlation"].dropna(), bins=50, color="blue", alpha=0.7)
        plt.title(f"Histogram of Pearson Correlations across {phase} cell states")
        plt.xlabel("Correlation")
        plt.ylabel("Frequency")
        seq_corr_plot = wandb.Image(plt)
        wandb_logger.log({f"{phase}_seq_corr_histogram": seq_corr_plot})
        plt.close()


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
            **kwargs,
        )
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

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

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Prediction step for the model."""
        seq_idx, atac_idx, seq, atac, gex = batch

        y_hat = self.forward(seq, atac)
        return y_hat

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
            first_kernel_size=15,
            kernel_size=5,
            num_kernel=256,
            conv_layers=6,
            pooling_size=2,
            lr=1e-3,
            weight_decay=weight_decay,
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
            self.fc_atac.append(nn.Sequential(nn.Linear(outdim, h_dim), nn.ReLU(), nn.Dropout(dropout)))
            outdim = h_dim

        self.fc_seq.append(nn.Sequential(nn.Linear(outdim, 1), nn.ReLU()))
        self.fc_atac.append(nn.Sequential(nn.Linear(outdim, 1), nn.ReLU()))
        if loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss == "PoissonNLL":
            self.loss_fn = nn.PoissonNLLLoss(
                log_input=False
            )  # make sure the predict is the counts if using poisson loss

    def forward(self, seq, atac):
        """Forward pass of the model."""
        x_seq = self.DNAEncoder(seq)
        x_atac = self.atac_encoder(atac)
        x_seq_atac = x_seq + x_seq * x_atac

        x_seq = x_seq.flatten(1)
        for fc in self.fc_seq:
            x_seq = fc(x_seq)

        x_seq_atac = x_seq_atac.flatten(1)
        for fc in self.fc_atac:
            x_seq_atac = fc(x_seq_atac)

        x = x_seq.squeeze(1) + x_seq_atac.squeeze(1)
        return x

class DrumMultiTaskGEX(pl.LightningModule):
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
        super().__init__()
        self.save_hyperparameters()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

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
                y = y.exp() - 1
            return self.loss_fn(y_hat, y)
        elif loss_type == "mse":
            if not log_input:
                y = y.log1p()
                y_hat = y_hat.log1p()
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

    def configure_optimizers(self):
        """Configures the optimizer for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

    def on_test_epoch_end(self):
        """Calculate and log metrics at the end of each test epoch."""
        self._aggregate_and_log_metrics(self.test_step_outputs, phase="test")
        self.test_step_outputs.clear()
