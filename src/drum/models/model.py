import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb

from .encoders import GeneExpressionEncoder, SeqEncoder


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

        self.fc_output = nn.Sequential(nn.Linear(self.outdim, 1))

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

    def _aggregate_and_log_metrics(self, step_outputs, phase):
        """Aggregate data from step outputs and log metrics."""
        all_seq_ids, all_cell_states, all_actuals, all_preds = self._aggregate_step_outputs(step_outputs)
        seq_corrs_df, cell_state_corrs_df = self._calculate_correlations(
            all_seq_ids, all_cell_states, all_actuals, all_preds
        )
        self._log_metrics(seq_corrs_df, cell_state_corrs_df, phase)

    def _aggregate_step_outputs(self, step_outputs):
        """Aggregate data from the step outputs."""
        all_seq_ids = []
        all_cell_states = []
        all_actuals = []
        all_preds = []

        for seq_idx, atac_idx, y_hat, gex in step_outputs:
            for i in range(len(seq_idx)):
                all_actuals.append(gex[i].item())
                all_preds.append(y_hat[i].item())
                all_seq_ids.append(seq_idx[i].item())
                all_cell_states.append(atac_idx[i].item())

        if not self.hparams.log_input:
            all_actuals = np.log1p(np.array(all_actuals))
            all_preds = np.log1p(np.array(all_preds))

        return (
            np.array(all_seq_ids),
            np.array(all_cell_states),
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


class GatedDrumATAC(pl.LightningModule):
    """A gated DRUM model for ATAC signal prediction."""

    def __init__(
        self,
        seq_len,
        gex_dim,  # Dimension of input gex vector (e.g., 25299)
        # SeqEncoder params
        first_kernel_size=15,
        kernel_size=5,
        num_kernel=256,
        conv_layers=6,
        pooling_size=2,
        fc_layers=1,
        fc_hidden_dim=64,
        # GeneExpressionEncoder params
        gex_hidden_dim=None,  # Hidden dimensions for the gene expression encoder
        # Common params
        lr=1e-3,
        weight_decay=1e-5,
        loss="mse",
        log_input=False,  # Added log_input, assuming ATAC values might need log transform for MSE
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Set default gex_hidden_dim if None
        if gex_hidden_dim is None:
            gex_hidden_dim = [64, 32]

        self.DNAEncoder = SeqEncoder(
            4, num_kernel, first_kernel_size, kernel_size, pooling_size=pooling_size, layers=conv_layers, **kwargs
        )
        # Calculate the flattened output dimension from the DNA encoder
        self.outdim = num_kernel * (seq_len // pooling_size**conv_layers)

        # Use the GeneExpressionEncoder class
        self.GeneExpressionEncoder = GeneExpressionEncoder(input_dim=gex_dim, hidden_dims=gex_hidden_dim)

        self.fc_output = nn.ModuleList()
        outdim = self.outdim + gex_hidden_dim[-1]  # Adjusted for concatenation with GEX output
        for _ in range(fc_layers):
            self.fc_output.append(nn.Sequential(nn.Linear(outdim, fc_hidden_dim), nn.ReLU(), nn.Dropout(0.2)))
            outdim = fc_hidden_dim
        self.fc_output.append(nn.Sequential(nn.Linear(outdim, 1)))

        if loss == "mse":
            self.loss_fn = nn.MSELoss()
        # Add other loss options if needed
        else:
            raise ValueError(f"Unsupported loss function: {loss}")

    def forward(self, seq, gex):
        """Forward pass of the model."""
        x_seq = self.DNAEncoder(seq)
        x_seq = x_seq.flatten(1)  # Flatten the output from the DNA encoder

        x_gex = self.GeneExpressionEncoder(gex)  # Shape: (batch, self.outdim)

        x = torch.concatenate((x_seq, x_gex), dim=1)  # Concatenate along the feature dimension

        for fc in self.fc_output:
            x = fc(x)
        x = x.squeeze(1)  # Shape: (batch,)
        return x

    def compute_loss(self, y_hat, y):
        """Compute the loss for the model, handling log transformation if needed."""
        loss_type = self.hparams.loss
        log_input = self.hparams.log_input  # Check if log transform is used

        # Handle Mean Squared Error (MSE) loss
        if loss_type == "mse":
            if log_input:
                # If input is already log-transformed, use it directly
                return self.loss_fn(y_hat, y)
            else:
                # If input is not log-transformed, apply log1p before MSE
                # Ensure y is float for log1p
                y = torch.log1p(y.float())
                y_hat = torch.log1p(y_hat.float())  # Apply to prediction as well
                return self.loss_fn(y_hat, y)
        # Add other loss computations if needed
        else:
            # For other loss types, compute directly (assuming no log transform needed)
            return self.loss_fn(y_hat, y.float())  # Ensure target is float

    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        # Assuming batch format: seq_idx, atac_idx, seq, gex, atac, mean_atac
        _, _, seq, gex, atac, _ = batch  # Unpack, target is atac
        y_hat = self.forward(seq, gex)
        loss = self.compute_loss(y_hat, atac)  # Predict atac signal
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # self.training_step_outputs.append(loss) # Optional: if needed for epoch end
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        """Validation step for the model."""
        seq_idx, atac_idx, seq, gex, atac, _ = batch
        y_hat = self.forward(seq, gex)
        loss = self.compute_loss(y_hat, atac)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # Store outputs for metric calculation at epoch end
        # Ensure tensors are detached and moved to CPU if necessary for aggregation
        self.validation_step_outputs.append((seq_idx.cpu(), atac_idx.cpu(), y_hat.detach().cpu(), atac.detach().cpu()))
        return loss

    def test_step(self, batch, batch_idx):
        """Test step for the model."""
        seq_idx, atac_idx, seq, gex, atac, _ = batch
        y_hat = self.forward(seq, gex)
        loss = self.compute_loss(y_hat, atac)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # Store outputs for metric calculation at epoch end
        self.test_step_outputs.append((seq_idx.cpu(), atac_idx.cpu(), y_hat.detach().cpu(), atac.detach().cpu()))
        return loss

    def configure_optimizers(self):
        """Configures the optimizer for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Prediction step for the model."""
        # Handle cases where the batch might only contain seq and gex for prediction
        if len(batch) == 6:
            _, _, seq, gex, _, _ = batch
        elif len(batch) == 2:  # Assuming (seq, gex) format for prediction
            seq, gex = batch
        else:
            raise ValueError("Unsupported batch format for prediction")

        y_hat = self.forward(seq, gex)
        return y_hat

    # --- Methods for Metric Calculation (adapted from GatedDrumGEX) ---

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

    def _aggregate_and_log_metrics(self, step_outputs, phase):
        """Aggregate data from step outputs and log metrics."""
        if not step_outputs:
            print(f"No outputs to aggregate for {phase} phase.")
            return

        all_seq_ids, all_atac_ids, all_actuals, all_preds = self._aggregate_step_outputs(step_outputs)

        if all_seq_ids.size == 0:
            print(f"Aggregated data is empty for {phase} phase. Skipping metric calculation.")
            return

        seq_corrs_df, atac_corrs_df = self._calculate_correlations(all_seq_ids, all_atac_ids, all_actuals, all_preds)
        self._log_metrics(seq_corrs_df, atac_corrs_df, phase)

    def _aggregate_step_outputs(self, step_outputs):
        """Aggregate data from the step outputs (predicting ATAC)."""
        unique_pairs = {}
        all_actuals_list = []
        all_preds_list = []

        for seq_idx_batch, atac_idx_batch, y_hat_batch, atac_batch in step_outputs:
            # Ensure data is on CPU and converted to numpy
            seq_idx_batch_np = seq_idx_batch.numpy()
            atac_idx_batch_np = atac_idx_batch.numpy()
            y_hat_batch_np = y_hat_batch.numpy()
            atac_batch_np = atac_batch.numpy()

            for idx in range(len(seq_idx_batch_np)):
                pair = (seq_idx_batch_np[idx], atac_idx_batch_np[idx])
                if pair not in unique_pairs:
                    unique_pairs[pair] = True
                    all_actuals_list.append(atac_batch_np[idx])  # Target is ATAC
                    all_preds_list.append(y_hat_batch_np[idx])  # Prediction

        if not unique_pairs:
            return np.array([]), np.array([]), np.array([]), np.array([])

        seq_ids, atac_ids = zip(*unique_pairs.keys())

        all_actuals = np.array(all_actuals_list)
        all_preds = np.array(all_preds_list)

        # Apply log1p if loss is MSE and log_input is False (meaning original data was not logged)
        if self.hparams.loss == "mse" and not self.hparams.log_input:
            all_actuals = np.log1p(np.maximum(all_actuals, 0))
            all_preds = np.log1p(np.maximum(all_preds, 0))
        # If log_input is True for MSE, or for other losses, data is assumed to be in the correct space

        return (
            np.array(seq_ids),
            np.array(atac_ids),  # These are the ATAC peak/region indices or cell state indices
            all_actuals,
            all_preds,
        )

    def _calculate_correlations(self, all_seq_ids, all_atac_ids, all_actuals, all_preds):
        """Calculate Pearson correlations for sequences and ATAC indices/cell states."""
        if all_seq_ids.size == 0:
            return pd.DataFrame(columns=["SeqID", "Correlation"]), pd.DataFrame(columns=["AtacID", "Correlation"])

        seq_correlations = self._calculate_per_correlations(all_seq_ids, all_actuals, all_preds)
        atac_correlations = self._calculate_per_correlations(all_atac_ids, all_actuals, all_preds)

        seq_corrs_df = pd.DataFrame(seq_correlations, columns=["SeqID", "Correlation"])
        # Use "AtacID" or a more descriptive name if atac_idx represents cell state
        atac_corrs_df = pd.DataFrame(atac_correlations, columns=["AtacID", "Correlation"])

        return seq_corrs_df, atac_corrs_df

    def _calculate_per_correlations(self, ids, actuals, preds):
        """Calculate Pearson correlations for unique identifiers."""
        correlations = []
        unique_ids = np.unique(ids)

        for uid in unique_ids:
            mask = ids == uid
            actuals_masked = actuals[mask]
            preds_masked = preds[mask]

            # Ensure there's enough data points and variance to calculate correlation
            # Also check for NaNs/Infs which can occur after log1p
            valid_mask = np.isfinite(actuals_masked) & np.isfinite(preds_masked)
            actuals_masked = actuals_masked[valid_mask]
            preds_masked = preds_masked[valid_mask]

            if len(actuals_masked) < 2 or np.std(actuals_masked) < 1e-6 or np.std(preds_masked) < 1e-6:
                corr = np.nan  # Assign NaN if correlation cannot be computed
            else:
                # Calculate Pearson correlation coefficient
                try:
                    corr = np.corrcoef(actuals_masked, preds_masked)[0, 1]
                    if not np.isfinite(corr):  # Handle potential NaN from corrcoef itself
                        corr = np.nan
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    corr = np.nan  # Handle specific exceptions that might occur during correlation calculation

            correlations.append((uid, corr))

        return correlations

    def _log_metrics(self, seq_corrs_df, atac_corrs_df, phase):
        """Log the computed metrics to WandB, adapted for ATAC prediction."""
        if seq_corrs_df.empty or atac_corrs_df.empty:
            print(f"Correlation dataframes are empty for {phase} phase. Skipping logging.")
            return

        # Drop NaNs before calculating median
        median_seq_corr = seq_corrs_df["Correlation"].dropna().median()
        median_atac_corr = (
            atac_corrs_df["Correlation"].dropna().median()
        )  # Median correlation across ATAC indices/cell states

        # Log median correlations, handle potential NaN median if all correlations were NaN
        self.log(
            f"{phase}_median_seq_correlation", median_seq_corr if pd.notna(median_seq_corr) else 0.0, sync_dist=True
        )
        self.log(
            f"{phase}_median_atac_correlation", median_atac_corr if pd.notna(median_atac_corr) else 0.0, sync_dist=True
        )  # Log ATAC correlation

        # Check if logger is available and is WandB logger
        if (
            self.logger
            and hasattr(self.logger, "experiment")
            and isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run)
        ):
            wandb_logger = self.logger.experiment

            # Log table of correlations
            wandb_logger.log({f"{phase}_seq_correlations": wandb.Table(dataframe=seq_corrs_df.dropna())})
            wandb_logger.log(
                {f"{phase}_atac_correlations": wandb.Table(dataframe=atac_corrs_df.dropna())}
            )  # Log ATAC correlations table

            # Sequence correlations histogram
            try:
                plt.figure()
                plt.hist(seq_corrs_df["Correlation"].dropna(), bins=50, color="blue", alpha=0.7)
                plt.title(f"Histogram of Pearson Correlations across Sequences ({phase})")
                plt.xlabel("Correlation")
                plt.ylabel("Frequency")
                seq_corr_plot = wandb.Image(plt)
                wandb_logger.log({f"{phase}_seq_corr_histogram": seq_corr_plot})
                plt.close()
            except (ValueError, RuntimeError, TypeError) as e:
                print(f"Could not generate/log sequence correlation histogram for {phase}: {e}")
                plt.close()  # Ensure plot is closed even if error occurs

            # ATAC correlations histogram
            try:
                plt.figure()
                plt.hist(atac_corrs_df["Correlation"].dropna(), bins=50, color="green", alpha=0.7)
                # Update title based on what atac_idx represents
                plt.title(f"Histogram of Pearson Correlations across ATAC Indices/Cell States ({phase})")
                plt.xlabel("Correlation")
                plt.ylabel("Frequency")
                atac_corr_plot = wandb.Image(plt)
                wandb_logger.log({f"{phase}_atac_corr_histogram": atac_corr_plot})  # Log ATAC histogram
                plt.close()
            except (ValueError, RuntimeError, TypeError) as e:
                print(f"Could not generate/log ATAC correlation histogram for {phase}: {e}")
                plt.close()
        elif self.trainer.is_global_zero:
            # Only print warning on global rank 0 to avoid spam
            print(f"WandB logger not available or not initialized for {phase} phase. Skipping table/histogram logging.")


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
            first_kernel_size=first_kernel_size,
            kernel_size=kernel_size,
            num_kernel=num_kernel,
            conv_layers=conv_layers,
            pooling_size=pooling_size,
            lr=lr,
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

        # Create sequential layers for sequence and ATAC data processing
        for _ in range(h_layers):
            self.fc_seq.append(nn.Sequential(nn.Linear(outdim, h_dim), nn.ReLU(), nn.Dropout(dropout)))
            self.fc_atac.append(nn.Sequential(nn.Linear(outdim, h_dim), nn.ReLU(), nn.Dropout(dropout)))
            outdim = h_dim

        # Output layers for sequence and ATAC data
        self.fc_seq.append(nn.Sequential(nn.Linear(outdim, 1), nn.ReLU()))
        self.fc_atac.append(nn.Sequential(nn.Linear(outdim, 1), nn.ReLU()))

        # Loss function initialization
        if loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss == "PoissonNLL":
            self.loss_fn = nn.PoissonNLLLoss(log_input=False)  # Ensure prediction is in counts space for Poisson loss

    def forward(self, seq, atac):
        """Forward pass of the model.

        Args:
            seq (torch.Tensor): DNA sequence data
            atac (torch.Tensor): ATAC-seq data

        Returns
        -------
            torch.Tensor: Predicted gene expression values
        """
        # Extract features from sequence and ATAC data
        x_seq = self.DNAEncoder(seq)
        x_seq = x_seq.flatten(1)  # Flatten the output from the DNA encoder

        x_atac = self.atac_encoder(atac)

        x = torch.concatenate((x_seq, x_atac), dim=1)  # Concatenate along the feature dimension

        for fc in self.fc_output:
            x = fc(x)
        x = x.squeeze(1)  # Shape: (batch,)
        return x

    def compute_loss(self, y_hat, y):
        """Compute the loss for the model, handling log transformation if needed."""
        loss_type = self.hparams.loss
        log_input = self.hparams.log_input  # Check if log transform is used

        # Handle Mean Squared Error (MSE) loss
        if loss_type == "mse":
            if log_input:
                # If input is already log-transformed, use it directly
                return self.loss_fn(y_hat, y)
            else:
                # If input is not log-transformed, apply log1p before MSE
                # Ensure y is float for log1p
                y = torch.log1p(y.float())
                y_hat = torch.log1p(y_hat.float())  # Apply to prediction as well
                return self.loss_fn(y_hat, y)
        # Add other loss computations if needed
        else:
            # For other loss types, compute directly (assuming no log transform needed)
            return self.loss_fn(y_hat, y.float())  # Ensure target is float

    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        # Assuming batch format: seq_idx, atac_idx, seq, gex, atac, mean_atac
        _, _, seq, gex, atac, _ = batch  # Unpack, target is atac
        y_hat = self.forward(seq, gex)
        loss = self.compute_loss(y_hat, atac)  # Predict atac signal
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # self.training_step_outputs.append(loss) # Optional: if needed for epoch end
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        """Validation step for the model."""
        seq_idx, atac_idx, seq, gex, atac, _ = batch
        y_hat = self.forward(seq, gex)
        loss = self.compute_loss(y_hat, atac)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # Store outputs for metric calculation at epoch end
        # Ensure tensors are detached and moved to CPU if necessary for aggregation
        self.validation_step_outputs.append((seq_idx.cpu(), atac_idx.cpu(), y_hat.detach().cpu(), atac.detach().cpu()))
        return loss

    def test_step(self, batch, batch_idx):
        """Test step for the model."""
        seq_idx, atac_idx, seq, gex, atac, _ = batch
        y_hat = self.forward(seq, gex)
        loss = self.compute_loss(y_hat, atac)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # Store outputs for metric calculation at epoch end
        self.test_step_outputs.append((seq_idx.cpu(), atac_idx.cpu(), y_hat.detach().cpu(), atac.detach().cpu()))
        return loss

    def configure_optimizers(self):
        """Configures the optimizer for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Prediction step for the model."""
        # Handle cases where the batch might only contain seq and gex for prediction
        if len(batch) == 6:
            _, _, seq, gex, _, _ = batch
        elif len(batch) == 2:  # Assuming (seq, gex) format for prediction
            seq, gex = batch
        else:
            raise ValueError("Unsupported batch format for prediction")

        y_hat = self.forward(seq, gex)
        return y_hat

    # --- Methods for Metric Calculation (adapted from GatedDrumGEX) ---

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

    def _aggregate_and_log_metrics(self, step_outputs, phase):
        """Aggregate data from step outputs and log metrics."""
        if not step_outputs:
            print(f"No outputs to aggregate for {phase} phase.")
            return

        seq_ids_list = []
        y_hat_list = []
        gex_list = []
        for seq_idx, y_hat, gex in step_outputs:
            # Ensure data is on CPU and converted to numpy
            seq_idx = seq_idx.cpu()
            y_hat = y_hat.cpu()
            gex = gex.cpu()

            seq_ids_list.append(seq_idx)
            y_hat_list.append(y_hat)
            gex_list.append(gex)

        all_y_hat = torch.cat(y_hat_list).numpy()
        all_gex = torch.cat(gex_list).numpy()

        # compute correlations for each task
        correlations = []
        for task_idx in range(all_y_hat.shape[1]):
            task_y_hat = all_y_hat[:, task_idx]
            task_gex = all_gex[:, task_idx]

            # Calculate Pearson correlation coefficient
            corr = np.corrcoef(task_y_hat, task_gex)[0, 1]
            if not np.isfinite(corr):
                corr = np.nan
            correlations.append((task_idx, corr))
        correlations_df = pd.DataFrame(correlations, columns=["CellID", "Correlation"])
        correlations_df["Correlation"] = correlations_df["Correlation"].astype(float)

        # Log the median correlation
        median_corr = correlations_df["Correlation"].median()
        self.log(f"{phase}_CellID_median_correlation", median_corr, sync_dist=True)
        # Log the correlations dataframe
        if (
            self.logger
            and hasattr(self.logger, "experiment")
            and isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run)
        ):
            wandb_logger = self.logger.experiment
            # Log the correlations dataframe to WandB
            wandb_logger.log({f"{phase}_correlations": wandb.Table(dataframe=correlations_df)})

            # Histogram of correlations
            plt.figure()
            plt.hist(correlations_df["Correlation"].dropna(), bins=50, color="blue", alpha=0.7)
            plt.title(f"Histogram of Pearson Correlations across {phase} cell states")
            plt.xlabel("Correlation")
            plt.ylabel("Frequency")
            corr_hist_plot = wandb.Image(plt)
            wandb_logger.log({f"{phase}_correlation_histogram": corr_hist_plot})
            plt.close()
        else:
            print(f"WandB logger not available or not initialized for {phase} phase. Skipping table/histogram logging.")
            # Optionally, print the correlations DataFrame
            print(f"{phase} correlations:\n", correlations_df)


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
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

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
        # print(atac.shape)
        diff_pred = torch.log1p(torch.sum(atac, dim=(1, 2)))

        # Final prediction is the sum of diff prediction and mean prediction
        final_pred = diff_pred + mean_pred  # Updated to include mean prediction

        return final_pred, mean_pred, diff_pred


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
        """Forward pass through the cross-attention layer."""
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        query = self.norm1(query + self.dropout(attn_output))
        ffn_output = self.ffn(query)
        query = self.norm2(query + self.dropout(ffn_output))
        return query


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
        log_input="True",
        **kwargs,  # For any other SeqEncoder specific params
    ):
        # Call parent's __init__ to set up shared attributes like output lists,
        # loss_fn, and potentially save common hyperparameters.
        # Pass only the arguments GatedDrumGEX's __init__ expects.
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
        # Now, save hyperparameters specific to CrossAttentionDrumGEX or all of them.
        # If super().__init__ already called self.save_hyperparameters(), this might
        # overwrite or extend. Pytorch Lightning handles this gracefully by typically
        # using the hparams from the most derived class.
        # To be safe and explicit:
        self.save_hyperparameters()

        # === Define/Override Architecture for CrossAttentionDrumGEX ===
        # Encoders (these will overwrite the ones from GatedDrumGEX's __init__)
        self.DNAEncoder = SeqEncoder(
            4, num_kernel, first_kernel_size, kernel_size, pooling_size=pooling_size, layers=conv_layers, **kwargs
        )
        self.atac_encoder = SeqEncoder(
            1, num_kernel, first_kernel_size, kernel_size, pooling_size=pooling_size, layers=conv_layers, **kwargs
        )

        _encoded_seq_len = seq_len
        for _ in range(conv_layers):
            if pooling_size > 1:
                _encoded_seq_len = _encoded_seq_len // pooling_size
        self.encoded_seq_len = _encoded_seq_len

        self.atac_q_dna_kv_attention = nn.ModuleList(
            [
                CrossAttentionLayer(num_kernel, num_attention_heads, attention_dropout)
                for _ in range(num_attention_layers)
            ]
        )

        # FC Output head specific to this model
        fc_in_dim = num_kernel
        fc_layers_list = []
        for _ in range(h_layers):  # Use self.hparams.h_layers for clarity
            fc_layers_list.append(nn.Linear(fc_in_dim, self.hparams.h_dim))
            fc_layers_list.append(nn.ReLU())
            fc_layers_list.append(nn.Dropout(self.hparams.mlp_dropout))
            fc_in_dim = self.hparams.h_dim
        fc_layers_list.append(nn.Linear(fc_in_dim, 1))
        self.fc_output = nn.Sequential(*fc_layers_list)  # Overwrites self.fc_output_gated if parent named it fc_output

        # self.loss_fn is already initialized by super().__init__ and should be compatible.

    def forward(self, seq, atac):
        """Forward pass of the cross-attention model."""
        x_seq = self.DNAEncoder(seq)
        x_atac = self.atac_encoder(atac)

        x_seq = x_seq.permute(0, 2, 1)
        x_atac = x_atac.permute(0, 2, 1)

        contextualized_atac = x_atac
        for layer in self.atac_q_dna_kv_attention:
            contextualized_atac = layer(query=contextualized_atac, key_value=x_seq)

        x = contextualized_atac.mean(dim=1)
        x = self.fc_output(x)  # Use this class's fc_output
        x = x.squeeze(1)
        return x

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Prediction step for the model."""
        if isinstance(batch, list) or isinstance(batch, tuple):
            if len(batch) == 5:  # Common training/val format: seq_idx, atac_idx, seq, atac, gex
                _, _, seq, atac, _ = batch
            elif len(batch) == 4:  # E.g., seq_idx, atac_idx, seq, atac
                _, _, seq, atac = batch
            elif len(batch) == 2:  # E.g., seq, atac
                seq, atac = batch
            else:
                raise ValueError(
                    f"Unsupported batch format for prediction. Batch type: {type(batch)}, Length: {len(batch) if isinstance(batch, (list,tuple)) else 'N/A'}"
                )
        elif isinstance(batch, dict):  # Dictionary batch
            seq = batch.get("seq")
            atac = batch.get("atac")
            if seq is None or atac is None:
                raise ValueError("Batch dictionary missing 'seq' or 'atac' key for prediction.")
        else:
            raise ValueError(f"Unsupported batch type for prediction: {type(batch)}")

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
        # Common params for both SeqEncoders
        # MLP head params
        h_dim=128,
        h_layers=1,
        mlp_dropout=0.2,
        # Common training params
        lr=1e-3,
        weight_decay=1e-5,
        loss="mse",
        log_input="True",
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

        # Encoders (overwriting from GatedDrumGEX if necessary, or using its init)
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
            fc_layers_list.append(nn.Linear(current_dim, self.hparams.h_dim))
            fc_layers_list.append(nn.ReLU())
            fc_layers_list.append(nn.Dropout(self.hparams.mlp_dropout))
            current_dim = self.hparams.h_dim
        fc_layers_list.append(nn.Linear(current_dim, 1))
        self.fc_output = nn.Sequential(*fc_layers_list)

    def forward(self, seq, atac):
        """Forward pass of the dual cross-attention model."""
        # seq: (batch, 4, seq_len_raw)
        # atac: (batch, 1, seq_len_raw)

        # 1. Initial Convolutional Encoding
        # x_seq_encoded: (batch, num_kernel, encoded_seq_len)
        x_seq_encoded = self.DNAEncoder(seq)
        # x_atac_encoded: (batch, num_kernel, encoded_seq_len)
        x_atac_encoded = self.atac_encoder(atac)

        # 2. Permute for Attention: (batch, features, length) -> (batch, length, features)
        x_seq = x_seq_encoded.permute(0, 2, 1)
        x_atac = x_atac_encoded.permute(0, 2, 1)

        # 3. Pathway 1: DNA queries ATAC context
        # query_dna starts as the original encoded DNA sequence
        # key_value_atac is the original encoded ATAC sequence
        seq_contextualized_by_atac = x_seq  # Initial query for this pathway
        for layer in self.dna_query_atac_kv_attention:
            seq_contextualized_by_atac = layer(query=seq_contextualized_by_atac, key_value=x_atac)
            # Output shape: (batch, encoded_seq_len_dna, num_kernel)

        # 4. Pathway 2: ATAC queries DNA context
        # query_atac starts as the original encoded ATAC sequence
        # key_value_dna is the original encoded DNA sequence
        atac_contextualized_by_dna = x_atac  # Initial query for this pathway
        for layer in self.atac_query_dna_kv_attention:
            atac_contextualized_by_dna = layer(query=atac_contextualized_by_dna, key_value=x_seq)
            # Output shape: (batch, encoded_seq_len_atac, num_kernel)

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
        log_input="True",
        h_dim=128,
        h_layers=1,
        dropout=0.2,
        **kwargs,
    ):
        # Initialize the parent class, but we'll override some of its components
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
        # The 5 channels come from DNA (4 channels) and ATAC (1 channel)
        self.combined_encoder = SeqEncoder(
            5,  # 5 input channels (4 DNA + 1 ATAC)
            num_kernel,
            first_kernel_size,
            kernel_size,
            pooling_size=pooling_size,
            layers=conv_layers,
            **kwargs,
        )

        # Recalculate output dimension (same calculation as parent class)
        self.outdim = num_kernel * (seq_len // pooling_size**conv_layers)

        # Optional: Replace the simple linear output with a more complex MLP
        if h_layers > 0:
            fc_layers = []
            in_dim = self.outdim
            for _ in range(h_layers):
                fc_layers.append(nn.Linear(in_dim, h_dim))
                fc_layers.append(nn.ReLU())
                fc_layers.append(nn.Dropout(dropout))
                in_dim = h_dim
            fc_layers.append(nn.Linear(in_dim, 1))
            self.fc_output = nn.Sequential(*fc_layers)
        else:
            # Use simple linear layer if h_layers=0
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
        # seq: (batch_size, 4, seq_len), atac: (batch_size, 1, seq_len)
        x = torch.cat([seq, atac], dim=1)  # (batch_size, 5, seq_len)

        # Pass through the combined encoder
        x = self.combined_encoder(x)  # (batch_size, num_kernel, seq_len/pooling_size^conv_layers)

        # Flatten and pass through MLP
        x = x.flatten(1)  # (batch_size, num_kernel * (seq_len/pooling_size^conv_layers))
        x = self.fc_output(x)  # (batch_size, 1)

        # Squeeze output to match expected shape
        x = x.squeeze(1)  # (batch_size,)

        return x
