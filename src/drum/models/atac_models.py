# filepath: /homes/gws/tuxm/Project/drum-dev/src/drum/models/atac_models.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

from .base import BaseDrumModel
from .encoders import GeneExpressionEncoder, SeqEncoder


class GatedDrumATAC(BaseDrumModel):
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
        super().__init__(lr=lr, weight_decay=weight_decay, loss=loss, log_input=log_input)
        self.save_hyperparameters()

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

    def forward(self, seq, gex):
        """Forward pass of the model."""
        x_seq = self.DNAEncoder(seq)
        x_seq = x_seq.flatten(1)  # Flatten the output from the DNA encoder

        x_gex = self.GeneExpressionEncoder(gex)  # Shape: (batch, self.outdim)

        x = torch.cat((x_seq, x_gex), dim=1)  # Concatenate along the feature dimension

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
                return self.loss_fn(y_hat, torch.log1p(y.float()))
            else:
                return self.loss_fn(y_hat, y.float())
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

    # --- Methods for Metric Calculation ---

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
                seq_id = seq_idx_batch_np[idx]
                atac_id = atac_idx_batch_np[idx]
                pred = y_hat_batch_np[idx]
                actual = atac_batch_np[idx]

                # Store unique (seq_id, atac_id) pairs
                unique_pairs[(seq_id, atac_id)] = True
                all_actuals_list.append(actual)
                all_preds_list.append(pred)

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
                corr = float("nan")
            else:
                corr = np.corrcoef(actuals_masked, preds_masked)[0, 1]

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
                plt.title(f"Histogram of Pearson Correlations across {phase} sequences")
                plt.xlabel("Correlation")
                plt.ylabel("Frequency")
                seq_corr_plot = wandb.Image(plt)
                wandb_logger.log({f"{phase}_seq_corr_histogram": seq_corr_plot})
                plt.close()
            except (ValueError, RuntimeError, TypeError) as e:
                print(f"Error creating sequence correlation histogram: {e}")

            # ATAC correlations histogram
            try:
                plt.figure()
                plt.hist(atac_corrs_df["Correlation"].dropna(), bins=50, color="green", alpha=0.7)
                plt.title(f"Histogram of Pearson Correlations across {phase} ATAC peaks")
                plt.xlabel("Correlation")
                plt.ylabel("Frequency")
                atac_corr_plot = wandb.Image(plt)
                wandb_logger.log({f"{phase}_atac_corr_histogram": atac_corr_plot})
                plt.close()
            except (ValueError, RuntimeError, TypeError) as e:
                print(f"Error creating ATAC correlation histogram: {e}")
        elif self.trainer.is_global_zero:
            # Only print warning on global rank 0 to avoid spam
            print(f"WandB logger not available or not initialized for {phase} phase. Skipping table/histogram logging.")
