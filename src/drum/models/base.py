# filepath: /homes/gws/tuxm/Project/drum-dev/src/drum/models/base.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb


class BaseDrumModel(pl.LightningModule):
    """Base class for DRUM models."""

    def __init__(self, lr=1e-3, weight_decay=1e-5, loss="mse", log_input=True, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        if loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss == "PoissonNLL":
            self.loss_fn = nn.PoissonNLLLoss(log_input=False)
        else:
            raise ValueError(f"Unsupported loss function: {loss}")

    def compute_loss(self, y_hat, y):
        """Compute the loss for the model."""
        loss_type = self.hparams.loss
        log_input = self.hparams.log_input

        # Handle the Poisson Negative Log-Likelihood loss
        if loss_type == "PoissonNLL":
            if log_input:
                y_hat = torch.exp(y_hat)
            # Compute the PoissonNLL loss
            return self.loss_fn(y_hat, y)

        # Handle Mean Squared Error (MSE) loss
        elif loss_type == "mse":
            if not log_input:
                y_hat = torch.exp(y_hat) - 1  # Reverse the log1p transform
            # Compute the MSE loss
            return self.loss_fn(y_hat, y)

        # For other loss types, compute directly
        return self.loss_fn(y_hat, y)

    def configure_optimizers(self):
        """Configures the optimizer for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

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

        for seq_idx, cell_idx, y_hat, gex in step_outputs:
            for i in range(len(seq_idx)):
                all_seq_ids.append(seq_idx[i].item())
                all_cell_states.append(cell_idx[i].item())
                all_actuals.append(gex[i].item())
                all_preds.append(y_hat[i].item())

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
                corr = float("nan")
            else:
                corr = np.corrcoef(actuals[mask], preds[mask])[0, 1]
            correlations.append((uid, corr))

        return correlations

    def _log_metrics(self, seq_corrs_df, cell_state_corrs_df, phase):
        """Log the computed metrics."""
        median_seq_corr = seq_corrs_df["Correlation"].median()
        median_cell_state_corr = cell_state_corrs_df["Correlation"].median()

        self.log(f"{phase}_median_seq_correlation", median_seq_corr)
        self.log(f"{phase}_median_cell_state_correlation", median_cell_state_corr)

        if hasattr(self.logger, "experiment") and isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
            wandb_logger = self.logger.experiment

            # Log correlation tables
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
