"""LSTM model for continuous glucose monitor (CGM) forecasting."""
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

INPUT_SIZE = 7
HIDDEN_SIZE = 256
NUM_LAYERS = 3
DROPOUT = 0.2
OUTPUT_STEPS = 24
LEARNING_RATE = 0.001

def random_sweep(
        train_loader: DataLoader,
        n_trials: int = 20,
        epochs: int = 4,
) -> dict:
    """Randomized search over combo_loss weights for an untrained model.

    Each trial trains a fresh model from random initialization for
    ``epochs`` epochs using a random (value_w, slope_w, curve_w)
    combination. The final epoch's average loss is used to score
    the combination. The combination resulting in the lowest score is returned.

    Weight search ranges:
        - value_w : Uniform[0.5, 2.0]
        - slope_w : Uniform[0.5, 2.0]
        - curve_w : Uniform[0.0, 1.0]

    Args:
        train_loader: DataLoader with (inputs, targets) training batches.
        n_trials: Number of random weight combinations to try.
        epochs: Training epochs per trial.

    Returns:
        Dict with keys:
            ``"loss"``    – best final epoch training loss achieved (float).
            ``"weights"`` – (value_w, slope_w, curve_w) tuple from the best trial.
    """
    best: dict = {"loss": float("inf"), "weights": None}

    for trial in range(1, n_trials + 1):
        vw = random.uniform(0.5, 2.0)
        sw = random.uniform(0.5, 2.0)
        cw = random.uniform(0.0, 1.0)

        model = CgmLstm().to("cuda")
        avg_loss = 0.0
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to("cuda"), targets.to("cuda")
                model.optimizer.zero_grad()
                loss = model.combo_loss(model(inputs), targets, vw, sw, cw)
                loss.backward()
                model.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        is_best = avg_loss < best["loss"]
        print(
            f"Trial {trial:>3}/{n_trials} | "
            f"value_w={vw:.3f}  slope_w={sw:.3f}  curve_w={cw:.3f} | "
            f"loss={avg_loss:.6f}"
            + (" (best)" if is_best else "")
        )

        if is_best:
            best = {"loss": avg_loss, "weights": (vw, sw, cw)}
    return best


class CgmLstm(nn.Module):
    """Three layer LSTM that predicts the next 24 CGM readings.

    Architecture:
        - LSTM: input_size=7, hidden_size=256, num_layers=3, dropout=0.2
        - Fully connected output layer: 128 to 24

    Loss:
        Composite of loss on predicted values, slopes, and curvatures. Utilizes
        exponential moving average and variance for values, slopes, and curvatures
        to make loss values of different weightings comparable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
        )


        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_STEPS)
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.SmoothL1Loss()

        self._loss_decay = 0.99
        self._ema_initialized = False
        self.register_buffer("_value_mean",  torch.tensor(0))
        self.register_buffer("_value_var",   torch.tensor(1e-4))
        self.register_buffer("_slope_mean",  torch.tensor(0))
        self.register_buffer("_slope_var",   torch.tensor(1e-4))
        self.register_buffer("_curve_mean",  torch.tensor(0))
        self.register_buffer("_curve_var",   torch.tensor(1e-4))

    def _update_ema(self, mean_buf, var_buf, x: torch.Tensor):
        """Update EMA mean and variance buffers in-place."""
        d = self._loss_decay
        new_mean = d * mean_buf + (1 - d) * x
        new_var  = d * var_buf  + (1 - d) * (x - mean_buf).pow(2)
        mean_buf.copy_(new_mean)
        var_buf.copy_(new_var)

    def combo_loss(self, pred, target,
                   value_w=1.839, slope_w=1.275, curve_w=0.088,) -> torch.Tensor:
        """Composite loss combining value, slope, and curvature loss. Updates
        exponential moving averages and variances for value, slope, and curvature.
        Normalizes weights of value, slope, and curvature loss.

        Args:
            pred: Predicted CGM sequence, shape (batch, OUTPUT_STEPS).
            target: Ground truth CGM sequence, shape (batch, OUTPUT_STEPS).
            value_w: Weighting of value loss.
            slope_w: Weighting of slope loss.
            curve_w: Weighting of curvature loss.

        Returns:
            Scalar loss tensor.
        """
        value_loss = self.loss_fn(pred, target)

        delta_pred = pred[:, 1:]   - pred[:, :-1]
        delta_true = target[:, 1:] - target[:, :-1]
        slope_loss = self.loss_fn(delta_pred, delta_true)

        curve_pred = delta_pred[:, 1:] - delta_pred[:, :-1]
        curve_true = delta_true[:, 1:] - delta_true[:, :-1]
        curve_loss = self.loss_fn(curve_pred, curve_true)

        if not self._ema_initialized:
            self._value_mean.copy_(value_loss.detach())
            self._value_var.copy_(value_loss.detach().pow(2))
            self._slope_mean.copy_(slope_loss.detach())
            self._slope_var.copy_(slope_loss.detach().pow(2))
            self._curve_mean.copy_(curve_loss.detach())
            self._curve_var.copy_(curve_loss.detach().pow(2))
            self._ema_initialized = True
        else:
            self._update_ema(self._value_mean, self._value_var, value_loss.detach())
            self._update_ema(self._slope_mean, self._slope_var, slope_loss.detach())
            self._update_ema(self._curve_mean, self._curve_var, curve_loss.detach())

        eps = 1e-8
        value_norm = value_loss / (self._value_var.sqrt() + eps)
        slope_norm = slope_loss / (self._slope_var.sqrt() + eps)
        curve_norm = curve_loss / (self._curve_var.sqrt() + eps)

        weight = value_w + slope_w + curve_w
        return (value_w * value_norm + slope_w * slope_norm + curve_w * curve_norm) / weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the LSTM and output layer.

        Args:
            x: Input tensor of shape (batch, seq_len, INPUT_SIZE).

        Returns:
            Predicted CGM values of shape (batch, OUTPUT_STEPS).
        """
        _output, (hidden, _cell) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.fc(last_hidden)

    def train_model(self, epochs: int, train_loader: DataLoader) -> None:
        """Train the model for a given number of epochs.

        Args:
            epochs: Number of full passes over the training data.
            train_loader: DataLoader yielding (inputs, targets) batches.
        """
        self.to("cuda")
        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to("cuda")
                targets = targets.to("cuda")

                self.optimizer.zero_grad()
                predictions = self(inputs)
                loss = self.combo_loss(predictions, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def evaluate_model(self, test_loader: DataLoader) -> float:
        """Evaluate the model on a test set.

        Args:
            test_loader: DataLoader yielding (inputs, targets) batches.

        Returns:
            Average loss over the validation set.
        """
        self.eval()
        self.to("cuda")
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to("cuda")
                targets = targets.to("cuda")

                predictions = self(inputs)
                loss = self.combo_loss(predictions, targets)
                total_loss += loss.item()

        return total_loss / len(test_loader)