"""LSTM model for continuous glucose monitor (CGM) forecasting."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

INPUT_SIZE = 7
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.2
OUTPUT_STEPS = 24
LEARNING_RATE = 0.001

class CgmLstm(nn.Module):
    """Three layer LSTM that predicts the next 24 CGM readings.

    Architecture:
        - LSTM: input_size=7, hidden_size=128, num_layers=3, dropout=0.2
        - Fully connected output layer: 128 → 24

    Loss:
        Composite of MSE on predicted values, slopes, and curvatures.
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
        self.loss_fn = nn.MSELoss()

    def combo_loss(self, pred: torch.Tensor, target: torch.Tensor,
                   value_w: float = 1.0, slope_w: float = 0.15, curve_w: float = 0.05) -> torch.Tensor:
        """Composite loss combining value MSE, slope MSE, and curvature MSE.

        Args:
            pred: Predicted CGM sequence, shape (batch, OUTPUT_STEPS).
            target: Ground truth CGM sequence, shape (batch, OUTPUT_STEPS).
            value_w: Weighting of value loss.
            slope_w: Weighting of slope loss.
            curve_w: Weighting of curvature loss.

        Returns:
            Scalar loss tensor.
        """
        weights = torch.tensor([value_w, slope_w, curve_w], dtype=pred.dtype, device=pred.device)
        weights = weights / weights.sum()

        eps = 1e-8

        value_loss = self.loss_fn(pred, target) / (target.var() + eps)

        delta_pred = pred[:, 1:] - pred[:, :-1]
        delta_true = target[:, 1:] - target[:, :-1]
        slope_loss = self.loss_fn(delta_pred, delta_true) / (delta_true.var() + eps)

        curve_pred = delta_pred[:, 1:] - delta_pred[:, :-1]
        curve_true = delta_true[:, 1:] - delta_true[:, :-1]
        curve_loss = self.loss_fn(curve_pred, curve_true) / (curve_true.var() + eps)


        return weights[0] * value_loss + weights[1] * slope_loss + weights[2] * curve_loss

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

    def evaluate_model(self, test_loader: DataLoader) -> None:
        """Evaluate the model on a test set.

        Args:
            test_loader: DataLoader yielding (inputs, targets) batches.
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

        avg_loss = total_loss / len(test_loader)
        print(f"Validation Loss: {avg_loss:.4f}")