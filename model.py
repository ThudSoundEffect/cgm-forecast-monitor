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
COMBO_LOSS_WEIGHT = 0.5


class CgmLstm(nn.Module):
    """Three-layer LSTM that predicts the next 24 CGM readings.

    Architecture:
        - LSTM: input_size=7, hidden_size=128, num_layers=3, dropout=0.2
        - Fully connected output layer: 128 → 24

    Loss:
        Composite of MSE on predicted values and MSE on successive deltas,
        weighted equally (0.5 / 0.5).
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

    def combo_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Composite loss combining value MSE and shape (delta) MSE.

        Args:
            pred: Predicted CGM sequence, shape (batch, OUTPUT_STEPS).
            target: Ground-truth CGM sequence, shape (batch, OUTPUT_STEPS).

        Returns:
            Scalar loss tensor.
        """
        value_loss = self.loss_fn(pred, target)
        delta_pred = pred[:, 1:] - pred[:, :-1]
        delta_true = target[:, 1:] - target[:, :-1]
        shape_loss = self.loss_fn(delta_pred, delta_true)
        return COMBO_LOSS_WEIGHT * value_loss + COMBO_LOSS_WEIGHT * shape_loss

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
        """Evaluate the model on a held-out test set.

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