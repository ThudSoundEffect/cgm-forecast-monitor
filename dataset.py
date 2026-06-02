"""PyTorch Dataset for time-series CGM sequences."""

import torch
from torch.utils.data import Dataset
import pandas as pd


class SeriesSet(Dataset):
    """Sliding-window dataset that yields input sequences and future CGM targets.

    Each sample consists of a fixed-length history window and a 2-hour (24-step)
    prediction horizon of CGM readings.

    Args:
        data: Pre-processed DataFrame with feature columns.
        seq_len: Number of time steps in the input window.
    """

    def __init__(self, data: pd.DataFrame, seq_len: int) -> None:
        self.data = data
        self.seq_len = seq_len
        self.columns = list(data.columns)

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - 24

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_window, future_cgm_readings) for a given index.

        Args:
            idx: Starting index of the window.

        Returns:
            Tuple of (X, y) tensors where X is the full feature window and
            y contains the next 24 CGM readings.
        """
        x = self.data.iloc[idx : idx + self.seq_len].values
        y = self.data.iloc[
            idx + self.seq_len : idx + self.seq_len + 24
        ]["Readings (mg/dL)"].values
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )