"""Data processing utilities for CGM, basal, and bolus data merging."""

import numpy as np
import pandas as pd

FLOOR_INTERVAL = "5min"
CGM_MIN = 40
CGM_RANGE = 360
MINUTES_PER_DAY = 1440
IOB_DURATION_MINUTES = 240
IOB_WINDOW_STEPS = 48


def round_times(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Round a datetime column down to the nearest 5-minute interval.

    Args:
        data: DataFrame containing the datetime column.
        col_name: Name of the column to round and rename to 'Time'.

    Returns:
        DataFrame with the column renamed to 'Time' and floored to 5 minutes.
    """
    rounded = data.rename(columns={col_name: "Time"})
    rounded["Time"] = pd.to_datetime(rounded["Time"]).dt.floor(FLOOR_INTERVAL)
    return rounded


def calc_iob(dose: float, time: float) -> float:
    """Calculate insulin on board for a given dose at a given time offset.

    Uses a simple linear decay model over IOB_DURATION_MINUTES minutes.

    Args:
        dose: Insulin dose in units.
        time: Minutes elapsed since the dose was delivered.

    Returns:
        Remaining insulin on board in units.
    """
    return dose * (1 - time / IOB_DURATION_MINUTES)


class MergedDf:
    """Merge CGM, basal, and bolus DataFrames into a single time-aligned DataFrame.

    Attributes:
        cgm: Continuous glucose monitor readings.
        bolus: Bolus insulin delivery records.
        basal: Basal insulin delivery records.
        df: The merged and feature-engineered DataFrame.
    """

    def __init__(
        self,
        cgm: pd.DataFrame,
        bolus: pd.DataFrame,
        basal: pd.DataFrame,
    ) -> None:
        self.cgm = cgm
        self.bolus = bolus
        self.basal = basal
        self.df = self._merge()
        self._add_iob_col()

    def _add_iob_col(self) -> None:
        """Compute and add the 'Insulin on Board' column to the merged DataFrame."""
        self.df["Insulin on Board"] = np.zeros(len(self.df))
        iob_loc = self.df.columns.get_loc("Insulin on Board")

        for i in range(len(self.df)):
            dose = self.df["Insulin Delivered"].iloc[i]
            if pd.notna(dose):
                for j in range(IOB_WINDOW_STEPS):
                    idx = i + j
                    if idx < len(self.df):
                        self.df.iat[idx, iob_loc] += calc_iob(dose, 5 * j)

    def _merge(self) -> pd.DataFrame:
        """Align and merge CGM, basal, and bolus data on rounded timestamps.

        Returns:
            Merged DataFrame with time-encoding and normalised CGM column.
        """
        self.cgm = round_times(self.cgm, "Event Date Time")
        self.basal = round_times(self.basal, "Event Date Time")
        self.bolus = round_times(self.bolus, "Completion Date Time")

        df = (
            self.cgm.merge(self.basal, on="Time", how="left")
            .merge(self.bolus, on="Time", how="left")
            .sort_values("Time")
            .reset_index(drop=True)
        )

        minutes = df["Time"].dt.hour * 60 + df["Time"].dt.minute
        df["time_sin"] = np.sin(2 * np.pi * minutes / MINUTES_PER_DAY)
        df["time_cos"] = np.cos(2 * np.pi * minutes / MINUTES_PER_DAY)

        df["Readings (mg/dL)"] = (
            (df["Readings (mg/dL)"].ffill().bfill() - CGM_MIN) / CGM_RANGE
        )
        df["Commanded Basal Dose (units of insulin)"] = (
            df["Commanded Basal Dose (units of insulin)"].ffill().bfill()
        )

        df = df.drop(columns="Time")
        return df