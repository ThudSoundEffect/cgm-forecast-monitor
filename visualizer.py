"""Visualisation tools for CGM predictions and aggregate statistics."""

import random
from datetime import datetime, timedelta

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_processing import round_times

CGM_MIN = 40
CGM_RANGE = 360
PLOT_Y_MIN = 40
PLOT_Y_MAX = 400

HYPO_THRESHOLD = 70
LOW_THRESHOLD = 80
HIGH_THRESHOLD = 180
HYPER_THRESHOLD = 250

STEP_MINUTES = 5


def to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a tensor or array like to a flat NumPy array.

    Args:
        x: Input tensor or array.

    Returns:
        1-D NumPy float array.
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.array(x).flatten()


def inverse_scale(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Reverse the CGM normalization: ``value * CGM_RANGE + CGM_MIN``.

    Args:
        x: Normalized CGM values.

    Returns:
        CGM values in mg/dL.
    """
    x = to_numpy(x)
    return x * CGM_RANGE + CGM_MIN


def _glucose_color(value: float) -> str:
    """Return a color string reflecting glucose severity."""
    if value < HYPO_THRESHOLD:
        return "#d62728"   # red
    if value < LOW_THRESHOLD:
        return "#ff7f0e"   # orange
    if value > HYPER_THRESHOLD:
        return "#d62728"   # red
    if value > HIGH_THRESHOLD:
        return "#ff7f0e"   # orange
    return "#2ca02c"       # green


def _add_glucose_bands(ax: plt.Axes) -> None:
    ax.axhspan(PLOT_Y_MIN, HYPO_THRESHOLD, alpha=0.15, color="#d62728", zorder=0)
    ax.axhspan(HYPO_THRESHOLD, LOW_THRESHOLD, alpha=0.15, color="#ff7f0e", zorder=0)
    ax.axhspan(LOW_THRESHOLD, HIGH_THRESHOLD, alpha=0.15, color="#2ca02c", zorder=0)
    ax.axhspan(HIGH_THRESHOLD, HYPER_THRESHOLD, alpha=0.15, color="#ff7f0e", zorder=0)
    ax.axhspan(HYPER_THRESHOLD, PLOT_Y_MAX, alpha=0.15, color="#d62728", zorder=0)

    for y, label, va in [
        (HYPO_THRESHOLD, "Hypo (<70)", "bottom"),
        (HIGH_THRESHOLD, "High (>180)", "bottom"),
        (HYPER_THRESHOLD, "Hyper (>250)", "bottom"),
    ]:
        ax.axhline(y, color="grey", linewidth=0.6, linestyle="--", zorder=1)
        ax.text(
            0.5, y + 2, label,
            transform=ax.get_yaxis_transform(),
            fontsize=7, color="grey", va=va,
        )


def graph_comparison(
    input_vals: np.ndarray,
    real_vals: np.ndarray,
    predicted_vals: np.ndarray,
    start_time: datetime | None = None,
) -> None:
    """Plot input history, truth, and model predictions.

    With:
    - Glucose range bands (hypo / in-range / hyper zones)
    - Color-coded true CGM reading line
    - Real time x-axis labels when ``start_time`` is provided

    Args:
        input_vals: Historical CGM readings (mg/dL) fed into the model.
        real_vals: Actual CGM readings (mg/dL).
        predicted_vals: Model forecasted CGM readings (mg/dL).
        start_time: Datetime of the first input step; enables clock labels on
            the x-axis. Defaults to None (step numbers used instead).
    """
    n_in = len(input_vals)
    n_fut = len(real_vals)
    total_steps = n_in + n_fut

    if start_time is not None:
        times = [start_time + timedelta(minutes=i * STEP_MINUTES) for i in range(total_steps)]
        x_labels = [t.strftime("%H:%M") for t in times]
        tick_every = max(1, total_steps // 12)
        x_ticks = list(range(0, total_steps, tick_every))
        tick_labels = [x_labels[i] for i in x_ticks]
    else:
        x_ticks = list(range(0, total_steps, max(1, total_steps // 12)))
        tick_labels = [str(i) for i in x_ticks]

    x_input = np.arange(n_in)
    x_future = np.arange(n_in, n_in + n_fut)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#f9f9f9")
    ax.set_facecolor("#f9f9f9")

    _add_glucose_bands(ax)

    for i in range(n_in - 1):
        seg_color = _glucose_color((input_vals[i] + input_vals[i + 1]) / 2)
        ax.plot(x_input[i : i + 2], input_vals[i : i + 2], color=seg_color, linewidth=2, zorder=3)
    ax.scatter(x_input, input_vals, c=[_glucose_color(v) for v in input_vals],
               zorder=3, linewidths=1)

    for i in range(n_fut - 1):
        seg_color = _glucose_color((real_vals[i] + real_vals[i + 1]) / 2)
        ax.plot(x_future[i : i + 2], real_vals[i : i + 2], color=seg_color, linewidth=2, zorder=4)
    ax.scatter(x_future, real_vals, c=[_glucose_color(v) for v in real_vals],
               zorder=4, linewidths=1)

    ax.axvline(n_in - 0.5, color="steelblue", linewidth=2, linestyle=":", alpha=0.7)
    ax.text(
        n_in - 0.3, PLOT_Y_MAX - 15, "Forecasted →",
        fontsize=8, color="steelblue", va="top",
    )

    ax.plot(x_future, predicted_vals, color="#1f77b4", linewidth=2,
            linestyle="--", label="Predicted", zorder=5)
    ax.scatter(x_future, predicted_vals, color="#1f77b4", zorder=6)

    ax.set_xlim(-1, total_steps)
    ax.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Time" if start_time else "Time Step (5-min intervals)", fontsize=9)
    ax.set_ylabel("Glucose (mg/dL)", fontsize=9)
    ax.set_title("CGM Forecast", fontsize=12, fontweight="bold")

    zone_patches = [
        patches.Patch(color="#2ca02c", label="In range (80–180)"),
        patches.Patch(color="#ff7f0e", label="Low / High"),
        patches.Patch(color="#d62728", label="Hypo / Hyper"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles + zone_patches,
        labels + [p.get_label() for p in zone_patches],
        loc="upper left",
        fontsize=8,
    )

    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    plt.show()


def aggregate_stats(cgm_df: "pd.DataFrame") -> "pd.DataFrame":
    """Compute mean, Q1, and Q3 of CGM readings grouped by time of day.

    Args:
        cgm_df: DataFrame with 'Time' and 'Readings (mg/dL)' columns.

    Returns:
        DataFrame with columns ['Time', 'mean', 'Q1', 'Q3'].
    """
    return (
        cgm_df.groupby("Time")["Readings (mg/dL)"]
        .agg(
            mean="mean",
            Q1=lambda x: x.quantile(0.25),
            Q3=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )


def graph_avg(cgm_df: "pd.DataFrame") -> None:
    """Plot daily average CGM with Q1/Q3 bands across all available days.

    Args:
        cgm_df: Raw CGM DataFrame with an 'Event Date Time' column.
    """
    cgm_df = round_times(cgm_df, "Event Date Time")
    cgm_df["Time"] = cgm_df["Time"].dt.time
    cgm_df = aggregate_stats(cgm_df)

    x_vals = np.arange(288)
    plt.figure(figsize=(12, 5))
    plt.ylim(PLOT_Y_MIN, PLOT_Y_MAX)
    plt.scatter(x_vals, cgm_df["Q1"], color="red")
    plt.scatter(x_vals, cgm_df["mean"], color="black")
    plt.scatter(x_vals, cgm_df["Q3"], color="red")
    plt.xlabel("Time Step")
    plt.ylabel("Readings (mg/dL)")
    plt.grid(True)
    plt.show()


class Visualizer:
    """Interactive visualizer that cycles through random test set predictions.

    Args:
        model: Trained CgmLstm model.
        data_set: SeriesSet instance providing (X, y) samples.
    """

    def __init__(self, model: "CgmLstm", data_set: "SeriesSet") -> None:  # noqa: F821
        self.model = model
        self.data_set = data_set
        self.cgm_idx = data_set.columns.index("Readings (mg/dL)")

    def visualize(self) -> None:
        """Loop: display a random prediction, then ask whether to continue."""
        running = True
        while running:
            idx = random.randint(0, len(self.data_set) - 1)
            data_in, data_out = self.data_set[idx]
            predictions = self.model(data_in)

            input_vals = inverse_scale(to_numpy(data_in[:, self.cgm_idx]))
            predicted_vals = inverse_scale(to_numpy(predictions))
            output_vals = inverse_scale(to_numpy(data_out))

            graph_comparison(input_vals, output_vals, predicted_vals)
            running = input("Visualize another? (y/n): ").lower() == "y"