"""Live anomaly detection using the Tandem pump API."""
import os
import time
from datetime import datetime, timedelta

import joblib
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tconnectsync.api.tandemsource import TandemSourceApi
from tconnectsync.eventparser.events import (
    LidBasalDelivery,
    LidBolusCompleted,
    LidBolusRequestedMsg1,
    LidCgmDataG7,
)

from data_processing import MergedDf
from model import CgmLstm
from visualizer import graph_comparison, inverse_scale, to_numpy
from dotenv import load_dotenv

HISTORY_HOURS = 6
INPUT_WINDOW = 72
REAL_WINDOW = 24
OFFSET = 24

STRONG_THRESHOLD_MULTIPLIER = 3
MILD_THRESHOLD_MULTIPLIER = 2

DETECTION_STATS_PATH = "detection_stats.joblib"

VALUE_WEIGHT = 1.0
SLOPE_WEIGHT = 1.5
RISE_WEIGHT = 3.0


def compute_error_detection_score(
    pred: torch.Tensor, real: torch.Tensor
) -> float:
    """Score the difference between predicted and actual CGM trajectories.

    Combines value error, slope error, and an asymmetric rising-slope penalty
    to flag sensor anomalies or unexpected glucose changes.

    Args:
        pred: Predicted CGM sequence (1-D tensor, mg/dL).
        real: Actual CGM sequence (1-D tensor, mg/dL).

    Returns:
        Scalar anomaly score (higher = more anomalous).
    """
    value_loss = nn.MSELoss()(pred, real)

    slope_pred = pred[1:] - pred[:-1]
    slope_real = real[1:] - real[:-1]
    slope_loss = F.smooth_l1_loss(slope_pred, slope_real)

    rise_mask = (slope_real > 0).float()
    rise_error = (rise_mask * (slope_pred - slope_real).pow(2)).mean()

    return (
        VALUE_WEIGHT * value_loss
        + SLOPE_WEIGHT * slope_loss
        + RISE_WEIGHT * rise_error
    ).item()


class Simulate:
    """Pull recent pump data, run the LSTM, and flag CGM anomalies.

    Args:
        email: Tandem account e-mail address.
        password: Tandem account password.
        pump_id: Numeric pump identifier.
    """

    _EMPTY_COLUMNS = [
        "Readings (mg/dL)",
        "Commanded Basal Dose (units of insulin)",
        "Insulin Delivered",
        "Carb Size",
        "time_sin",
        "time_cos",
        "Insulin on Board",
    ]

    def __init__(self, email: str, password: str, pump_id: int) -> None:
        self.df = pd.DataFrame(columns=self._EMPTY_COLUMNS, dtype=float)
        self.pump_id = pump_id

        self.api = TandemSourceApi(email, password)
        self.api.login(email, password)

        self.model = CgmLstm()
        self.model.load_state_dict(torch.load("model_weights.pth"))

        self.preprocessor = joblib.load("preprocessor.joblib")

        detection_stats = joblib.load(DETECTION_STATS_PATH)
        self.anomaly_mean: float = detection_stats["mean"]
        self.anomaly_std: float = detection_stats["std"]

    def _fetch_recent_data(self) -> None:
        """Query the pump API for the last 6 hours of events and build ``self.df``."""
        cgm_rows: list[dict] = []
        basal_rows: list[dict] = []
        bolus_rows: list[dict] = []
        boluses: dict[int, dict] = {}

        all_events = self.api.pump_events(
            self.pump_id,
            datetime.now() - timedelta(hours=HISTORY_HOURS),
            datetime.now(),
        )

        for event in all_events:
            if isinstance(event, LidCgmDataG7):
                cgm_rows.append(
                    {
                        "Event Date Time": event.eventTimestamp.datetime,
                        "Readings (mg/dL)": event.currentglucosedisplayvalue,
                    }
                )
            elif isinstance(event, LidBasalDelivery):
                basal_rows.append(
                    {
                        "Event Date Time": event.eventTimestamp.datetime,
                        "Commanded Basal Dose (units of insulin)": event.commandedRate / 1000,
                    }
                )
            elif isinstance(event, LidBolusRequestedMsg1):
                boluses[event.bolusid] = {
                    "Completion Date Time": event.eventTimestamp.datetime,
                    "Carb Size": event.carbamount,
                }
            elif isinstance(event, LidBolusCompleted):
                boluses.setdefault(event.bolusid, {})
                boluses[event.bolusid]["Insulin Delivered"] = event.insulindelivered

        for data in boluses.values():
            bolus_rows.append(
                {
                    "Completion Date Time": data.get("Completion Date Time"),
                    "Insulin Delivered": data.get("Insulin Delivered", 0),
                    "Carb Size": data.get("Carb Size", 0),
                }
            )

        cgm_df = pd.DataFrame(cgm_rows, columns=["Event Date Time", "Readings (mg/dL)"])
        basal_df = pd.DataFrame(
            basal_rows,
            columns=["Event Date Time", "Commanded Basal Dose (units of insulin)"],
        )
        bolus_df = pd.DataFrame(
            bolus_rows,
            columns=["Completion Date Time", "Insulin Delivered", "Carb Size"],
        )

        self.df = MergedDf(cgm_df, bolus_df, basal_df).df
        self.df = self.preprocessor.transform(self.df)

    def detect(self) -> None:
        """Fetch live data, run inference, plot results, and print anomaly verdict."""
        self._fetch_recent_data()

        window = self.df.iloc[-INPUT_WINDOW:-OFFSET].values
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

        predictions = inverse_scale(to_numpy(self.model(x)))
        in_vals = inverse_scale(self.df["Readings (mg/dL)"].iloc[-INPUT_WINDOW:-OFFSET].values)
        real_vals = inverse_scale(self.df["Readings (mg/dL)"].iloc[-REAL_WINDOW:].values)

        history_start = datetime.now() - timedelta(minutes=(INPUT_WINDOW - 1) * 5)
        graph_comparison(in_vals, real_vals, predictions, start_time=history_start)

        detection_score = compute_error_detection_score(
            torch.tensor(predictions), torch.tensor(real_vals)
        )

        strong_threshold = self.anomaly_mean + STRONG_THRESHOLD_MULTIPLIER * self.anomaly_std
        mild_threshold = self.anomaly_mean + MILD_THRESHOLD_MULTIPLIER * self.anomaly_std

        sep = "─" * 52
        print(f"\n{sep}")
        print(f"  CGM Anomaly Detection Report  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
        print(sep)
        print(
            f"  Detection score   : {detection_score:>8.1f}   (thresholds: {mild_threshold:.0f} / {strong_threshold:.0f})")
        if detection_score > strong_threshold:
            status = "STRONG ANOMALY — CGM readings diverge significantly from forecast"
        elif detection_score > mild_threshold:
            status = "ANOMALY — CGM readings diverge from forecast"
        else:
            status = "No anomaly detected"
        print(f"  {status}")


if __name__ == "__main__":
    load_dotenv()
    sim = Simulate(
        email = os.getenv("TANDEM_EMAIL"),
        password = os.getenv("TANDEM_PASSWORD"),
        pump_id = int(os.getenv("TANDEM_PUMP_ID")),
    )
    while True:
        try:
            sim.detect()
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M')}] Detection failed: {e}")
        time.sleep(600)