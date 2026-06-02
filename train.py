"""Training script: load data, fit preprocessor, train and evaluate the LSTM."""

import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_processing import MergedDf
from dataset import SeriesSet
from model import CgmLstm
from model_preprocessing import get_preprocessor
from simulate import compute_error_detection_score
from visualizer import inverse_scale, to_numpy

TRAIN_SPLIT = 0.8
SEQ_LEN = 48
BATCH_SIZE = 16
NUM_EPOCHS = 10
PREPROCESSOR_PATH = "preprocessor.joblib"
WEIGHTS_PATH = "model_weights.pth"
DETECTION_STATS_PATH = "detection_stats.joblib"
TRAINING_DIR = "Training"


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract zip archives in the Training directory and concat CSV shards.

    Returns:
        Tuple of (cgm_df, bolus_df, basal_df) concatenated across CSV's.
    """
    all_basal: list[pd.DataFrame] = []
    all_cgm: list[pd.DataFrame] = []
    all_bolus: list[pd.DataFrame] = []

    for zip_path in Path(TRAINING_DIR).glob("*.zip"):
        extract_dir = Path(zip_path.stem)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

        for file in extract_dir.glob("*.csv"):
            name = file.name
            if name.endswith("-Basal-doses.csv"):
                all_basal.append(pd.read_csv(file, usecols=[2, 3]))
            elif name.endswith("-CGM.csv"):
                all_cgm.append(pd.read_csv(file, usecols=[3, 4]))
            elif name.endswith("-Bolus.csv"):
                all_bolus.append(pd.read_csv(file, usecols=[5, 6, 16]))

    return (
        pd.concat(all_cgm, ignore_index=True),
        pd.concat(all_bolus, ignore_index=True),
        pd.concat(all_basal, ignore_index=True),
    )


def aggregate_detection_losses(
    model: CgmLstm, data: SeriesSet
) -> list[float]:
    """Compute anomaly detection scores for every window in a dataset.

    Args:
        model: Trained CgmLstm instance (must be in eval mode).
        data: SeriesSet to iterate over.

    Returns:
        List of scalar detection scores, one per sample.
    """
    losses: list[float] = []
    for i in range(len(data)):
        history, real = data[i]
        predictions = model(history)

        predicted_mgdl = inverse_scale(to_numpy(predictions))
        real_mgdl = inverse_scale(to_numpy(real))

        loss = compute_error_detection_score(
            torch.tensor(predicted_mgdl), torch.tensor(real_mgdl)
        )
        losses.append(loss)
    return losses


if __name__ == "__main__":
    cgm_df, bolus_df, basal_df = load_training_data()
    merged = MergedDf(cgm_df, bolus_df, basal_df)

    train_size = int(len(merged.df) * TRAIN_SPLIT)
    train_df = merged.df.iloc[:train_size]
    test_df = merged.df.iloc[train_size:]

    preprocessor = get_preprocessor(merged.df)
    preprocessor.fit(train_df)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Saved {PREPROCESSOR_PATH}")

    train_scaled = preprocessor.transform(train_df)
    test_scaled = preprocessor.transform(test_df)

    train_data = SeriesSet(train_scaled, SEQ_LEN)
    test_data = SeriesSet(test_scaled, SEQ_LEN)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = CgmLstm()
    model.train_model(NUM_EPOCHS, train_loader)
    model.evaluate_model(test_loader)

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Saved {WEIGHTS_PATH}")
    model.eval()
    detection_losses = aggregate_detection_losses(model, train_data)
    detection_stats = {
        "mean": float(np.mean(detection_losses)),
        "std": float(np.std(detection_losses)),
    }
    joblib.dump(detection_stats, DETECTION_STATS_PATH)
    print(f"Saved {DETECTION_STATS_PATH}  (mean={detection_stats['mean']:.1f}, std={detection_stats['std']:.1f})")
