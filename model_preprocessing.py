"""Scikit-learn preprocessing pipeline for model input features."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

BOLUS_COLS = ["Insulin Delivered", "Insulin on Board", "Carb Size"]
SCALE_COLS = [
    "Commanded Basal Dose (units of insulin)",
    "Insulin Delivered",
    "Carb Size",
    "Insulin on Board",
]


def get_preprocessor(df: pd.DataFrame) -> Pipeline:
    """Build a two-step sklearn Pipeline: zero-imputation then min-max scaling.

    Missing bolus/IOB/carb values are filled with 0 (no delivery recorded).
    Basal and bolus numeric columns are scaled to [0, 1].

    Args:
        df: DataFrame used to determine column layout (values not consumed here;
            call ``pipeline.fit(train_df)`` separately).

    Returns:
        Unfitted sklearn Pipeline ready to be fitted on training data.
    """
    imputer = ColumnTransformer(
        transformers=[
            ("blanks", SimpleImputer(strategy="constant", fill_value=0), BOLUS_COLS)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    imputer.set_output(transform="pandas")

    scaler = ColumnTransformer(
        transformers=[("scaler", MinMaxScaler(), SCALE_COLS)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    scaler.set_output(transform="pandas")

    pipeline = Pipeline([("imputer", imputer), ("scaler", scaler)])
    return pipeline