import os
import pandas as pd
from src.feature_engineering import run_feature_engineering


def test_feature_engineering_creates_features_file():
    run_feature_engineering("config/main.yml")

    features_path = "data/processed/features.csv"

    assert os.path.exists(features_path), "Features file was not created."

    df = pd.read_csv(features_path)
    assert not df.empty, "Features dataset is empty."

    # Check for lag and rolling columns
    expected_cols = [
        "Global_active_power_lag_1",
        "Global_active_power_lag_24",
        "Global_active_power_rolling_mean_24",
    ]

    for col in expected_cols:
        assert col in df.columns, f"Missing expected feature column: {col}"