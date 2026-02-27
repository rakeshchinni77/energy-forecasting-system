import os
import logging
import pandas as pd
import yaml
from pathlib import Path


# Config Loader
def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Logger Setup
def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# Calendar Features
def add_calendar_features(df: pd.DataFrame):
    logging.info("Adding calendar features...")

    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


# Lag Features
def add_lag_features(df: pd.DataFrame, target_col: str, lags: list):
    logging.info(f"Adding lag features: {lags}")

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    return df


# Rolling Features (Leakage-safe)
def add_rolling_features(df: pd.DataFrame, target_col: str, windows: list):
    logging.info(f"Adding rolling features: {windows}")

    for window in windows:
        df[f"{target_col}_rolling_mean_{window}"] = (
            df[target_col].shift(1).rolling(window=window).mean()
        )
        df[f"{target_col}_rolling_std_{window}"] = (
            df[target_col].shift(1).rolling(window=window).std()
        )

    return df


# Main Feature Engineering Pipeline
def run_feature_engineering(config_path: str):
    config = load_config(config_path)

    processed_file = config["paths"]["processed_file"]
    features_file = config["paths"]["features_file"]
    log_dir = config["paths"]["logs_dir"]

    setup_logger(log_dir)

    logging.info("Starting feature engineering pipeline...")

    if not os.path.exists(processed_file):
        raise FileNotFoundError("Processed file not found. Run preprocessing first.")

    df = pd.read_csv(processed_file, parse_dates=["datetime"], index_col="datetime")

    target_col = config["data"]["target_column"]
    lags = config["features"]["lag_hours"]
    rolling_windows = config["features"]["rolling_windows"]

    # Calendar
    if config["features"]["calendar_features"]:
        df = add_calendar_features(df)

    # Lag
    df = add_lag_features(df, target_col, lags)

    # Rolling
    df = add_rolling_features(df, target_col, rolling_windows)

    logging.info("Dropping rows with NaNs from lag/rolling operations...")
    df = df.dropna()

    logging.info(f"Final feature dataset shape: {df.shape}")

    os.makedirs(os.path.dirname(features_file), exist_ok=True)
    df.to_csv(features_file)

    logging.info(f"Feature data saved to {features_file}")
    logging.info("Feature engineering completed successfully.")


if __name__ == "__main__":
    run_feature_engineering("config/main.yml")