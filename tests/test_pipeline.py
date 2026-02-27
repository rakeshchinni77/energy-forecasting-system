import os
import json
import pandas as pd


def test_pipeline_outputs_exist():
    """
    Verifies that the full pipeline generated required artifacts.

    NOTE:
    This test assumes the training + evaluation pipeline
    has already been executed (e.g., via Docker).
    """

    required_files = [
        "data/processed/processed.csv",
        "data/processed/features.csv",
        "results/best_lstm.pt",
        "results/metrics.json",
        "results/forecasts.csv",
        "results/forecast_visualization.png",
        "logs/training.log",
    ]

    for file_path in required_files:
        assert os.path.exists(file_path), f"Missing required artifact: {file_path}"


def test_metrics_json_schema():
    """
    Verifies metrics.json follows strict schema and contains float values.
    """

    metrics_path = "results/metrics.json"
    assert os.path.exists(metrics_path), "metrics.json not found."

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Top-level keys
    assert "deep_learning_model" in metrics
    assert "baseline_model" in metrics

    dl_keys = ["mae", "rmse", "mape", "quantile_loss_p50", "quantile_loss_p95"]
    base_keys = ["mae", "rmse", "mape"]

    for key in dl_keys:
        assert key in metrics["deep_learning_model"], f"Missing DL metric: {key}"
        assert isinstance(metrics["deep_learning_model"][key], float)

    for key in base_keys:
        assert key in metrics["baseline_model"], f"Missing baseline metric: {key}"
        assert isinstance(metrics["baseline_model"][key], float)


def test_forecast_csv_structure():
    """
    Verifies forecasts.csv columns and interval constraints.
    """

    forecast_path = "results/forecasts.csv"
    assert os.path.exists(forecast_path), "forecasts.csv not found."

    df = pd.read_csv(forecast_path)

    expected_cols = [
        "timestamp",
        "actual",
        "prediction",
        "lower_bound",
        "upper_bound",
    ]

    for col in expected_cols:
        assert col in df.columns, f"Missing column in forecasts.csv: {col}"

    # Interval constraint check
    assert (df["lower_bound"] <= df["prediction"]).all(), "Lower bound > prediction"
    assert (df["prediction"] <= df["upper_bound"]).all(), "Prediction > upper bound"