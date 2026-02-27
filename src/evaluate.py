import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models import LSTMForecaster, SequenceDataset, ProphetBaseline


# Load YAML Config
def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Quantile Loss for probabilistic evaluation
def quantile_loss(y_true, y_pred, q):
    e = y_true - y_pred
    return np.mean(np.maximum(q * e, (q - 1) * e))


# Prepare Train/Test split with scaler fit ONLY on train
def prepare_test_data(df, target_col, window_size, test_horizon):
    train_df = df.iloc[:-test_horizon]
    test_df = df.iloc[-test_horizon:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    train_scaled = pd.DataFrame(train_scaled, columns=train_df.columns, index=train_df.index)
    test_scaled = pd.DataFrame(test_scaled, columns=test_df.columns, index=test_df.index)

    return train_scaled, test_scaled, train_df, test_df


# Monte Carlo Dropout Prediction
def mc_dropout_predictions(model, dataset, device, n_samples):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds = []

    for X, _ in loader:
        X = X.to(device)
        preds = model.predict_mc(X, n_samples=n_samples)
        all_preds.append(preds)

    all_preds = np.concatenate(all_preds, axis=1)  # (n_samples, total_points)
    return all_preds


# Compute MAE, RMSE, MAPE
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return mae, rmse, mape


# Main Evaluation Pipeline
def run_evaluation(config_path: str):
    config = load_config(config_path)
    paths = config["paths"]

    df = pd.read_csv(paths["features_file"], parse_dates=["datetime"], index_col="datetime")

    target_col = config["data"]["target_column"]
    test_horizon = config["data"]["test_horizon"]
    device = torch.device(config["training"]["device"])

    # Load best hyperparameters (ensures model shape match)
    with open(os.path.join(paths["results_dir"], "best_params.json"), "r") as f:
        best_params = json.load(f)

    window_size = best_params["window_size"]

    # Prepare scaled data (no leakage)
    train_scaled, test_scaled, train_df, test_df = prepare_test_data(
        df, target_col, window_size, test_horizon
    )

    test_dataset = SequenceDataset(test_scaled, target_col, window_size)

    # Load trained LSTM model with best params
    model = LSTMForecaster(
        input_size=test_scaled.shape[1],
        hidden_size=best_params["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=best_params["dropout"],
    ).to(device)

    model.load_state_dict(torch.load(paths["model_path"], map_location=device))

    # Monte Carlo Dropout Predictions
    preds_mc = mc_dropout_predictions(
        model, test_dataset, device, config["mc_dropout"]["n_samples"]
    )

    preds_mean = np.mean(preds_mc, axis=0)
    preds_lower = np.quantile(preds_mc, 0.05, axis=0)
    preds_upper = np.quantile(preds_mc, 0.95, axis=0)

    # Align ground truth
    y_true = test_df[target_col].values[window_size:]

    # Deep Learning Metrics
    dl_mae, dl_rmse, dl_mape = compute_metrics(y_true, preds_mean)

    q50_loss = quantile_loss(y_true, preds_mean, 0.5)
    q95_loss = quantile_loss(y_true, preds_upper, 0.95)

    # Prophet Baseline
    prophet = ProphetBaseline()
    prophet.fit(train_df, target_col)
    prophet_forecast = prophet.predict(periods=test_horizon)
    prophet_preds = prophet_forecast["yhat"].values

    base_mae, base_rmse, base_mape = compute_metrics(
        test_df[target_col].values, prophet_preds
    )

    # Save Metrics (STRICT SCHEMA)
    metrics = {
        "deep_learning_model": {
            "mae": float(dl_mae),
            "rmse": float(dl_rmse),
            "mape": float(dl_mape),
            "quantile_loss_p50": float(q50_loss),
            "quantile_loss_p95": float(q95_loss),
        },
        "baseline_model": {
            "mae": float(base_mae),
            "rmse": float(base_rmse),
            "mape": float(base_mape),
        },
    }

    os.makedirs(paths["results_dir"], exist_ok=True)

    with open(os.path.join(paths["results_dir"], "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save Forecast CSV (STRICT COLUMNS)
    forecast_df = pd.DataFrame(
        {
            "timestamp": test_df.index[window_size:],
            "actual": y_true,
            "prediction": preds_mean,
            "lower_bound": preds_lower,
            "upper_bound": preds_upper,
        }
    )

    forecast_df.to_csv(os.path.join(paths["results_dir"], "forecasts.csv"), index=False)

    # Visualization: Actual vs Prediction + Confidence Band
    plt.figure(figsize=(14, 7))

    plt.plot(
        forecast_df["timestamp"],
        forecast_df["actual"],
        label="Actual",
        linewidth=2,
    )

    plt.plot(
        forecast_df["timestamp"],
        forecast_df["prediction"],
        label="Prediction",
        linewidth=2,
    )

    plt.fill_between(
        forecast_df["timestamp"],
        forecast_df["lower_bound"],
        forecast_df["upper_bound"],
        alpha=0.3,
        label="Prediction Interval (90%)",
    )

    plt.xlabel("Time")
    plt.ylabel("Global Active Power")
    plt.title("Energy Consumption Forecast (LSTM with MC Dropout)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(paths["results_dir"], "forecast_visualization.png")

    plt.savefig(plot_path, dpi=150)  # Ensures file size > 1KB
    plt.close()


# Entry Point
if __name__ == "__main__":
    run_evaluation("config/main.yml")