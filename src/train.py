import os
import json
import yaml
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
import wandb

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from src.preprocess import run_preprocessing
from src.feature_engineering import run_feature_engineering
from src.models import LSTMForecaster, SequenceDataset


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def evaluate_model(model, loader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            outputs = model(X).cpu().numpy()
            preds.extend(outputs)
            targets.extend(y.numpy())

    rmse = np.sqrt(mean_squared_error(targets, preds))
    return rmse


def walk_forward_splits(df, n_folds, val_size):
    total_size = len(df)
    fold_size = (total_size - val_size) // n_folds

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        val_start = train_end
        val_end = val_start + val_size

        train_df = df.iloc[:train_end]
        val_df = df.iloc[val_start:val_end]

        yield fold + 1, train_df, val_df


def run_training(config_path: str):
    config = load_config(config_path)

    paths = config["paths"]
    setup_logger(paths["logs_dir"])

    logging.info("Starting training pipeline...")

    if not os.path.exists(paths["processed_file"]):
        run_preprocessing(config_path)

    if not os.path.exists(paths["features_file"]):
        run_feature_engineering(config_path)

    df = pd.read_csv(paths["features_file"], parse_dates=["datetime"], index_col="datetime")

    target_col = config["data"]["target_column"]
    val_size = config["data"]["validation_horizon"]
    n_folds = config["walk_forward"]["n_folds"]
    device = torch.device(config["training"]["device"])

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "energy-forecasting-system"),
        config=config,
        mode="offline",
    )

    def objective(trial):
        window_size = trial.suggest_categorical("window_size", config["optuna"]["window_size"])
        hidden_size = trial.suggest_categorical("hidden_size", config["optuna"]["hidden_size"])
        dropout = trial.suggest_categorical("dropout", config["optuna"]["dropout"])
        lr = trial.suggest_float("learning_rate", *config["optuna"]["learning_rate"], log=True)
        batch_size = trial.suggest_categorical("batch_size", config["optuna"]["batch_size"])

        fold_rmses = []

        for fold, train_df, val_df in walk_forward_splits(df, n_folds, val_size):
            logging.info(f"Starting walk-forward fold {fold}")
            logging.info(f"Training fold {fold}")

            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_df)
            val_scaled = scaler.transform(val_df)

            train_scaled = pd.DataFrame(train_scaled, columns=train_df.columns, index=train_df.index)
            val_scaled = pd.DataFrame(val_scaled, columns=val_df.columns, index=val_df.index)

            train_dataset = SequenceDataset(train_scaled, target_col, window_size)
            val_dataset = SequenceDataset(val_scaled, target_col, window_size)

            if len(train_dataset) == 0 or len(val_dataset) == 0:
                logging.warning(f"Skipping fold {fold} due to insufficient sequence length.")
                continue

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = LSTMForecaster(
                input_size=train_scaled.shape[1],
                hidden_size=hidden_size,
                num_layers=config["model"]["num_layers"],
                dropout=dropout,
            ).to(device)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            for _ in range(config["training"]["epochs"]):
                train_one_epoch(model, train_loader, criterion, optimizer, device)

            logging.info(f"Evaluating fold {fold}")
            rmse = evaluate_model(model, val_loader, device)

            fold_rmses.append(rmse)
            wandb.log({f"fold_{fold}_rmse": rmse})

        if len(fold_rmses) == 0:
            return float("inf")

        avg_rmse = np.mean(fold_rmses)
        wandb.log({"avg_rmse": avg_rmse})

        return avg_rmse

    study = optuna.create_study(direction=config["optuna"]["direction"])
    study.optimize(objective, n_trials=config["optuna"]["n_trials"])

    best_params = study.best_params
    logging.info(f"Best params: {best_params}")

    os.makedirs(paths["results_dir"], exist_ok=True)

    with open(os.path.join(paths["results_dir"], "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

    dataset = SequenceDataset(df_scaled, target_col, best_params["window_size"])
    loader = DataLoader(dataset, batch_size=best_params["batch_size"], shuffle=False)

    model = LSTMForecaster(
        input_size=df_scaled.shape[1],
        hidden_size=best_params["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=best_params["dropout"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])

    for _ in range(config["training"]["epochs"]):
        train_one_epoch(model, loader, criterion, optimizer, device)

    torch.save(model.state_dict(), paths["model_path"])

    logging.info(f"Best model saved to {paths['model_path']}")
    wandb.finish()
    logging.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main.yml")
    args = parser.parse_args()

    run_training(args.config)