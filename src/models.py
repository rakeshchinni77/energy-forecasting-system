import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from prophet import Prophet
from cmdstanpy import install_cmdstan


# Sequence Dataset for LSTM
# Creates sliding windows for time series forecasting
class SequenceDataset(Dataset):
    def __init__(self, data, target_col, window_size):
        self.data = data.values
        self.target_idx = data.columns.get_loc(target_col)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.window_size]
        y = self.data[idx + self.window_size, self.target_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# LSTM Forecaster with Dropout (for MC Dropout)
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out.squeeze(-1)

    # Monte Carlo Dropout Inference
    # Keeps dropout ON during prediction to get uncertainty
    def predict_mc(self, x, n_samples=50):
        self.train()  # Enable dropout at inference

        preds = []
        for _ in range(n_samples):
            preds.append(self.forward(x).detach().cpu().numpy())

        preds = np.array(preds)  # (n_samples, batch_size)
        return preds


# =========================================================
# Prophet Baseline Wrapper (Docker-safe, no CmdStan build)
# =========================================================
class ProphetBaseline:
    def __init__(self):
        """
        Use Prophet with default backend.
        Avoids CmdStan build at runtime (Docker-safe).
        """
        self.model = Prophet()

    def fit(self, df: pd.DataFrame, target_col: str):
        prophet_df = df.reset_index()[["datetime", target_col]]
        prophet_df.columns = ["ds", "y"]
        self.model.fit(prophet_df)

    def predict(self, periods: int, freq: str = "H"):
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast[["ds", "yhat"]].tail(periods).set_index("ds")