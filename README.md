# Energy Forecasting System (Production-Grade)

A fully reproducible, Dockerized time-series forecasting pipeline that predicts
household energy consumption using:

- LSTM (PyTorch) with Monte Carlo Dropout for probabilistic forecasting  
- Prophet as a statistical baseline  
- Walk-forward validation (no data leakage)  
- Optuna hyperparameter tuning  
- W&B experiment tracking (offline-safe)  

The system automatically downloads data, trains models, evaluates performance,
and generates forecasts with confidence intervals — all with **one Docker command**.

---

# Project Overview

This project builds a **production-grade time-series forecasting pipeline**
for predicting **Global Active Power** using multivariate historical signals.

Key capabilities:

✔ End-to-end automated pipeline  
✔ Auto dataset download (no manual steps)  
✔ Feature engineering (lags, rolling, calendar)  
✔ Walk-forward validation  
✔ Probabilistic prediction intervals  
✔ Baseline model comparison  
✔ Strict evaluation artifacts  
✔ CI-safe test suite  

---

# Dataset

**Source:**  
UCI Machine Learning Repository  
https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip

### Auto-download behavior
The dataset is:

- Downloaded automatically at runtime
- Extracted into `data/raw/`
- Cleaned and resampled to **hourly frequency**
- Saved as:
data/processed/processed.csv

No manual data handling is required.

---

# Folder Structure

```
energy-forecasting-system/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/              # Optional (kept non-empty)
│
├── results/
│   ├── metrics.json
│   ├── forecasts.csv
│   └── forecast_visualization.png
│
├── logs/
│   └── training.log
│
│
├── tests/
│   ├── conftest.py
│   ├── test_preprocess.py
│   ├── test_feature_engineering.py
│   ├── test_models.py
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
│   └── test_pipeline.py
│
├── config/
│   └── main.yml
│
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

```
---
# Architecture diagram
```
                           ┌──────────────────────────────┐
                           │        Docker Compose        │
                           │  (One-command ML pipeline)   │
                           └──────────────┬───────────────┘
                                          │
                                          ▼
                           ┌──────────────────────────────┐
                           │        train.py (Orchestrator)│
                           │  W&B + Optuna + Walk-forward  │
                           └──────────────┬───────────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            ▼                             ▼                             ▼
┌────────────────────┐       ┌────────────────────┐       ┌────────────────────┐
│   preprocess.py    │       │ feature_engineering│       │      models.py      │
│--------------------│       │--------------------│       │--------------------│
│ Auto download UCI  │       │ Calendar features  │       │ LSTM (PyTorch)     │
│ Extract ZIP        │       │ Lag features       │       │ MC Dropout         │
│ Clean missing      │       │ Rolling statistics │       │ Prophet baseline   │
│ Hourly resample    │       │ Leakage-safe shift │       │ SequenceDataset    │
└─────────┬──────────┘       └─────────┬──────────┘       └─────────┬──────────┘
          │                            │                            │
          ▼                            ▼                            ▼
      data/raw/                 data/processed/              Trained models
                                    │                              │
                                    └──────────────┬───────────────┘
                                                   ▼
                                       ┌────────────────────┐
                                       │    evaluate.py     │
                                       │--------------------│
                                       │ MC Dropout infer   │
                                       │ Prediction bounds  │
                                       │ Metrics (MAE/RMSE) │
                                       │ Prophet comparison │
                                       │ Visualization      │
                                       └─────────┬──────────┘
                                                 │
                         ┌───────────────────────┼────────────────────────┐
                         ▼                       ▼                        ▼
               results/metrics.json   results/forecasts.csv   forecast_visualization.png
                                                 │
                                                 ▼
                                         logs/training.log
```

# Setup & Installation (Docker Only)

> No local Python setup required.  
> Everything runs inside Docker.

---

## 1️.Build and Run the Full Pipeline

```bash
docker-compose up --build
```

### This single command will:

- Download dataset  
- Preprocess data  
- Engineer features  
- Run Optuna + walk-forward LSTM training  
- Train Prophet baseline  
- Generate probabilistic forecasts  
- Save evaluation artifacts  
- Exit with code `0`

---

## Run Tests (After Docker Pipeline)

```bash
pytest tests
```

### Expected Output

```bash
7 passed
```

### Tests Validate

- Data pipeline outputs  
- Feature correctness  
- Model tensor shapes  
- Metrics schema  
- Forecast interval constraints  

---

# Methodology

---

# Data Processing

- Handle missing values (`? → NaN → imputation`)
- Datetime parsing
- Hourly resampling
- Multivariate feature selection

---

# Feature Engineering

## Calendar Features

- `hour`
- `day_of_week`
- `month`
- `weekend_flag`

---

## Lag Features

- `lag_1`
- `lag_24`
- `lag_168`

---

## Rolling Statistics (Leakage-Safe)

- `rolling_mean_24`
- `rolling_std_24`

> All rolling features are **shifted** to prevent data leakage.

---

# Walk-Forward Validation

We use **time-ordered walk-forward splits**.

For each fold:

- Train on past data  
- Validate on future horizon  
- Fit scaler on **train only**  
- Apply the same scaler to validation  

### This ensures:

- ✔ No data leakage  
- ✔ Realistic forecasting scenario  
- ✔ Production-aligned evaluation  

Training logs are saved to:

```
logs/training.log
```

---

# Probabilistic Forecasting (MC Dropout)

The LSTM uses **dropout at inference time** to generate uncertainty.

## Procedure

1. Perform **N stochastic forward passes**
2. Collect prediction distribution
3. Compute:
   - Mean prediction  
   - 5th percentile → Lower bound  
   - 95th percentile → Upper bound  

This produces **90% prediction intervals**.

Output saved to:

```
results/forecasts.csv
```

With enforced constraint:

```
lower_bound ≤ prediction ≤ upper_bound
```

---

# Models

## Deep Learning Model

- PyTorch LSTM  
- Multivariate input  
- Sequence windowing  
- Dropout regularization  
- MC Dropout uncertainty  
- Optuna-tuned hyperparameters  

Saved model:

```
results/best_lstm.pt
```

---

## Baseline Model

- Prophet  
- Univariate statistical model  
- Used for performance comparison  

---

# Hyperparameter Optimization

## Optuna Search Space

- `window_size`: [24, 48, 72, 168]  
- `hidden_size`: [32, 64, 128]  
- `dropout`: [0.1 – 0.5]  
- `learning_rate`: [1e-4 – 1e-2] (log scale)  
- `batch_size`: [32, 64]  

## Objective

**Minimize RMSE using walk-forward validation**

---

# Results Summary

## Artifacts Generated

```
results/metrics.json
results/forecasts.csv
results/forecast_visualization.png
```

---

## Metrics (Example)

### Deep Learning Model

- MAE  
- RMSE  
- MAPE  
- Quantile Loss (P50, P95)  

### Baseline Model

- MAE  
- RMSE  
- MAPE  

---

## Visualization Includes

- ✔ Actual vs Prediction  
- ✔ Shaded Confidence Interval  

---

# Data Leakage Prevention

To ensure strict time-series integrity and production realism, the following safeguards are enforced:

- ✔ Scaler fit **only on training folds**
- ✔ Lag features properly **shifted**
- ✔ Rolling features properly **shifted**
- ✔ Strict **time-ordered splits**

These measures guarantee zero future information leakage into model training.

---

# Reproducibility

The entire pipeline is:

- Dockerized  
- Config-driven  
- Environment-driven  
- Fully automated  

Run anywhere with:

```bash
docker-compose up --build
```

No manual steps required.

---

# CI-Safe Testing Strategy

### Tests:

- ✔ Do **not** retrain models  
- ✔ Validate generated artifacts only  
- ✔ Fast and deterministic  

### Ensures Compatibility With:

- Docker environments  
- CI/CD pipelines  
- Evaluator runtime limits  

Designed for reliable automated execution across systems.

---