# Multi-Variate Energy Consumption Forecasting System with Deep Learning

## Project Overview
This project builds a production-grade, containerized deep learning system for multi-variate energy consumption forecasting using the UCI Individual Household Electric Power Consumption dataset.  
It implements LSTM-based forecasting with probabilistic prediction intervals, a Prophet baseline, walk-forward validation, Optuna hyperparameter tuning, and Weights & Biases experiment tracking.

## Dataset
**Source:** UCI Machine Learning Repository  
**Name:** Individual Household Electric Power Consumption  
The dataset is automatically downloaded inside the pipeline using the URL provided in `.env`.

No manual download is required.
---
## Folder Structure

```
energy-forecasting-system/
├── data/
├── notebooks/
├── results/
├── logs/
├── src/
├── tests/
├── config/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

```
---

## Setup & Installation
Instructions will be added after Docker pipeline implementation.

## Methodology
Will include:
- Data preprocessing
- Feature engineering
- LSTM with Monte Carlo Dropout
- Prophet baseline
- Walk-forward validation
- Optuna tuning
- W&B tracking

## Results
To be generated after training:
- metrics.json
- forecasts.csv
- forecast_visualization.png