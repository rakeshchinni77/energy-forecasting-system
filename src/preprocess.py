import os
import zipfile
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

DATASET_URL = os.getenv("DATASET_URL")

# Logging Configuration
def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# Config Loader
def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Download Dataset
def download_dataset(raw_dir: str):
    os.makedirs(raw_dir, exist_ok=True)

    zip_path = os.path.join(raw_dir, "household_power_consumption.zip")

    if os.path.exists(zip_path):
        logging.info("Dataset ZIP already exists. Skipping download.")
        return zip_path

    logging.info("Downloading dataset...")
    response = requests.get(DATASET_URL, stream=True, timeout=60)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logging.info("Dataset downloaded successfully.")
    return zip_path


# Extract Dataset
def extract_dataset(zip_path: str, raw_dir: str):
    logging.info("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)
    logging.info("Extraction complete.")


# Load & Clean Data
def load_and_clean_data(raw_dir: str, config: dict):
    file_path = os.path.join(raw_dir, "household_power_consumption.txt")

    if not os.path.exists(file_path):
        raise FileNotFoundError("Expected dataset file not found after extraction.")

    logging.info("Loading dataset...")

    df = pd.read_csv(
        file_path,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )

    logging.info("Parsing datetime...")
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    df = df.drop(columns=["Date", "Time"])
    df = df.set_index("datetime")

    logging.info("Converting numeric columns...")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logging.info("Handling missing values (ffill → bfill)...")
    df = df.ffill().bfill()

    logging.info("Resampling to hourly frequency...")
    df = df.resample(config["data"]["resample_frequency"]).mean()

    target_col = config["data"]["target_column"]

    # Keep multivariate features including target
    df = df.dropna(subset=[target_col])

    logging.info(f"Final dataset shape after resampling: {df.shape}")

    return df


# Save Processed Data
def save_processed_data(df: pd.DataFrame, processed_file: str):
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    df.to_csv(processed_file)
    logging.info(f"Processed data saved to {processed_file}")


# Main Preprocessing Pipeline
def run_preprocessing(config_path: str):
    config = load_config(config_path)

    raw_dir = config["paths"]["raw_data_dir"]
    processed_file = config["paths"]["processed_file"]
    log_dir = config["paths"]["logs_dir"]

    setup_logger(log_dir)

    logging.info("Starting preprocessing pipeline...")

    zip_path = download_dataset(raw_dir)
    extract_dataset(zip_path, raw_dir)

    df = load_and_clean_data(raw_dir, config)

    save_processed_data(df, processed_file)

    logging.info("Preprocessing completed successfully.")


if __name__ == "__main__":
    run_preprocessing("config/main.yml")