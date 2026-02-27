import os
import pandas as pd
from src.preprocess import run_preprocessing


def test_preprocessing_creates_processed_file():
    run_preprocessing("config/main.yml")

    processed_path = "data/processed/processed.csv"

    assert os.path.exists(processed_path), "Processed file was not created."

    df = pd.read_csv(processed_path)
    assert not df.empty, "Processed dataset is empty."