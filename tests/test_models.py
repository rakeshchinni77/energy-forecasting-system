import torch
import pandas as pd
import numpy as np

from src.models import LSTMForecaster, SequenceDataset


def test_lstm_forward_pass_shape():
    """
    Verifies:
    1. LSTM accepts (batch, seq_len, features)
    2. Output shape is (batch,)
    """
    batch_size = 4
    seq_len = 24
    num_features = 10

    model = LSTMForecaster(
        input_size=num_features,
        hidden_size=32,
        num_layers=2,
        dropout=0.2,
    )

    x = torch.randn(batch_size, seq_len, num_features)
    output = model(x)

    assert output.shape == (batch_size,), "Incorrect LSTM output shape"


def test_sequence_dataset_structure():
    """
    Verifies:
    1. Dataset length calculation
    2. Sample input shape
    3. Target is scalar float
    """
    df = pd.DataFrame(
        np.random.rand(100, 5),
        columns=["a", "b", "c", "d", "target"],
    )

    window_size = 24
    dataset = SequenceDataset(df, target_col="target", window_size=window_size)

    # Length check
    assert len(dataset) == 100 - window_size, "SequenceDataset length is incorrect"

    # Sample structure check
    x, y = dataset[0]

    assert x.shape == (window_size, 5), "Input sequence shape is incorrect"
    assert isinstance(y.item(), float), "Target should be a scalar float"