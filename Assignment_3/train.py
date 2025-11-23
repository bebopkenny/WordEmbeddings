import os
import ast
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from model import MLP


def _load_vectorized_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a vectorized dataset with columns:
    date, symbol, news_vector, impact_score
    and return (X, y) where:
      - X is a 2D float32 array of document embeddings
      - y is a 1D float32 array of binary labels (impact > 0)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Vectorized dataset not found: {csv_path}")

    # Parse news_vector as Python list using ast.literal_eval
    df = pd.read_csv(csv_path, converters={"news_vector": ast.literal_eval})

    for col in ["news_vector", "impact_score"]:
        if col not in df.columns:
            raise ValueError(f"Expected '{col}' column in {csv_path}")

    X = np.array(df["news_vector"].tolist(), dtype=np.float32)

    impact_scores = df["impact_score"].astype(float).values
    # Binary label: positive impact vs non-positive
    y = (impact_scores > 0).astype(np.float32)

    return X, y


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str | None = None,
):
    """
    Train an MLP on the provided embeddings and labels.
    Returns: (trained_model, accuracy, classification_report_str)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train/test split; stratify only if we have at least 2 unique classes
    unique_labels = np.unique(y)
    stratify = y if len(unique_labels) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test).unsqueeze(1)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MLP(input_dim=X.shape[1], num_classes=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(train_dataset)
        # Uncomment for debug:
        # print(f"Epoch {epoch+1}/{num_epochs} - loss: {epoch_loss:.4f}")

    # Evaluation
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            all_logits.append(logits.cpu())
            all_labels.append(batch_y)

    logits = torch.cat(all_logits, dim=0).squeeze(1)
    y_true = torch.cat(all_labels, dim=0).squeeze(1).numpy()
    y_prob = torch.sigmoid(logits).numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    return model, acc, report


def train_on_embeddings(
    csv_path: str,
    model_path: str,
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> tuple[float, str]:
    """
    Utility to train on a given vectorized CSV and save the model checkpoint.
    Returns (accuracy, classification_report_str).
    """
    X, y = _load_vectorized_dataset(csv_path)
    model, acc, report = train_mlp(
        X,
        y,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": X.shape[1],
        },
        model_path,
    )

    return acc, report


def train_skipgram(
    csv_path: str = os.path.join("datasets", "vectorized_news_skipgram_embeddings.csv"),
    model_path: str = "skipgram.pth",
    **kwargs,
) -> tuple[float, str]:
    """
    Train MLP using Skip-gram embeddings and save to skipgram.pth
    """
    return train_on_embeddings(csv_path, model_path, **kwargs)


def train_cbow(
    csv_path: str = os.path.join("datasets", "vectorized_news_cbow_embeddings.csv"),
    model_path: str = "cbow.pth",
    **kwargs,
) -> tuple[float, str]:
    """
    Train MLP using CBOW embeddings and save to cbow.pth
    """
    return train_on_embeddings(csv_path, model_path, **kwargs)
