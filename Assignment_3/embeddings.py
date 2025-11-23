import os
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

import torch
import torch.nn as nn

# Shared hyperparameters (keep consistent with MLP input size)
EMBEDDING_DIM = 100
SKIPGRAM_WINDOW = 5
CBOW_WINDOW = 2
MIN_COUNT = 2


def preprocess(text: str) -> List[str]:
    """
    Tokenize text and remove stopwords using gensim helpers.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    tokens = [t for t in simple_preprocess(text) if t not in STOPWORDS]
    return tokens


def _load_aggregated_news(aggregated_path: str) -> pd.DataFrame:
    """
    Load aggregated_news.csv and attach a 'tokens' column.
    Expected columns: date, symbol, news
    """
    if not os.path.exists(aggregated_path):
        raise FileNotFoundError(f"aggregated_news.csv not found at {aggregated_path}")

    df = pd.read_csv(aggregated_path)

    required_cols = {"date", "symbol", "news"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"aggregated_news.csv is missing columns: {missing}")

    df["tokens"] = df["news"].apply(preprocess)
    # Drop completely empty documents
    df = df[df["tokens"].map(len) > 0].reset_index(drop=True)
    return df


def _document_vector(tokens: List[str], model, vector_size: int) -> np.ndarray:
    """
    Average word embeddings for a list of tokens.
    If no token is in the vocabulary, return a zero vector.
    """
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return np.zeros(vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)


# ---------------------------------------------------------------------
# 1.1 Skip-gram embedding using gensim Word2Vec
# ---------------------------------------------------------------------
def train_skipgram_embedding(
    aggregated_path: str,
    vector_size: int = EMBEDDING_DIM,
    window: int = SKIPGRAM_WINDOW,
    min_count: int = MIN_COUNT,
) -> Tuple[Word2Vec, pd.DataFrame]:
    """
    Train a Skip-gram Word2Vec model using gensim.
    """
    df = _load_aggregated_news(aggregated_path)
    tokenized_docs = df["tokens"].tolist()

    model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,  # 1 = Skip-gram
        workers=os.cpu_count() or 1,
    )

    return model, df


def create_skipgram_dataset(
    aggregated_path: str = os.path.join("datasets", "aggregated_news.csv"),
    out_path: str = os.path.join("datasets", "vectorized_news_skipgram_embeddings.csv"),
    vector_size: int = EMBEDDING_DIM,
    window: int = SKIPGRAM_WINDOW,
    min_count: int = MIN_COUNT,
) -> pd.DataFrame:
    """
    Create vectorized_news_skipgram_embeddings.csv with schema:
    (date, symbol, news_vector, impact_score)
    """
    model, df = train_skipgram_embedding(
        aggregated_path, vector_size=vector_size, window=window, min_count=min_count
    )

    vectors = df["tokens"].apply(
        lambda toks: _document_vector(toks, model, vector_size)
    )

    # If impact_score already exists from Assignment 2, keep it.
    # Otherwise, create a dummy 0.0 column so schema validation still passes.
    if "impact_score" not in df.columns:
        print(
            "[Warning] 'impact_score' column not found in aggregated_news.csv. "
            "Defaulting impact_score to 0.0. You should replace this with your "
            "real labels from Assignment #2."
        )
        df["impact_score"] = 0.0

    out_df = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]),
            "symbol": df["symbol"].astype(str),
            # store as Python list so type=object; CSV will hold string representation
            "news_vector": vectors.apply(lambda v: list(map(float, v.tolist()))),
            "impact_score": df["impact_score"].astype(float),
        }
    )

    out_df.to_csv(out_path, index=False)
    return out_df


# ---------------------------------------------------------------------
# 1.2 CBOW implementation from scratch (bonus)
# ---------------------------------------------------------------------
class CBOWModel(nn.Module):
    """
    Simple CBOW implementation:
    - Input: indices of context words
    - Embedding: learned input embeddings
    - Output: logits over vocabulary for the target word
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_indices: torch.Tensor) -> torch.Tensor:
        # context_indices: (context_len,)
        embeds = self.in_embeddings(context_indices)  # (context_len, embedding_dim)
        hidden = embeds.mean(dim=0, keepdim=True)  # (1, embedding_dim)
        logits = self.out_layer(hidden)  # (1, vocab_size)
        return logits


def _build_vocab(tokenized_docs: List[List[str]], min_count: int):
    """
    Build vocabulary from tokens with a minimum frequency.
    """
    counter = Counter()
    for doc in tokenized_docs:
        counter.update(doc)

    vocab = [w for w, c in counter.items() if c >= min_count]
    vocab.sort()
    if not vocab:
        raise ValueError("Empty vocabulary; try lowering min_count.")

    word_to_idx = {w: i for i, w in enumerate(vocab)}
    return word_to_idx, vocab


def train_cbow_embedding(
    aggregated_path: str,
    embedding_dim: int = EMBEDDING_DIM,
    window_size: int = CBOW_WINDOW,
    min_count: int = MIN_COUNT,
    epochs: int = 5,
    learning_rate: float = 0.01,
) -> Tuple[CBOWModel, dict, pd.DataFrame]:
    """
    Train a CBOW embedding from scratch using PyTorch.
    """
    df = _load_aggregated_news(aggregated_path)
    tokenized_docs = df["tokens"].tolist()

    word_to_idx, vocab = _build_vocab(tokenized_docs, min_count)
    vocab_size = len(vocab)

    model = CBOWModel(vocab_size, embedding_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for tokens in tokenized_docs:
            idxs = [word_to_idx[w] for w in tokens if w in word_to_idx]
            for center_pos in range(len(idxs)):
                # Build context window around the target word
                context_indices = []
                for offset in range(-window_size, window_size + 1):
                    if offset == 0:
                        continue
                    ctx_pos = center_pos + offset
                    if 0 <= ctx_pos < len(idxs):
                        context_indices.append(idxs[ctx_pos])
                if not context_indices:
                    continue

                context_tensor = torch.tensor(context_indices, dtype=torch.long)
                target_tensor = torch.tensor([idxs[center_pos]], dtype=torch.long)

                optimizer.zero_grad()
                logits = model(context_tensor)
                loss = loss_fn(logits, target_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Uncomment if you want to see training progress
        # print(f"CBOW epoch {epoch+1}/{epochs}, loss={total_loss:.4f}")

    return model, word_to_idx, df


def create_cbow_dataset(
    aggregated_path: str = os.path.join("datasets", "aggregated_news.csv"),
    out_path: str = os.path.join("datasets", "vectorized_news_cbow_embeddings.csv"),
    embedding_dim: int = EMBEDDING_DIM,
    window_size: int = CBOW_WINDOW,
    min_count: int = MIN_COUNT,
    epochs: int = 5,
) -> pd.DataFrame:
    """
    Create vectorized_news_cbow_embeddings.csv with schema:
    (date, symbol, news_vector, impact_score)
    """
    model, word_to_idx, df = train_cbow_embedding(
        aggregated_path=aggregated_path,
        embedding_dim=embedding_dim,
        window_size=window_size,
        min_count=min_count,
        epochs=epochs,
    )
    tokenized_docs = df["tokens"].tolist()

    def doc_vector(tokens: List[str]) -> np.ndarray:
        idxs = [word_to_idx[w] for w in tokens if w in word_to_idx]
        if not idxs:
            return np.zeros(embedding_dim, dtype=np.float32)
        with torch.no_grad():
            idx_tensor = torch.tensor(idxs, dtype=torch.long)
            embeds = model.in_embeddings(idx_tensor)
            vec = embeds.mean(dim=0).numpy()
        return vec.astype(np.float32)

    vectors = [doc_vector(tokens) for tokens in tokenized_docs]

    if "impact_score" not in df.columns:
        print(
            "[Warning] 'impact_score' column not found in aggregated_news.csv. "
            "Defaulting impact_score to 0.0. You should replace this with your "
            "real labels from Assignment #2."
        )
        df["impact_score"] = 0.0

    out_df = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]),
            "symbol": df["symbol"].astype(str),
            "news_vector": [list(map(float, v.tolist())) for v in vectors],
            "impact_score": df["impact_score"].astype(float),
        }
    )

    out_df.to_csv(out_path, index=False)
    return out_df
