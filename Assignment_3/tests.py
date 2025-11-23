import pytest
import pandas as pd
import pandera.pandas as pa
from dataset_schema import aggregated_news_schema, vectorized_news_skipgram_schema, vectorized_news_cbow_schema
import os

# Define the directory where your datasets are stored
DATASET_DIR = "datasets"

def load_and_validate_csv(filepath, schema):
    df = pd.read_csv(filepath)
    schema.validate(df, lazy=True)
    return df

def test_aggregated_news_schema_validation():
    filename = "aggregated_news.csv"
    filepath = os.path.join(DATASET_DIR, filename)
    print(f"Validating {filepath} against schema: aggregated_news_schema")

    try:
        df = load_and_validate_csv(filepath, aggregated_news_schema)
    except FileNotFoundError:
        pytest.fail(f"File not found: {filepath}")

    try:
        aggregated_news_schema.validate(df, lazy=True)
        print(f"Validation successful for {filepath}")
    except pa.errors.SchemaErrors as err:
        error_message = f"Schema validation failed for {filepath}:\n"
        pytest.fail(error_message)


def test_vectorized_news_skipgram_embeddings():
    filename = "vectorized_news_skipgram_embeddings.csv"
    filepath = os.path.join(DATASET_DIR, filename)
    print(f"Validating {filepath} against schema: vectorized_news_skipgram_schema")

    try:
        df = load_and_validate_csv(filepath, vectorized_news_skipgram_schema)
    except FileNotFoundError:
        pytest.fail(f"File not found: {filepath}")

    try:
        vectorized_news_skipgram_schema.validate(df, lazy=True)
        print(f"Validation successful for {filepath}")
    except pa.errors.SchemaErrors as err:
        error_message = f"Schema validation failed for {filepath}:\n"
        pytest.fail(error_message)


def test_vectorized_news_cbow_embeddings():
    filename = "vectorized_news_cbow_embeddings.csv"
    filepath = os.path.join(DATASET_DIR, filename)
    print(f"Validating {filepath} against schema: vectorized_news_cbow_schema")

    try:
        df = load_and_validate_csv(filepath, vectorized_news_cbow_schema)
    except FileNotFoundError:
        pytest.fail(f"File not found: {filepath}")

    try:
        vectorized_news_cbow_schema.validate(df, lazy=True)
        print(f"Validation successful for {filepath}")
    except pa.errors.SchemaErrors as err:
        error_message = f"Schema validation failed for {filepath}:\n"
        pytest.fail(error_message)

