import os

from embeddings import create_skipgram_dataset, create_cbow_dataset
from train import train_skipgram, train_cbow

DATASET_DIR = "datasets"


def main():
    aggregated_path = os.path.join(DATASET_DIR, "aggregated_news.csv")
    skipgram_path = os.path.join(
        DATASET_DIR, "vectorized_news_skipgram_embeddings.csv"
    )
    cbow_path = os.path.join(DATASET_DIR, "vectorized_news_cbow_embeddings.csv")

    # 1) Generate datasets (Skip-gram + CBOW)
    print("=== Generating Skip-gram embeddings and dataset ===")
    create_skipgram_dataset(aggregated_path=aggregated_path, out_path=skipgram_path)

    print("=== Generating CBOW embeddings and dataset ===")
    create_cbow_dataset(aggregated_path=aggregated_path, out_path=cbow_path)

    # 2) Train and evaluate on Skip-gram embeddings
    print("=== Training MLP on Skip-gram embeddings ===")
    skip_acc, skip_report = train_skipgram(
        csv_path=skipgram_path,
        model_path="skipgram.pth",
    )
    print(f"Skip-gram accuracy: {skip_acc:.4f}")
    print("Skip-gram classification report:")
    print(skip_report)

    # 3) Train and evaluate on CBOW embeddings
    print("=== Training MLP on CBOW embeddings ===")
    cbow_acc, cbow_report = train_cbow(
        csv_path=cbow_path,
        model_path="cbow.pth",
    )
    print(f"CBOW accuracy: {cbow_acc:.4f}")
    print("CBOW classification report:")
    print(cbow_report)


if __name__ == "__main__":
    main()
