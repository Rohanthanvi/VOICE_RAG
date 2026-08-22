from __future__ import annotations

import time
import random
import json
import statistics
from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

QUERY_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "msmarco_hi_train_5000.jsonl"
)


def percentile(values, p):

    values = sorted(values)

    if not values:
        return float("nan")

    index = min(
        int(len(values) * p / 100),
        len(values) - 1,
    )

    return values[index]


def main(n=30):

    from sentence_transformers import SentenceTransformer
    import chromadb

    print("=" * 80)
    print("DENSE LATENCY BREAKDOWN")
    print("=" * 80)

    print("\nLoading model...")

    model = SentenceTransformer(
        "google/embeddinggemma-300m",
        model_kwargs={
            "torch_dtype": "float16"
        },
    )

    print("Loading Chroma...")

    chroma_dir = (
        PROJECT_DIR
        / "chroma_db"
    )

    client = chromadb.PersistentClient(
        path=str(chroma_dir)
    )

    collection = client.get_collection(
        name="msmarco_hi_chunks"
    )

    with open(
        QUERY_FILE,
        encoding="utf-8",
    ) as f:

        rows = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    random.seed(7)

    rows = random.sample(
        rows,
        min(n, len(rows)),
    )

    queries = [
        row["query"]
        for row in rows
    ]

    # ---------------------------------------------------------
    # WARMUP
    # ---------------------------------------------------------

    print("\nWarming up...")

    embedding = model.encode_query(
        "साइरीन क्या है",
        normalize_embeddings=True,
    )

    collection.query(
        query_embeddings=[
            embedding.tolist()
        ],
        n_results=5,
        where={
            "strategy": "passage_level"
        },
    )

    # ---------------------------------------------------------
    # TIMINGS
    # ---------------------------------------------------------

    embedding_times = []
    chroma_times = []
    total_times = []

    print(
        f"\nRunning {len(queries)} queries..."
    )

    for i, query in enumerate(
        queries,
        1,
    ):

        # -----------------------------------------------------
        # EMBEDDING
        # -----------------------------------------------------

        start = time.perf_counter()

        embedding = model.encode_query(
            query,
            normalize_embeddings=True,
        )

        embedding_ms = (
            time.perf_counter()
            - start
        ) * 1000

        embedding_times.append(
            embedding_ms
        )

        # -----------------------------------------------------
        # CHROMA
        # -----------------------------------------------------

        start = time.perf_counter()

        collection.query(
            query_embeddings=[
                embedding.tolist()
            ],
            n_results=5,
            where={
                "strategy": "passage_level"
            },
        )

        chroma_ms = (
            time.perf_counter()
            - start
        ) * 1000

        chroma_times.append(
            chroma_ms
        )

        total_times.append(
            embedding_ms + chroma_ms
        )

        print(
            f"\rProgress: {i}/{len(queries)}",
            end="",
        )

    print("\n")

    # ---------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------

    def report(
        name,
        values,
    ):

        print(
            f"{name:<25}"
            f"P50={percentile(values,50):8.1f} ms   "
            f"P70={percentile(values,70):8.1f} ms   "
            f"P95={percentile(values,95):8.1f} ms   "
            f"P100={percentile(values,100):8.1f} ms   "
            f"Mean={statistics.mean(values):8.1f} ms"
        )

    print("=" * 80)
    print("DENSE BREAKDOWN")
    print("=" * 80)

    report(
        "Query embedding",
        embedding_times,
    )

    report(
        "Chroma search",
        chroma_times,
    )

    report(
        "Embedding + Chroma",
        total_times,
    )


if __name__ == "__main__":

    main(30)