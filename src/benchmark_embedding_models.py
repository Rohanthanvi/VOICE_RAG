"""
Hacker House Goa
Embedding Model Latency Comparison

Purpose:
    Compare EmbeddingGemma-300M against a smaller multilingual
    embedding model BEFORE rebuilding the Chroma index.

IMPORTANT:
    This benchmark does NOT modify ChromaDB.

Run:
    python src/benchmark_embedding_models.py --n 30
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent.parent

QUERY_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "msmarco_hi_train_5000.jsonl"
)

RANDOM_SEED = 7


MODELS = {
    "EmbeddingGemma-300M": "google/embeddinggemma-300m",
    "Multilingual-E5-small": "intfloat/multilingual-e5-small",
}


def percentile(values: list[float], p: float) -> float:

    if not values:
        return float("nan")

    values = sorted(values)

    index = min(
        int(len(values) * p / 100),
        len(values) - 1,
    )

    return values[index]


def load_queries(n: int) -> list[str]:

    with open(
        QUERY_FILE,
        encoding="utf-8",
    ) as f:

        rows = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    random.seed(RANDOM_SEED)

    rows = random.sample(
        rows,
        min(n, len(rows)),
    )

    return [
        row["query"]
        for row in rows
    ]


def benchmark_model(
    model_name: str,
    model_path: str,
    queries: list[str],
):

    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print(f"PATH : {model_path}")
    print("=" * 80)

    print("\nLoading model...")

    model = SentenceTransformer(
        model_path,
        model_kwargs={
            "torch_dtype": "float16"
        },
    )

    print("Model loaded.")

    # ---------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------

    print("Warming up...")

    if "e5" in model_path.lower():

        model.encode(
            ["query: साइरीन क्या है"],
            normalize_embeddings=True,
        )

    else:

        model.encode_query(
            "साइरीन क्या है",
            normalize_embeddings=True,
        )

    # ---------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------

    times = []

    print(
        f"Running {len(queries)} queries..."
    )

    for i, query in enumerate(
        queries,
        1,
    ):

        start = time.perf_counter()

        if "e5" in model_path.lower():

            # E5 retrieval convention:
            # queries use "query:"
            model.encode(
                [f"query: {query}"],
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        else:

            model.encode_query(
                query,
                normalize_embeddings=True,
            )

        elapsed = (
            time.perf_counter()
            - start
        ) * 1000

        times.append(elapsed)

        print(
            f"\rProgress: {i}/{len(queries)}",
            end="",
            flush=True,
        )

    print()

    print("\nResults:")

    print(
        f"P50  = {percentile(times, 50):.1f} ms"
    )

    print(
        f"P70  = {percentile(times, 70):.1f} ms"
    )

    print(
        f"P95  = {percentile(times, 95):.1f} ms"
    )

    print(
        f"P100 = {percentile(times, 100):.1f} ms"
    )

    print(
        f"Mean = {statistics.mean(times):.1f} ms"
    )

    return {
        "model": model_name,
        "path": model_path,
        "p50_ms": percentile(times, 50),
        "p70_ms": percentile(times, 70),
        "p95_ms": percentile(times, 95),
        "p100_ms": percentile(times, 100),
        "mean_ms": statistics.mean(times),
    }


def main(n: int):

    print("=" * 80)
    print("HACKER HOUSE GOA — EMBEDDING MODEL BENCHMARK")
    print("=" * 80)

    print(
        "\nIMPORTANT:"
        "\nThis test only measures query embedding."
        "\nChromaDB is NOT modified."
        "\nNo existing index is changed."
    )

    queries = load_queries(n)

    print(
        f"\nLoaded {len(queries)} queries."
    )

    results = []

    for model_name, model_path in MODELS.items():

        result = benchmark_model(
            model_name,
            model_path,
            queries,
        )

        results.append(result)

    # ---------------------------------------------------------
    # Final comparison
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(
        f"{'Model':<30}"
        f"{'P50':>10}"
        f"{'P70':>10}"
        f"{'P95':>10}"
        f"{'P100':>10}"
    )

    print("-" * 80)

    for result in results:

        print(
            f"{result['model']:<30}"
            f"{result['p50_ms']:>9.1f} "
            f"{result['p70_ms']:>9.1f} "
            f"{result['p95_ms']:>9.1f} "
            f"{result['p100_ms']:>9.1f}"
        )

    # ---------------------------------------------------------
    # Recommendation
    # ---------------------------------------------------------

    fastest = min(
        results,
        key=lambda x: x["p50_ms"],
    )

    print("\n" + "=" * 80)
    print("FASTEST MODEL")
    print("=" * 80)

    print(
        f"\n{fastest['model']}"
    )

    print(
        f"P50 query embedding: "
        f"{fastest['p50_ms']:.1f} ms"
    )

    print(
        f"P70 query embedding: "
        f"{fastest['p70_ms']:.1f} ms"
    )

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------

    output = (
        PROJECT_DIR
        / "data"
        / "processed"
        / "embedding_model_latency.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "n_queries": n,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nReport saved to:\n{output}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    main(args.n)