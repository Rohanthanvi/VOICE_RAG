"""
Hacker House Goa — Controlled E5 Benchmark

Fair E5-small evaluation:
- Same 10,000 passage corpus used for indexing
- Only queries whose relevant passage exists in that corpus
- Measures Recall@1, Recall@3, Recall@5, MRR@5
- Also measures end-to-end retrieval latency

Does NOT modify any index.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from tqdm import tqdm

from retrieval_e5_controlled import retrieve


PROJECT_DIR = Path(__file__).resolve().parent.parent

EVAL_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "controlled_eval_10k.jsonl"
)


def load_eval_data():

    rows = []

    with open(
        EVAL_FILE,
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    __import__("json").loads(line)
                )

    return rows


def percentile(values, p):

    if not values:
        return float("nan")

    values = sorted(values)

    index = min(
        int(len(values) * p / 100),
        len(values) - 1,
    )

    return values[index]


def main():

    print("=" * 80)
    print("HACKER HOUSE GOA — CONTROLLED E5 BENCHMARK")
    print("=" * 80)

    # ========================================================
    # LOAD EVALUATION DATA
    # ========================================================

    print(
        "\nLoading controlled evaluation data..."
    )

    rows = load_eval_data()

    print(
        f"Loaded {len(rows):,} controlled queries."
    )

    if not rows:
        raise RuntimeError(
            "No controlled evaluation queries found."
        )

    # ========================================================
    # WARMUP
    # ========================================================

    print(
        "\nWarming up E5 model + Chroma..."
    )

    retrieve(
        "साइरीन क्या है",
        top_k=5,
    )

    print("Warmup complete.")

    # ========================================================
    # METRICS
    # ========================================================

    recall1_hits = 0
    recall3_hits = 0
    recall5_hits = 0

    reciprocal_ranks = []

    latency_ms = []

    # ========================================================
    # EVALUATION
    # ========================================================

    print(
        "\nEvaluating..."
    )

    for row in tqdm(
        rows,
        desc="Progress",
    ):

        query = row["query"]

        relevant = set(
            row["relevant_passage_ids"]
        )

        start = time.perf_counter()

        results = retrieve(
            query,
            top_k=5,
        )

        elapsed = (
            time.perf_counter()
            - start
        ) * 1000

        latency_ms.append(
            elapsed
        )

        retrieved_ids = [
            result["passage_id"]
            for result in results
        ]

        # ----------------------------------------------------
        # Recall@1
        # ----------------------------------------------------

        if any(
            passage_id in relevant
            for passage_id in retrieved_ids[:1]
        ):
            recall1_hits += 1

        # ----------------------------------------------------
        # Recall@3
        # ----------------------------------------------------

        if any(
            passage_id in relevant
            for passage_id in retrieved_ids[:3]
        ):
            recall3_hits += 1

        # ----------------------------------------------------
        # Recall@5
        # ----------------------------------------------------

        if any(
            passage_id in relevant
            for passage_id in retrieved_ids[:5]
        ):
            recall5_hits += 1

        # ----------------------------------------------------
        # MRR@5
        # ----------------------------------------------------

        rr = 0.0

        for rank, passage_id in enumerate(
            retrieved_ids[:5],
            start=1,
        ):

            if passage_id in relevant:

                rr = 1.0 / rank

                break

        reciprocal_ranks.append(rr)

    # ========================================================
    # CALCULATE RESULTS
    # ========================================================

    n = len(rows)

    recall1 = recall1_hits / n
    recall3 = recall3_hits / n
    recall5 = recall5_hits / n

    mrr5 = (
        sum(reciprocal_ranks)
        / n
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("CONTROLLED E5 RESULTS")
    print("=" * 80)

    print(
        f"\nEvaluation queries: {n:,}"
    )

    print(
        f"\nRecall@1 = {recall1:.4f}"
    )

    print(
        f"Recall@3 = {recall3:.4f}"
    )

    print(
        f"Recall@5 = {recall5:.4f}"
    )

    print(
        f"MRR@5    = {mrr5:.4f}"
    )

    # ========================================================
    # LATENCY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("E5 RETRIEVAL LATENCY")
    print("=" * 80)

    print(
        f"\nP50  = {percentile(latency_ms, 50):.1f} ms"
    )

    print(
        f"P70  = {percentile(latency_ms, 70):.1f} ms"
    )

    print(
        f"P95  = {percentile(latency_ms, 95):.1f} ms"
    )

    print(
        f"P100 = {percentile(latency_ms, 100):.1f} ms"
    )

    print(
        f"Mean = {statistics.mean(latency_ms):.1f} ms"
    )

    # ========================================================
    # BASELINE
    # ========================================================

    print("\n")
    print("=" * 80)
    print("REFERENCE — ORIGINAL EMBEDDINGGEMMA")
    print("=" * 80)

    print(
        "\nOriginal full-corpus passage_level:"
    )

    print(
        "Recall@1 = 0.3110"
    )

    print(
        "Recall@3 = 0.3616"
    )

    print(
        "Recall@5 = 0.3794"
    )

    print(
        "MRR@5    = 0.3378"
    )

    print(
        "\nOriginal dense latency:"
    )

    print(
        "Embedding P50 ≈ 172.9 ms"
    )

    print(
        "Chroma P50    ≈ 34.4 ms"
    )

    print(
        "Total P50     ≈ 207.0 ms"
    )

    # ========================================================
    # DECISION
    # ========================================================

    print("\n")
    print("=" * 80)
    print("DECISION")
    print("=" * 80)

    if recall5 >= 0.30:
        print(
            "\nE5 quality looks promising."
        )
        print(
            "Proceed to the full E5 index."
        )
    else:
        print(
            "\nE5 quality is weak on the controlled test."
        )
        print(
            "Do NOT build the full 336K E5 index yet."
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()