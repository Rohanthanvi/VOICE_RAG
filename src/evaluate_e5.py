"""
Hacker House Goa — E5-small retrieval benchmark

Evaluates the 10K E5 test index.

IMPORTANT:
    This does NOT modify either Chroma index.

Run:
    python src/evaluate_e5.py
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from tqdm import tqdm

from retrieval_e5 import retrieve


PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "msmarco_hi_train_5000.jsonl"
)

PASSAGE_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "passage_pool_hi.jsonl"
)

N_QUERIES = 5000

RANDOM_SEED = 42


def load_queries():

    rows = []

    with open(
        RAW_FILE,
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    json.loads(line)
                )

    return rows


def load_relevance():

    relevance = {}

    with open(
        PASSAGE_FILE,
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            passage_id = row["passage_id"]

            for query_id in row.get(
                "seen_with_query_ids",
                [],
            ):

                relevance.setdefault(
                    int(query_id),
                    set(),
                ).add(
                    passage_id
                )

    return relevance


def get_query_id(row):

    for key in (
        "query_id",
        "id",
        "qid",
    ):

        if key in row:
            return int(row[key])

    raise KeyError(
        f"Could not find query id in row: {row.keys()}"
    )


def get_query_text(row):

    for key in (
        "query",
        "text",
    ):

        if key in row:
            return row[key]

    raise KeyError(
        f"Could not find query text in row: {row.keys()}"
    )


def evaluate():

    print("=" * 80)
    print("HACKER HOUSE GOA — E5-SMALL BENCHMARK")
    print("=" * 80)

    print("\nLoading queries...")

    queries = load_queries()

    print(
        f"Loaded {len(queries):,} queries."
    )

    print("\nLoading relevance information...")

    relevance = load_relevance()

    print(
        f"Loaded relevance for "
        f"{len(relevance):,} queries."
    )

    # ---------------------------------------------------------
    # Select queries
    # ---------------------------------------------------------

    random.seed(RANDOM_SEED)

    if N_QUERIES >= len(queries):

        evaluation_queries = queries

    else:

        evaluation_queries = random.sample(
            queries,
            N_QUERIES,
        )

    print(
        f"\nFinal evaluation queries: "
        f"{len(evaluation_queries):,}"
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    recall1 = 0
    recall3 = 0
    recall5 = 0

    reciprocal_ranks = []

    evaluated = 0
    skipped = 0

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    print("\nEvaluating E5-small...")

    for row in tqdm(
        evaluation_queries,
        desc="Progress",
    ):

        try:

            query_id = get_query_id(row)

            query_text = get_query_text(row)

        except KeyError:

            skipped += 1

            continue

        relevant = relevance.get(
            query_id,
            set(),
        )

        if not relevant:

            skipped += 1

            continue

        results = retrieve(
            query_text,
            top_k=5,
        )

        retrieved_ids = [
            r["metadata"].get(
                "passage_id"
            )
            for r in results
        ]

        evaluated += 1

        # -----------------------------------------------------
        # Recall@1
        # -----------------------------------------------------

        if any(
            pid in relevant
            for pid in retrieved_ids[:1]
        ):

            recall1 += 1

        # -----------------------------------------------------
        # Recall@3
        # -----------------------------------------------------

        if any(
            pid in relevant
            for pid in retrieved_ids[:3]
        ):

            recall3 += 1

        # -----------------------------------------------------
        # Recall@5
        # -----------------------------------------------------

        if any(
            pid in relevant
            for pid in retrieved_ids[:5]
        ):

            recall5 += 1

        # -----------------------------------------------------
        # MRR@5
        # -----------------------------------------------------

        rr = 0.0

        for rank, pid in enumerate(
            retrieved_ids[:5],
            start=1,
        ):

            if pid in relevant:

                rr = 1.0 / rank

                break

        reciprocal_ranks.append(rr)

    # ---------------------------------------------------------
    # Final metrics
    # ---------------------------------------------------------

    if evaluated == 0:

        raise RuntimeError(
            "No queries could be evaluated."
        )

    r1 = recall1 / evaluated
    r3 = recall3 / evaluated
    r5 = recall5 / evaluated

    mrr5 = (
        sum(reciprocal_ranks)
        / evaluated
    )

    print("\n")
    print("=" * 80)
    print("E5-SMALL RESULTS")
    print("=" * 80)

    print(
        f"\nEvaluated: {evaluated:,}"
    )

    print(
        f"Skipped:   {skipped:,}"
    )

    print("\n")

    print(
        f"Recall@1 = {r1:.4f}"
    )

    print(
        f"Recall@3 = {r3:.4f}"
    )

    print(
        f"Recall@5 = {r5:.4f}"
    )

    print(
        f"MRR@5    = {mrr5:.4f}"
    )

    print("\n" + "=" * 80)
    print("CURRENT BASELINE")
    print("=" * 80)

    print(
        "\nEmbeddingGemma passage_level:"
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

    print("\n" + "=" * 80)
    print("E5 VS EMBEDDINGGEMMA")
    print("=" * 80)

    print(
        f"\nRecall@1: "
        f"{0.3110:.4f} → {r1:.4f} "
        f"({r1 - 0.3110:+.4f})"
    )

    print(
        f"Recall@3: "
        f"{0.3616:.4f} → {r3:.4f} "
        f"({r3 - 0.3616:+.4f})"
    )

    print(
        f"Recall@5: "
        f"{0.3794:.4f} → {r5:.4f} "
        f"({r5 - 0.3794:+.4f})"
    )

    print(
        f"MRR@5:    "
        f"{0.3378:.4f} → {mrr5:.4f} "
        f"({mrr5 - 0.3378:+.4f})"
    )

    print("\n" + "=" * 80)

    if r5 >= 0.35:

        print(
            "E5 LOOKS PROMISING — proceed to larger index."
        )

    else:

        print(
            "E5 QUALITY IS TOO LOW — optimize before full indexing."
        )


if __name__ == "__main__":
    evaluate()