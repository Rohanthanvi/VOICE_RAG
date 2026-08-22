"""
Hacker House Goa — Retrieval Latency Benchmark

Measures WARM query latency for:

1. Dense retrieval
2. Hybrid BM25 + RRF
3. BGE reranker
4. Hybrid + Reranker

Model loading and BM25 construction are warmed up BEFORE
the actual measurements.

Run:
    python src/benchmark_retrieval_latency.py --n 30

After validation:
    python src/benchmark_retrieval_latency.py --n 100
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

RAW_QUERY_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "msmarco_hi_train_5000.jsonl"
)

REPORT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "retrieval_latency_report.json"
)

RANDOM_SEED = 7


# ============================================================
# PERCENTILE
# ============================================================

def percentile(
    values: list[float],
    pct: float,
) -> float:

    if not values:
        return float("nan")

    values = sorted(values)

    index = int(
        len(values) * pct / 100
    )

    index = min(
        index,
        len(values) - 1,
    )

    return values[index]


# ============================================================
# LOAD QUERIES
# ============================================================

def load_queries(
    n: int,
) -> list[str]:

    with open(
        RAW_QUERY_FILE,
        encoding="utf-8",
    ) as f:

        rows = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    random.seed(RANDOM_SEED)

    sample = random.sample(
        rows,
        min(n, len(rows)),
    )

    return [
        row["query"]
        for row in sample
    ]


# ============================================================
# REPORT
# ============================================================

def report(
    name: str,
    values: list[float],
):

    if not values:
        print(
            f"{name:<30} NO DATA"
        )
        return

    print(
        f"{name:<30}"
        f"P50={percentile(values, 50):>8.1f} ms   "
        f"P70={percentile(values, 70):>8.1f} ms   "
        f"P95={percentile(values, 95):>8.1f} ms   "
        f"P100={percentile(values, 100):>8.1f} ms   "
        f"Mean={statistics.mean(values):>8.1f} ms"
    )


# ============================================================
# MAIN
# ============================================================

def main(n: int):

    print("=" * 90)
    print(
        "HACKER HOUSE GOA — RETRIEVAL LATENCY BENCHMARK"
    )
    print("=" * 90)

    # ========================================================
    # IMPORT COMPONENTS
    # ========================================================

    print(
        "\nLoading retrieval components..."
    )

    from retrieval import (
        retrieve,
        warmup as warmup_dense,
    )

    from hybrid_retrieval import (
        hybrid_retrieve,
    )

    from reranked_retrieval import (
        reranked_retrieve,
    )

    from reranker import (
        rerank,
    )

    # ========================================================
    # WARMUP
    # ========================================================

    print(
        "\n[1/3] Warming up dense retrieval..."
    )

    warmup_dense()

    print(
        "[2/3] Warming up hybrid retrieval..."
    )

    # hybrid_retrieve() itself loads/builds BM25
    # on the first call.
    hybrid_retrieve(
        "साइरीन क्या है",
        top_k=10,
    )

    print(
        "[3/3] Warming up reranker..."
    )

    reranked_retrieve(
        "साइरीन क्या है",
        top_k=5,
    )

    print(
        "\nAll components warmed up."
    )

    # ========================================================
    # LOAD QUERIES
    # ========================================================

    queries = load_queries(n)

    print(
        f"\nBenchmark queries: {len(queries)}"
    )

    # ========================================================
    # STORAGE
    # ========================================================

    dense_times = []
    hybrid_times = []
    reranker_times = []
    full_times = []

    # ========================================================
    # BENCHMARK
    # ========================================================

    print(
        "\nRunning warm-query benchmark..."
    )

    for index, query in enumerate(
        queries,
        start=1,
    ):

        # ----------------------------------------------------
        # DENSE
        # ----------------------------------------------------

        start = time.perf_counter()

        retrieve(
            query,
            top_k=5,
            strategy_filter="passage_level",
        )

        dense_ms = (
            time.perf_counter()
            - start
        ) * 1000

        dense_times.append(
            dense_ms
        )

        # ----------------------------------------------------
        # HYBRID
        # ----------------------------------------------------

        start = time.perf_counter()

        candidates = hybrid_retrieve(
            query,
            top_k=10,
        )

        hybrid_ms = (
            time.perf_counter()
            - start
        ) * 1000

        hybrid_times.append(
            hybrid_ms
        )

        # ----------------------------------------------------
        # RERANKER ONLY
        # ----------------------------------------------------

        start = time.perf_counter()

        rerank(
            query,
            candidates,
            top_k=5,
            batch_size=16,
        )

        reranker_ms = (
            time.perf_counter()
            - start
        ) * 1000

        reranker_times.append(
            reranker_ms
        )

        # ----------------------------------------------------
        # FULL HYBRID + RERANKER
        # ----------------------------------------------------

        start = time.perf_counter()

        reranked_retrieve(
            query,
            top_k=5,
        )

        full_ms = (
            time.perf_counter()
            - start
        ) * 1000

        full_times.append(
            full_ms
        )

        if (
            index % 5 == 0
            or index == len(queries)
        ):

            print(
                f"\rProgress: "
                f"{index}/{len(queries)}",
                end="",
                flush=True,
            )

    print("\n")

    # ========================================================
    # RESULTS
    # ========================================================

    print("=" * 90)
    print("LATENCY RESULTS")
    print("=" * 90)

    print()

    report(
        "Dense retrieval",
        dense_times,
    )

    report(
        "Hybrid BM25 + RRF",
        hybrid_times,
    )

    report(
        "BGE reranker only",
        reranker_times,
    )

    report(
        "Hybrid + Reranker",
        full_times,
    )

    # ========================================================
    # 200 MS TARGET
    # ========================================================

    print()
    print("=" * 90)
    print("200 ms RETRIEVAL TARGET")
    print("=" * 90)

    p50 = percentile(
        full_times,
        50,
    )

    p70 = percentile(
        full_times,
        70,
    )

    p95 = percentile(
        full_times,
        95,
    )

    p100 = percentile(
        full_times,
        100,
    )

    print(
        f"\nP50  : {p50:.1f} ms"
    )

    print(
        f"P70  : {p70:.1f} ms"
    )

    print(
        f"P95  : {p95:.1f} ms"
    )

    print(
        f"P100 : {p100:.1f} ms"
    )

    if p70 <= 200:

        print(
            "\n✅ Retrieval P70 is within 200 ms."
        )

    else:

        print(
            "\n⚠️ Retrieval P70 exceeds 200 ms."
        )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_data = {
        "n_queries": len(queries),

        "dense_ms": {
            "p50": percentile(
                dense_times,
                50,
            ),
            "p70": percentile(
                dense_times,
                70,
            ),
            "p95": percentile(
                dense_times,
                95,
            ),
            "p100": percentile(
                dense_times,
                100,
            ),
        },

        "hybrid_ms": {
            "p50": percentile(
                hybrid_times,
                50,
            ),
            "p70": percentile(
                hybrid_times,
                70,
            ),
            "p95": percentile(
                hybrid_times,
                95,
            ),
            "p100": percentile(
                hybrid_times,
                100,
            ),
        },

        "reranker_ms": {
            "p50": percentile(
                reranker_times,
                50,
            ),
            "p70": percentile(
                reranker_times,
                70,
            ),
            "p95": percentile(
                reranker_times,
                95,
            ),
            "p100": percentile(
                reranker_times,
                100,
            ),
        },

        "hybrid_reranker_ms": {
            "p50": p50,
            "p70": p70,
            "p95": p95,
            "p100": p100,
        },
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report_data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nReport saved to:\n{REPORT_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=int,
        default=30,
        help="number of benchmark queries",
    )

    args = parser.parse_args()

    main(args.n)