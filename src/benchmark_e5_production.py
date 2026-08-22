from __future__ import annotations

import statistics
import sys
import time

from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

from retrieval_e5 import warmup, retrieve_fast


QUERIES = [
    "साइरीन क्या है",
    "भारत की राजधानी क्या है",
    "महात्मा गांधी कौन थे",
    "जल चक्र क्या है",
    "पृथ्वी सूर्य के चारों ओर क्यों घूमती है",
    "कंप्यूटर क्या है",
    "कृत्रिम बुद्धिमत्ता क्या है",
    "रामायण क्या है",
    "हिमालय कहाँ स्थित है",
    "गंगा नदी कहाँ से निकलती है",
]


def percentile(values, pct):
    values = sorted(values)

    if not values:
        return 0.0

    index = min(
        int(len(values) * pct / 100),
        len(values) - 1,
    )

    return values[index]


def main():

    print("=" * 80)
    print("HACKER HOUSE GOA — PRODUCTION E5 LATENCY")
    print("=" * 80)

    print("\nLoading production components...")

    t0 = time.perf_counter()

    warmup()

    startup_ms = (
        time.perf_counter() - t0
    ) * 1000

    print(
        f"\nStartup / warmup: "
        f"{startup_ms:.1f} ms"
    )

    print(
        "\nRunning warm-query benchmark..."
    )

    latencies = []

    for i, query in enumerate(
        QUERIES,
        start=1,
    ):

        t0 = time.perf_counter()

        result = retrieve_fast(
            query,
            top_k=8,
        )

        elapsed = (
            time.perf_counter() - t0
        ) * 1000

        latencies.append(elapsed)

        best_distance = result[
            "best_distance"
        ]

        confidence = result[
            "confidence"
        ]

        fast_path = result[
            "fast_path"
        ]

        print(
            f"{i:02d}. "
            f"{elapsed:7.1f} ms | "
            f"distance={best_distance:.4f} | "
            f"confidence={confidence:.3f} | "
            f"fast={fast_path} | "
            f"{query}"
        )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(
        f"P50  = "
        f"{percentile(latencies, 50):.1f} ms"
    )

    print(
        f"P70  = "
        f"{percentile(latencies, 70):.1f} ms"
    )

    print(
        f"P95  = "
        f"{percentile(latencies, 95):.1f} ms"
    )

    print(
        f"P100 = "
        f"{percentile(latencies, 100):.1f} ms"
    )

    print(
        f"Mean = "
        f"{statistics.mean(latencies):.1f} ms"
    )

    print("\n" + "=" * 80)
    print("DECISION")
    print("=" * 80)

    p70 = percentile(
        latencies,
        70,
    )

    if p70 <= 200:
        print(
            "PASS — E5 fast path meets "
            "the 200 ms retrieval target."
        )

    else:
        print(
            "E5 fast path is above the "
            "200 ms target."
        )

    print(
        "\nStartup time is NOT included "
        "in query latency."
    )


if __name__ == "__main__":
    main()