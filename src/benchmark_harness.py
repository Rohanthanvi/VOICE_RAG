"""
Hacker House Goa — Full Harness Latency Benchmark

Benchmarks the complete text pipeline using the already-built
production E5 retrieval + query-aware fast answer path.

Measures:
    - P50
    - P70
    - P95
    - P100
    - Mean
    - Fast-path percentage
    - Generative fallback percentage
    - Failures

Warmup/startup time is excluded from query latency.

Run:
    python src/benchmark_harness.py
"""

from __future__ import annotations

import statistics
import time

from harness import warmup, run_pipeline_text


# ============================================================
# BENCHMARK QUERIES
# ============================================================

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

    "भारत का राष्ट्रीय पशु कौन सा है",
    "सौर मंडल क्या है",
    "पानी का रासायनिक सूत्र क्या है",
    "ताजमहल कहाँ स्थित है",
    "भारत का संविधान क्या है",
    "महाभारत क्या है",
    "माउंट एवरेस्ट कहाँ है",
    "सूर्य क्या है",
    "चंद्रमा क्या है",
    "पृथ्वी क्या है",

    "लोकतंत्र क्या है",
    "पर्यावरण क्या है",
    "वायु प्रदूषण क्या है",
    "जल प्रदूषण क्या है",
    "गुरुत्वाकर्षण क्या है",
    "प्रकाश संश्लेषण क्या है",
    "DNA क्या है",
    "कंप्यूटर विज्ञान क्या है",
    "मशीन लर्निंग क्या है",
    "डीप लर्निंग क्या है",
]


# ============================================================
# PERCENTILE
# ============================================================


def percentile(
    values: list[float],
    p: float,
) -> float:

    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (
        (len(values) - 1)
        * p
        / 100.0
    )

    lower = int(position)

    upper = min(
        lower + 1,
        len(values) - 1,
    )

    weight = position - lower

    return (
        values[lower]
        * (1 - weight)
        +
        values[upper]
        * weight
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 80)
    print("HACKER HOUSE GOA — FULL HARNESS LATENCY BENCHMARK")
    print("=" * 80)

    print(
        f"\nBenchmark queries: {len(QUERIES)}"
    )

    # ========================================================
    # WARMUP
    # ========================================================

    print("\nWarming up production system...")

    startup_start = time.perf_counter()

    warmup()

    startup_ms = (
        time.perf_counter()
        - startup_start
    ) * 1000

    print(
        f"\nStartup / warmup: "
        f"{startup_ms:.1f} ms"
    )

    print(
        "Startup time is NOT included "
        "in per-query latency."
    )

    # ========================================================
    # BENCHMARK
    # ========================================================

    print("\nRunning warm-query benchmark...\n")

    latencies = []

    fast_count = 0
    hybrid_count = 0
    generative_count = 0
    failed_count = 0

    results = []

    for i, query in enumerate(
        QUERIES,
        start=1,
    ):

        t0 = time.perf_counter()

        try:

            result = run_pipeline_text(
                query
            )

            elapsed = (
                time.perf_counter()
                - t0
            ) * 1000

            latencies.append(
                elapsed
            )

            mode = (
                result.answer_mode
                or "none"
            )

            if mode == "e5_fast":
                fast_count += 1

            elif mode == "hybrid_fast":
                hybrid_count += 1

            elif mode == "generative":
                generative_count += 1

            if not result.success:
                failed_count += 1

            results.append(
                {
                    "query": query,
                    "latency": elapsed,
                    "mode": mode,
                    "success": result.success,
                }
            )

            print(
                f"{i:02d}. "
                f"{elapsed:8.2f} ms | "
                f"{mode:13s} | "
                f"{query}"
            )

        except Exception as e:

            failed_count += 1

            print(
                f"{i:02d}. "
                f"FAILED | "
                f"{query}"
            )

            print(
                f"    Error: {e}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("HARNESS LATENCY RESULTS")
    print("=" * 80)

    if not latencies:

        print(
            "\nNo successful benchmark "
            "queries."
        )

        return

    p50 = percentile(
        latencies,
        50,
    )

    p70 = percentile(
        latencies,
        70,
    )

    p95 = percentile(
        latencies,
        95,
    )

    p100 = max(
        latencies
    )

    mean = statistics.mean(
        latencies
    )

    print(
        f"\nP50  = {p50:8.2f} ms"
    )

    print(
        f"P70  = {p70:8.2f} ms"
    )

    print(
        f"P95  = {p95:8.2f} ms"
    )

    print(
        f"P100 = {p100:8.2f} ms"
    )

    print(
        f"Mean = {mean:8.2f} ms"
    )

    # ========================================================
    # PATH STATISTICS
    # ========================================================

    total = len(QUERIES)

    print("\n")
    print("=" * 80)
    print("ANSWER PATH DISTRIBUTION")
    print("=" * 80)

    print(
        f"\nE5 fast path       : "
        f"{fast_count}/{total} "
        f"({fast_count / total * 100:.1f}%)"
    )

    print(
        f"Hybrid fast path   : "
        f"{hybrid_count}/{total} "
        f"({hybrid_count / total * 100:.1f}%)"
    )

    print(
        f"Generative fallback: "
        f"{generative_count}/{total} "
        f"({generative_count / total * 100:.1f}%)"
    )

    print(
        f"Failures           : "
        f"{failed_count}/{total}"
    )

    # ========================================================
    # TARGET
    # ========================================================

    print("\n")
    print("=" * 80)
    print("200 MS TARGET")
    print("=" * 80)

    if p70 <= 200:

        print(
            f"\nPASS — P70 "
            f"({p70:.2f} ms) "
            f"is below 200 ms."
        )

    else:

        print(
            f"\nFAIL — P70 "
            f"({p70:.2f} ms) "
            f"exceeds 200 ms."
        )

    if p95 <= 200:

        print(
            f"PASS — P95 "
            f"({p95:.2f} ms) "
            f"is below 200 ms."
        )

    else:

        print(
            f"WARNING — P95 "
            f"({p95:.2f} ms) "
            f"exceeds 200 ms."
        )

    # ========================================================
    # QUALITY / PATH CHECK
    # ========================================================

    print("\n")
    print("=" * 80)
    print("PRODUCTION READINESS")
    print("=" * 80)

    if failed_count == 0:

        print(
            "\nNo pipeline failures."
        )

    else:

        print(
            f"\nWARNING: "
            f"{failed_count} pipeline failures."
        )

    if fast_count >= total * 0.80:

        print(
            "Fast-path coverage: GOOD"
        )

    else:

        print(
            "Fast-path coverage: NEEDS REVIEW"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"\nStartup / warmup : "
        f"{startup_ms:.0f} ms"
    )

    print(
        f"P50              : "
        f"{p50:.2f} ms"
    )

    print(
        f"P70              : "
        f"{p70:.2f} ms"
    )

    print(
        f"P95              : "
        f"{p95:.2f} ms"
    )

    print(
        f"P100             : "
        f"{p100:.2f} ms"
    )

    print(
        f"Mean             : "
        f"{mean:.2f} ms"
    )

    print(
        f"Fast path        : "
        f"{fast_count / total * 100:.1f}%"
    )

    print(
        f"Generative       : "
        f"{generative_count / total * 100:.1f}%"
    )

    print(
        f"Failures         : "
        f"{failed_count}"
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
    