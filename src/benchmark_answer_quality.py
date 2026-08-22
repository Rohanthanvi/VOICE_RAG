"""
Hacker House Goa — Answer Quality Benchmark

Checks the production harness for:

    - query
    - answer
    - answer mode
    - retrieval latency
    - total latency
    - top retrieved chunk
    - distance
    - failures

This benchmark does NOT modify:
    - E5 index
    - ChromaDB
    - BM25 index
    - production harness

Run:
    python src/benchmark_answer_quality.py
"""

from __future__ import annotations

import time

from harness import warmup, run_pipeline_text


# ============================================================
# TEST QUERIES
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
# MAIN
# ============================================================


def main():

    print("=" * 90)
    print("HACKER HOUSE GOA — ANSWER QUALITY BENCHMARK")
    print("=" * 90)

    print(
        f"\nQueries: {len(QUERIES)}"
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
        "Startup time is excluded from "
        "per-query measurements."
    )

    # ========================================================
    # RUN
    # ========================================================

    print("\nRunning answer-quality tests...\n")

    successful = 0
    failed = 0

    e5_fast = 0
    hybrid_fast = 0
    generative = 0

    latencies = []

    for i, query in enumerate(
        QUERIES,
        start=1,
    ):

        print("-" * 90)

        print(
            f"\n[{i:02d}/{len(QUERIES)}] "
            f"QUERY:"
        )

        print(query)

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

            if result.success:
                successful += 1
            else:
                failed += 1

            if result.answer_mode == "e5_fast":
                e5_fast += 1

            elif result.answer_mode == "hybrid_fast":
                hybrid_fast += 1

            elif result.answer_mode == "generative":
                generative += 1

            # ------------------------------------------------
            # OUTPUT
            # ------------------------------------------------

            print(
                "\nANSWER:"
            )

            print(
                result.answer
                if result.answer
                else "[NO ANSWER]"
            )

            print(
                f"\nMODE: "
                f"{result.answer_mode}"
            )

            print(
                f"LATENCY: "
                f"{elapsed:.2f} ms"
            )

            print(
                f"RETRIEVAL: "
                f"{result.timings.retrieval_ms:.2f} ms"
                if result.timings.retrieval_ms
                is not None
                else "RETRIEVAL: N/A"
            )

            print(
                f"TOTAL: "
                f"{result.timings.total_ms:.2f} ms"
                if result.timings.total_ms
                is not None
                else "TOTAL: N/A"
            )

            # ------------------------------------------------
            # TOP RETRIEVED CHUNK
            # ------------------------------------------------

            if result.retrieved_chunks:

                top = (
                    result.retrieved_chunks[0]
                )

                print(
                    "\nTOP RETRIEVED CHUNK:"
                )

                print(
                    f"Strategy : "
                    f"{top.strategy}"
                )

                print(
                    f"Distance : "
                    f"{top.distance:.4f}"
                )

                print(
                    f"E5 rank  : "
                    f"{top.e5_rank}"
                )

                print(
                    f"BM25 rank: "
                    f"{top.bm25_rank}"
                )

                print(
                    "Text:"
                )

                print(
                    top.text[:500]
                )

            else:

                print(
                    "\nTOP RETRIEVED CHUNK: NONE"
                )

            # ------------------------------------------------
            # GROUNDING
            # ------------------------------------------------

            if (
                result.grounding_score
                is not None
            ):

                print(
                    f"\nGROUNDING SCORE: "
                    f"{result.grounding_score:.3f}"
                )

            if result.error_stage:

                print(
                    f"\nERROR STAGE: "
                    f"{result.error_stage}"
                )

                print(
                    f"ERROR: "
                    f"{result.error_message}"
                )

        except Exception as e:

            failed += 1

            print(
                f"\nFAILED: {e}"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 90)
    print("ANSWER QUALITY BENCHMARK SUMMARY")
    print("=" * 90)

    total = len(QUERIES)

    print(
        f"\nSuccessful queries : "
        f"{successful}/{total}"
    )

    print(
        f"Failed queries     : "
        f"{failed}/{total}"
    )

    print(
        f"E5 fast path       : "
        f"{e5_fast}/{total} "
        f"({e5_fast / total * 100:.1f}%)"
    )

    print(
        f"Hybrid fast path   : "
        f"{hybrid_fast}/{total} "
        f"({hybrid_fast / total * 100:.1f}%)"
    )

    print(
        f"Generative fallback: "
        f"{generative}/{total} "
        f"({generative / total * 100:.1f}%)"
    )

    if latencies:

        print(
            f"\nMean latency       : "
            f"{sum(latencies) / len(latencies):.2f} ms"
        )

        print(
            f"Fastest            : "
            f"{min(latencies):.2f} ms"
        )

        print(
            f"Slowest            : "
            f"{max(latencies):.2f} ms"
        )

    print("\n")
    print("=" * 90)
    print("MANUAL QUALITY CHECK")
    print("=" * 90)

    print(
        """
Review each ANSWER above.

Check:

1. Does the answer actually answer the question?
2. Is it about the correct entity?
3. Is it concise enough?
4. Is it supported by the retrieved chunk?
5. Is the answer mode appropriate?
"""
    )

    print("=" * 90)


if __name__ == "__main__":
    main()