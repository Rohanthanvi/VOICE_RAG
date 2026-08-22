"""
Day 4 (part 2) — Latency benchmark: P50 / P70 / P100

Runs the pipeline (retrieval + generation, starting from already-
transcribed text — see harness.py's run_pipeline_text) across many real
test queries and reports percentile latencies. STT is measured
separately (see the note printed at the end) since its latency is mostly
a function of network/audio length, not query content — it doesn't need
a 100-query sample to characterize the way retrieval and generation do.

Test query mix: pulls from BOTH the "findable" queries (answer passage
is actually in the index — normal successful path) and a sample of
random queries from the full 5000 (many of which will trigger the
no-relevant-context guardrail) — this gives percentiles that reflect
real mixed usage, not just best-case "it always finds an answer" runs.

Run:
    python src/benchmark_latency.py --n 100
"""

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

from tqdm import tqdm

from harness import run_pipeline_text, warmup_retrieval

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RANDOM_SEED = 7


def text_to_id(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


def load_test_queries(n: int) -> list[str]:
    """Mix of findable (answerable) and general (mixed outcome) queries,
    for percentiles that reflect real usage rather than only best-case
    successful lookups."""
    with open(RAW_DIR / "msmarco_hi_train_5000.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    random.seed(RANDOM_SEED)
    sample = random.sample(rows, min(n, len(rows)))
    return [row["query"] for row in sample]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(int(len(values) * pct / 100), len(values) - 1)
    return values[idx]


def main(n: int):
    print("Warming up (loading embedding model + ChromaDB collection) ...")
    warmup_retrieval()

    queries = load_test_queries(n)
    print(f"Running {len(queries)} test queries ...")

    retrieval_times, generation_times, total_times = [], [], []
    guardrail_counts = {"no_relevant_context": 0, "unsafe_input": 0}
    success_count, error_count = 0, 0
    records = []

    for query in tqdm(queries, desc="Benchmarking"):
        result = run_pipeline_text(query)
        records.append(result.model_dump())

        if not result.success:
            error_count += 1
            continue
        success_count += 1

        if result.guardrail_triggered:
            if result.guardrail_triggered.startswith("unsafe_input"):
                guardrail_counts["unsafe_input"] += 1
            elif result.guardrail_triggered == "no_relevant_context":
                guardrail_counts["no_relevant_context"] += 1
            # guardrail-triggered runs skip generation, so only count
            # retrieval/total time, not generation, for these
            if result.timings.retrieval_ms is not None:
                retrieval_times.append(result.timings.retrieval_ms)
            if result.timings.total_ms is not None:
                total_times.append(result.timings.total_ms)
            continue

        if result.timings.retrieval_ms is not None:
            retrieval_times.append(result.timings.retrieval_ms)
        if result.timings.generation_ms is not None:
            generation_times.append(result.timings.generation_ms)
        if result.timings.total_ms is not None:
            total_times.append(result.timings.total_ms)

    def report(name: str, values: list[float]):
        if not values:
            print(f"  {name}: no data")
            return
        print(f"  {name}: P50={percentile(values, 50):.1f}ms  "
              f"P70={percentile(values, 70):.1f}ms  "
              f"P100(max)={percentile(values, 100):.1f}ms  "
              f"mean={statistics.mean(values):.1f}ms  n={len(values)}")

    print(f"\n=== Results over {len(queries)} text queries (STT excluded — measured separately) ===")
    print(f"Successful runs: {success_count}, errors: {error_count}")
    print(f"Guardrail triggers: no_relevant_context={guardrail_counts['no_relevant_context']}, "
          f"unsafe_input={guardrail_counts['unsafe_input']}")
    print()
    report("Retrieval latency", retrieval_times)
    report("Generation latency", generation_times)
    report("Total (retrieval+generation) latency", total_times)

    out_path = REPORTS_DIR / "latency_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_queries": len(queries),
            "success_count": success_count,
            "error_count": error_count,
            "guardrail_counts": guardrail_counts,
            "retrieval_ms": {"p50": percentile(retrieval_times, 50), "p70": percentile(retrieval_times, 70), "p100": percentile(retrieval_times, 100)},
            "generation_ms": {"p50": percentile(generation_times, 50), "p70": percentile(generation_times, 70), "p100": percentile(generation_times, 100)},
            "total_ms": {"p50": percentile(total_times, 50), "p70": percentile(total_times, 70), "p100": percentile(total_times, 100)},
            "raw_records": records,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nFull report saved -> {out_path}")
    print("\nNOTE: STT latency is not included above. Measure it separately by running")
    print("src/harness.py on 5-10 real audio recordings and noting the stt_ms values,")
    print("then report combined latency as STT_typical + the percentiles above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="number of test queries to run")
    args = parser.parse_args()
    main(args.n)