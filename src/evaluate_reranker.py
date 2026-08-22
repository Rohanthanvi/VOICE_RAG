"""
Stage 3 — Reranker Evaluation
Hacker House Goa — RAG_GOA_V2

Compares:

1. Dense baseline
   EmbeddingGemma + ChromaDB + passage_level

2. Hybrid
   EmbeddingGemma + BM25 + RRF

3. Hybrid + multilingual reranker
   EmbeddingGemma + BM25 + RRF + BGE-reranker-v2-m3

Metrics:
    Recall@1
    Recall@3
    Recall@5
    MRR@5

First run:
    MAX_QUERIES = 100

After successful validation:
    MAX_QUERIES = None

No ChromaDB rebuilding.
No embedding rebuilding.
"""

from pathlib import Path
import json
import sys

# =========================================================
# PATHS
# =========================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

from retrieval import retrieve
from hybrid_retrieval import hybrid_retrieve
from reranked_retrieval import reranked_retrieve


# =========================================================
# DATA
# =========================================================

RAW_QUERY_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "msmarco_hi_train_5000.jsonl"
)

PASSAGE_POOL_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "passage_pool_hi.jsonl"
)


# =========================================================
# CONFIGURATION
# =========================================================

# Start with 100.
# After successful validation change to None
# for the complete 5,000-query benchmark.

MAX_QUERIES = 100

DENSE_TOP_K = 5
HYBRID_TOP_K = 5
RERANK_TOP_K = 5


# =========================================================
# LOAD QUERIES
# =========================================================

def load_queries():

    queries = {}

    print(
        "Loading queries from:"
    )
    print(RAW_QUERY_FILE)

    with open(
        RAW_QUERY_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            query_id = (
                item.get("query_id")
                or item.get("id")
            )

            query_text = (
                item.get("query")
                or item.get("query_text")
                or item.get("question")
            )

            if query_id is None:
                continue

            if not query_text:
                continue

            queries[str(query_id)] = (
                str(query_text).strip()
            )

    print(
        f"Loaded {len(queries):,} queries."
    )

    return queries


# =========================================================
# LOAD RELEVANCE
# =========================================================

def load_relevance():

    relevance = {}

    print(
        "\nLoading relevance information from:"
    )
    print(PASSAGE_POOL_FILE)

    with open(
        PASSAGE_POOL_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            passage = json.loads(line)

            passage_id = str(
                passage["passage_id"]
            )

            query_ids = passage.get(
                "seen_with_query_ids",
                [],
            )

            for query_id in query_ids:

                query_id = str(query_id)

                if query_id not in relevance:
                    relevance[query_id] = set()

                relevance[query_id].add(
                    passage_id
                )

    print(
        f"Found relevance information for "
        f"{len(relevance):,} queries."
    )

    return relevance


# =========================================================
# BUILD EVALUATION DATA
# =========================================================

def build_evaluation_data(
    queries,
    relevance,
):

    evaluation_data = []

    for query_id, relevant_ids in (
        relevance.items()
    ):

        if query_id not in queries:
            continue

        evaluation_data.append(
            {
                "query_id": query_id,
                "query": queries[query_id],
                "relevant": relevant_ids,
            }
        )

    if MAX_QUERIES is not None:

        evaluation_data = (
            evaluation_data[:MAX_QUERIES]
        )

    return evaluation_data


# =========================================================
# EXTRACT PASSAGE IDS
# =========================================================

def get_passage_ids(results):

    ids = []

    for result in results:

        metadata = result.get(
            "metadata",
            {},
        )

        passage_id = metadata.get(
            "passage_id"
        )

        if passage_id is not None:

            ids.append(
                str(passage_id)
            )

    return ids


# =========================================================
# RECALL@K
# =========================================================

def recall_at_k(
    retrieved_ids,
    relevant_ids,
    k,
):

    for passage_id in retrieved_ids[:k]:

        if passage_id in relevant_ids:
            return 1.0

    return 0.0


# =========================================================
# MRR@5
# =========================================================

def reciprocal_rank(
    retrieved_ids,
    relevant_ids,
):

    for rank, passage_id in enumerate(
        retrieved_ids[:5],
        start=1,
    ):

        if passage_id in relevant_ids:

            return 1.0 / rank

    return 0.0


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(records):

    if not records:

        return {
            "Recall@1": 0.0,
            "Recall@3": 0.0,
            "Recall@5": 0.0,
            "MRR@5": 0.0,
        }

    recall1 = []
    recall3 = []
    recall5 = []
    mrr5 = []

    for record in records:

        retrieved = record[
            "retrieved"
        ]

        relevant = record[
            "relevant"
        ]

        recall1.append(
            recall_at_k(
                retrieved,
                relevant,
                1,
            )
        )

        recall3.append(
            recall_at_k(
                retrieved,
                relevant,
                3,
            )
        )

        recall5.append(
            recall_at_k(
                retrieved,
                relevant,
                5,
            )
        )

        mrr5.append(
            reciprocal_rank(
                retrieved,
                relevant,
            )
        )

    return {
        "Recall@1":
            sum(recall1) / len(recall1),

        "Recall@3":
            sum(recall3) / len(recall3),

        "Recall@5":
            sum(recall5) / len(recall5),

        "MRR@5":
            sum(mrr5) / len(mrr5),
    }


# =========================================================
# DENSE BENCHMARK
# =========================================================

def evaluate_dense(
    evaluation_data,
):

    print("\n")
    print("=" * 70)
    print("EVALUATING DENSE BASELINE")
    print("=" * 70)

    records = []

    total = len(evaluation_data)

    for index, item in enumerate(
        evaluation_data,
        start=1,
    ):

        try:

            results = retrieve(
                item["query"],
                top_k=DENSE_TOP_K,
                strategy_filter="passage_level",
            )

            retrieved = get_passage_ids(
                results
            )

            records.append(
                {
                    "retrieved": retrieved,
                    "relevant": item["relevant"],
                }
            )

        except Exception as e:

            print(
                f"\nDense error on query "
                f"{index}: {e}"
            )

        if (
            index % 10 == 0
            or index == total
        ):

            print(
                f"\rProgress: "
                f"{index}/{total}",
                end="",
                flush=True,
            )

    print()

    return calculate_metrics(records)


# =========================================================
# HYBRID BENCHMARK
# =========================================================

def evaluate_hybrid(
    evaluation_data,
):

    print("\n")
    print("=" * 70)
    print("EVALUATING HYBRID BM25 + RRF")
    print("=" * 70)

    records = []

    total = len(evaluation_data)

    for index, item in enumerate(
        evaluation_data,
        start=1,
    ):

        try:

            results = hybrid_retrieve(
                item["query"],
                top_k=HYBRID_TOP_K,
            )

            retrieved = get_passage_ids(
                results
            )

            records.append(
                {
                    "retrieved": retrieved,
                    "relevant": item["relevant"],
                }
            )

        except Exception as e:

            print(
                f"\nHybrid error on query "
                f"{index}: {e}"
            )

        if (
            index % 10 == 0
            or index == total
        ):

            print(
                f"\rProgress: "
                f"{index}/{total}",
                end="",
                flush=True,
            )

    print()

    return calculate_metrics(records)


# =========================================================
# RERANKER BENCHMARK
# =========================================================

def evaluate_reranker(
    evaluation_data,
):

    print("\n")
    print("=" * 70)
    print("EVALUATING HYBRID + RERANKER")
    print("=" * 70)

    records = []

    total = len(evaluation_data)

    for index, item in enumerate(
        evaluation_data,
        start=1,
    ):

        try:

            results = reranked_retrieve(
                item["query"],
                top_k=RERANK_TOP_K,
            )

            retrieved = get_passage_ids(
                results
            )

            records.append(
                {
                    "retrieved": retrieved,
                    "relevant": item["relevant"],
                }
            )

        except Exception as e:

            print(
                f"\nReranker error on query "
                f"{index}: {e}"
            )

        if (
            index % 10 == 0
            or index == total
        ):

            print(
                f"\rProgress: "
                f"{index}/{total}",
                end="",
                flush=True,
            )

    print()

    return calculate_metrics(records)


# =========================================================
# PRINT RESULTS
# =========================================================

def print_results(
    dense,
    hybrid,
    reranked,
):

    print("\n")
    print("=" * 80)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 80)

    print(
        f"\n"
        f"{'Strategy':<30}"
        f"{'Recall@1':>12}"
        f"{'Recall@3':>12}"
        f"{'Recall@5':>12}"
        f"{'MRR@5':>12}"
    )

    print("-" * 80)

    print(
        f"{'Dense passage_level':<30}"
        f"{dense['Recall@1']:>12.4f}"
        f"{dense['Recall@3']:>12.4f}"
        f"{dense['Recall@5']:>12.4f}"
        f"{dense['MRR@5']:>12.4f}"
    )

    print(
        f"{'Hybrid BM25 + RRF':<30}"
        f"{hybrid['Recall@1']:>12.4f}"
        f"{hybrid['Recall@3']:>12.4f}"
        f"{hybrid['Recall@5']:>12.4f}"
        f"{hybrid['MRR@5']:>12.4f}"
    )

    print(
        f"{'Hybrid + Reranker':<30}"
        f"{reranked['Recall@1']:>12.4f}"
        f"{reranked['Recall@3']:>12.4f}"
        f"{reranked['Recall@5']:>12.4f}"
        f"{reranked['MRR@5']:>12.4f}"
    )

    print("-" * 80)


# =========================================================
# IMPROVEMENTS
# =========================================================

def print_improvements(
    dense,
    hybrid,
    reranked,
):

    print("\n")
    print("=" * 80)
    print("IMPROVEMENTS")
    print("=" * 80)

    metrics = [
        "Recall@1",
        "Recall@3",
        "Recall@5",
        "MRR@5",
    ]

    print("\nHybrid vs Dense:")

    for metric in metrics:

        change = (
            hybrid[metric]
            - dense[metric]
        )

        print(
            f"{metric:<10}: "
            f"{dense[metric]:.4f} → "
            f"{hybrid[metric]:.4f} "
            f"({change:+.4f})"
        )

    print("\nReranker vs Hybrid:")

    for metric in metrics:

        change = (
            reranked[metric]
            - hybrid[metric]
        )

        print(
            f"{metric:<10}: "
            f"{hybrid[metric]:.4f} → "
            f"{reranked[metric]:.4f} "
            f"({change:+.4f})"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 80)
    print(
        "HACKER HOUSE GOA — "
        "RERANKER BENCHMARK"
    )
    print("=" * 80)

    queries = load_queries()

    relevance = load_relevance()

    evaluation_data = build_evaluation_data(
        queries,
        relevance,
    )

    print(
        f"\nFinal evaluation queries: "
        f"{len(evaluation_data):,}"
    )

    if not evaluation_data:

        print(
            "\nERROR: No evaluation queries."
        )

        return

    # -----------------------------------------------------
    # Dense
    # -----------------------------------------------------

    dense_metrics = evaluate_dense(
        evaluation_data
    )

    # -----------------------------------------------------
    # Hybrid
    # -----------------------------------------------------

    hybrid_metrics = evaluate_hybrid(
        evaluation_data
    )

    # -----------------------------------------------------
    # Hybrid + Reranker
    # -----------------------------------------------------

    reranked_metrics = evaluate_reranker(
        evaluation_data
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print_results(
        dense_metrics,
        hybrid_metrics,
        reranked_metrics,
    )

    print_improvements(
        dense_metrics,
        hybrid_metrics,
        reranked_metrics,
    )

    # -----------------------------------------------------
    # Winner
    # -----------------------------------------------------

    print("\n")
    print("=" * 80)
    print("WINNER")
    print("=" * 80)

    scores = {
        "Dense passage_level":
            dense_metrics["MRR@5"],

        "Hybrid BM25 + RRF":
            hybrid_metrics["MRR@5"],

        "Hybrid + Reranker":
            reranked_metrics["MRR@5"],
    }

    winner = max(
        scores,
        key=scores.get,
    )

    print(
        f"\nBest MRR@5: {winner}"
    )

    print(
        f"MRR@5 = "
        f"{scores[winner]:.4f}"
    )

    print(
        "\nBenchmark complete."
    )


if __name__ == "__main__":
    main()