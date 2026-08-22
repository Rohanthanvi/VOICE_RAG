"""
Day 4 — Hybrid Retrieval Evaluation
Hacker House Goa — RAG_GOA_V2

Compares:

1. Dense baseline
   EmbeddingGemma + ChromaDB + passage_level

2. Hybrid retrieval
   EmbeddingGemma + BM25 + RRF

Metrics:
    Recall@1
    Recall@3
    Recall@5
    MRR@5

Uses the same 5,000-query evaluation set as the
previous retrieval benchmark.

Does NOT rebuild ChromaDB or embeddings.
"""

from pathlib import Path
import json
import sys

# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

from retrieval import retrieve
from hybrid_retrieval import hybrid_retrieve


# ---------------------------------------------------------
# DATA PATHS
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DENSE_TOP_K = 5
HYBRID_TOP_K = 5

MAX_QUERIES = None
# Set to 500 for a quick test.
# Set to None for all 5,000 queries.


# ---------------------------------------------------------
# LOAD QUERIES
# ---------------------------------------------------------

def load_queries():

    queries = {}

    print(
        f"Loading queries from:\n"
        f"{RAW_QUERY_FILE}"
    )

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

            queries[str(query_id)] = str(
                query_text
            ).strip()

    print(
        f"Loaded {len(queries):,} queries."
    )

    return queries


# ---------------------------------------------------------
# LOAD RELEVANCE INFORMATION
# ---------------------------------------------------------

def load_relevance():

    relevance = {}

    print(
        f"\nLoading relevance information from:\n"
        f"{PASSAGE_POOL_FILE}"
    )

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


# ---------------------------------------------------------
# BUILD EVALUATION DATASET
# ---------------------------------------------------------

def build_evaluation_set(
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
                "relevant_passages": relevant_ids,
            }
        )

    if MAX_QUERIES is not None:

        evaluation_data = (
            evaluation_data[:MAX_QUERIES]
        )

    return evaluation_data


# ---------------------------------------------------------
# EXTRACT PASSAGE IDS
# ---------------------------------------------------------

def get_passage_ids(results):

    passage_ids = []

    for result in results:

        metadata = result.get(
            "metadata",
            {},
        )

        passage_id = metadata.get(
            "passage_id"
        )

        if passage_id is not None:

            passage_ids.append(
                str(passage_id)
            )

    return passage_ids


# ---------------------------------------------------------
# RECALL@K
# ---------------------------------------------------------

def recall_at_k(
    retrieved_ids,
    relevant_ids,
    k,
):

    top_k = retrieved_ids[:k]

    for passage_id in top_k:

        if passage_id in relevant_ids:
            return 1.0

    return 0.0


# ---------------------------------------------------------
# MRR
# ---------------------------------------------------------

def reciprocal_rank(
    retrieved_ids,
    relevant_ids,
):

    for rank, passage_id in enumerate(
        retrieved_ids,
        start=1,
    ):

        if passage_id in relevant_ids:

            return 1.0 / rank

    return 0.0


# ---------------------------------------------------------
# CALCULATE METRICS
# ---------------------------------------------------------

def calculate_metrics(
    metric_data,
):

    recall1 = []
    recall3 = []
    recall5 = []
    mrr5 = []

    for item in metric_data:

        retrieved = item[
            "retrieved"
        ]

        relevant = item[
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

    if not recall1:

        return {
            "Recall@1": 0.0,
            "Recall@3": 0.0,
            "Recall@5": 0.0,
            "MRR@5": 0.0,
        }

    return {

        "Recall@1":
            sum(recall1)
            / len(recall1),

        "Recall@3":
            sum(recall3)
            / len(recall3),

        "Recall@5":
            sum(recall5)
            / len(recall5),

        "MRR@5":
            sum(mrr5)
            / len(mrr5),
    }


# ---------------------------------------------------------
# EVALUATE DENSE BASELINE
# ---------------------------------------------------------

def evaluate_dense(
    evaluation_data,
):

    print("\n")
    print("=" * 70)
    print("EVALUATING DENSE BASELINE")
    print("=" * 70)

    metric_data = []

    total = len(
        evaluation_data
    )

    for index, item in enumerate(
        evaluation_data,
        start=1,
    ):

        query = item["query"]

        relevant = item[
            "relevant_passages"
        ]

        try:

            results = retrieve(
                query,
                top_k=DENSE_TOP_K,
                strategy_filter="passage_level",
            )

            retrieved = get_passage_ids(
                results
            )

            metric_data.append(
                {
                    "retrieved": retrieved,
                    "relevant": relevant,
                }
            )

        except Exception as e:

            print(
                f"\nDense retrieval error "
                f"on query {index}: {e}"
            )

        if (
            index % 10 == 0
            or index == total
        ):

            print(
                f"\rProgress: "
                f"{index:,}/{total:,}",
                end="",
                flush=True,
            )

    print()

    return calculate_metrics(
        metric_data
    )


# ---------------------------------------------------------
# EVALUATE HYBRID
# ---------------------------------------------------------

def evaluate_hybrid(
    evaluation_data,
):

    print("\n")
    print("=" * 70)
    print("EVALUATING HYBRID BM25 + RRF")
    print("=" * 70)

    metric_data = []

    total = len(
        evaluation_data
    )

    for index, item in enumerate(
        evaluation_data,
        start=1,
    ):

        query = item["query"]

        relevant = item[
            "relevant_passages"
        ]

        try:

            results = hybrid_retrieve(
                query,
                top_k=HYBRID_TOP_K,
                strategy_filter=None,
            )

            retrieved = get_passage_ids(
                results
            )

            metric_data.append(
                {
                    "retrieved": retrieved,
                    "relevant": relevant,
                }
            )

        except Exception as e:

            print(
                f"\nHybrid retrieval error "
                f"on query {index}: {e}"
            )

        if (
            index % 10 == 0
            or index == total
        ):

            print(
                f"\rProgress: "
                f"{index:,}/{total:,}",
                end="",
                flush=True,
            )

    print()

    return calculate_metrics(
        metric_data
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print(
        "HACKER HOUSE GOA — "
        "DENSE VS HYBRID BENCHMARK"
    )
    print("=" * 70)

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    queries = load_queries()

    relevance = load_relevance()

    evaluation_data = build_evaluation_set(
        queries,
        relevance,
    )

    print(
        f"\nFinal evaluation queries: "
        f"{len(evaluation_data):,}"
    )

    if not evaluation_data:

        print(
            "\nNo evaluation data found."
        )

        return

    # -----------------------------------------------------
    # Dense baseline
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
    # FINAL RESULTS
    # -----------------------------------------------------

    print("\n")
    print("=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)

    print(
        f"\n"
        f"{'Strategy':<28}"
        f"{'Recall@1':>12}"
        f"{'Recall@3':>12}"
        f"{'Recall@5':>12}"
        f"{'MRR@5':>12}"
    )

    print("-" * 75)

    print(
        f"{'Dense passage_level':<28}"
        f"{dense_metrics['Recall@1']:>12.4f}"
        f"{dense_metrics['Recall@3']:>12.4f}"
        f"{dense_metrics['Recall@5']:>12.4f}"
        f"{dense_metrics['MRR@5']:>12.4f}"
    )

    print(
        f"{'Hybrid BM25 + RRF':<28}"
        f"{hybrid_metrics['Recall@1']:>12.4f}"
        f"{hybrid_metrics['Recall@3']:>12.4f}"
        f"{hybrid_metrics['Recall@5']:>12.4f}"
        f"{hybrid_metrics['MRR@5']:>12.4f}"
    )

    print("-" * 75)

    # -----------------------------------------------------
    # Improvement
    # -----------------------------------------------------

    print("\nIMPROVEMENT")

    for metric in [
        "Recall@1",
        "Recall@3",
        "Recall@5",
        "MRR@5",
    ]:

        baseline = dense_metrics[
            metric
        ]

        hybrid = hybrid_metrics[
            metric
        ]

        absolute = (
            hybrid - baseline
        )

        if baseline != 0:

            percentage = (
                absolute
                / baseline
                * 100
            )

        else:

            percentage = 0.0

        print(
            f"{metric:<10}: "
            f"{baseline:.4f} → "
            f"{hybrid:.4f} "
            f"({absolute:+.4f}, "
            f"{percentage:+.2f}%)"
        )

    # -----------------------------------------------------
    # Winner
    # -----------------------------------------------------

    dense_score = (
        dense_metrics["Recall@5"]
    )

    hybrid_score = (
        hybrid_metrics["Recall@5"]
    )

    print("\n")

    if hybrid_score > dense_score:

        print(
            "🏆 WINNER: HYBRID BM25 + RRF"
        )

        print(
            f"Recall@5 improved from "
            f"{dense_score:.4f} "
            f"to "
            f"{hybrid_score:.4f}"
        )

        print(
            "\nNext step: add a reranker."
        )

    elif hybrid_score < dense_score:

        print(
            "🏆 WINNER: DENSE PASSAGE_LEVEL"
        )

        print(
            f"Recall@5 decreased from "
            f"{dense_score:.4f} "
            f"to "
            f"{hybrid_score:.4f}"
        )

        print(
            "\nNext step: tune BM25/RRF "
            "before adding a reranker."
        )

    else:

        print(
            "DRAW: Both systems have "
            "the same Recall@5."
        )

        print(
            "\nNext step: compare MRR@5 "
            "and tune RRF."
        )

    print(
        "\nBenchmark complete."
    )


if __name__ == "__main__":
    main()