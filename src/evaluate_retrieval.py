"""
Day 4 — Retrieval Evaluation

Hacker House Goa — RAG_GOA_V2

Evaluates:
    passage_level
    semantic_sentence

Metrics:
    Recall@1
    Recall@3
    Recall@5
    MRR@5

Uses:
    data/raw/msmarco_hi_train_5000.jsonl
    data/processed/passage_pool_hi.jsonl
    existing ChromaDB index

No embeddings are rebuilt.
"""

from pathlib import Path
import json
import sys

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

from retrieval import retrieve


# ---------------------------------------------------------
# PATHS
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
# CONFIG
# ---------------------------------------------------------

STRATEGIES = [
    "passage_level",
    "semantic_sentence",
]

TOP_K = 5


# ---------------------------------------------------------
# LOAD RAW QUERIES
# ---------------------------------------------------------

def load_queries():
    """
    Load query_id -> query text from the raw MS MARCO file.
    """

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

            # Support common field names
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


# ---------------------------------------------------------
# LOAD RELEVANT PASSAGES
# ---------------------------------------------------------

def load_relevant_passages():
    """
    Build:

        query_id -> set(relevant passage_ids)

    from passage_pool_hi.jsonl.
    """

    relevant = {}

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

            passage_id = passage[
                "passage_id"
            ]

            query_ids = passage.get(
                "seen_with_query_ids",
                [],
            )

            for query_id in query_ids:

                query_id = str(query_id)

                if query_id not in relevant:
                    relevant[query_id] = set()

                relevant[query_id].add(
                    passage_id
                )

    print(
        f"Found relevance information for "
        f"{len(relevant):,} queries."
    )

    return relevant


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def get_retrieved_passage_ids(results):
    """
    Extract passage IDs from retrieval results.
    """

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


def recall_at_k(
    retrieved_ids,
    relevant_ids,
    k,
):
    """
    Hit/Recall@K.

    Returns 1 if at least one relevant passage
    appears in the first K results.
    """

    top_k = retrieved_ids[:k]

    return float(
        any(
            passage_id in relevant_ids
            for passage_id in top_k
        )
    )


def reciprocal_rank(
    retrieved_ids,
    relevant_ids,
):
    """
    Reciprocal rank of the first relevant result.
    """

    for rank, passage_id in enumerate(
        retrieved_ids,
        start=1,
    ):

        if passage_id in relevant_ids:
            return 1.0 / rank

    return 0.0


# ---------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------

def evaluate_strategy(
    evaluation_data,
    strategy,
):
    """
    Evaluate one retrieval strategy.
    """

    recall1 = []
    recall3 = []
    recall5 = []
    mrr5 = []

    total = len(evaluation_data)

    print(
        f"\nEvaluating: {strategy}"
    )

    print(
        f"Total queries: {total:,}"
    )

    for index, item in enumerate(
        evaluation_data,
        start=1,
    ):

        query = item["query"]

        relevant_ids = item[
            "relevant_passages"
        ]

        try:

            results = retrieve(
                query,
                top_k=TOP_K,
                strategy_filter=strategy,
            )

        except Exception as e:

            print(
                f"\nRetrieval error on "
                f"query {index}: {e}"
            )

            continue

        retrieved_ids = (
            get_retrieved_passage_ids(
                results
            )
        )

        recall1.append(
            recall_at_k(
                retrieved_ids,
                relevant_ids,
                1,
            )
        )

        recall3.append(
            recall_at_k(
                retrieved_ids,
                relevant_ids,
                3,
            )
        )

        recall5.append(
            recall_at_k(
                retrieved_ids,
                relevant_ids,
                5,
            )
        )

        mrr5.append(
            reciprocal_rank(
                retrieved_ids,
                relevant_ids,
            )
        )

        # Progress every 10 queries
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

    if not recall1:
        return {
            "Recall@1": 0.0,
            "Recall@3": 0.0,
            "Recall@5": 0.0,
            "MRR@5": 0.0,
        }

    return {
        "Recall@1": sum(recall1)
        / len(recall1),

        "Recall@3": sum(recall3)
        / len(recall3),

        "Recall@5": sum(recall5)
        / len(recall5),

        "MRR@5": sum(mrr5)
        / len(mrr5),
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 75)
    print(
        "HACKER HOUSE GOA — RETRIEVAL BENCHMARK"
    )
    print("=" * 75)

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    queries = load_queries()

    relevant = load_relevant_passages()

    # -----------------------------------------------------
    # Join query + relevance
    # -----------------------------------------------------

    evaluation_data = []

    for query_id, relevant_passages in (
        relevant.items()
    ):

        if query_id not in queries:
            continue

        evaluation_data.append(
            {
                "query_id": query_id,
                "query": queries[query_id],
                "relevant_passages":
                    relevant_passages,
            }
        )

    print(
        f"\nFinal evaluation queries: "
        f"{len(evaluation_data):,}"
    )

    if not evaluation_data:

        print(
            "\nERROR: No matching query IDs "
            "were found."
        )

        return

    # -----------------------------------------------------
    # Evaluate strategies
    # -----------------------------------------------------

    results = {}

    for strategy in STRATEGIES:

        results[strategy] = (
            evaluate_strategy(
                evaluation_data,
                strategy,
            )
        )

    # -----------------------------------------------------
    # Results table
    # -----------------------------------------------------

    print("\n")
    print("=" * 75)
    print("FINAL RESULTS")
    print("=" * 75)

    print(
        f"\n"
        f"{'Strategy':<25}"
        f"{'Recall@1':>12}"
        f"{'Recall@3':>12}"
        f"{'Recall@5':>12}"
        f"{'MRR@5':>12}"
    )

    print("-" * 75)

    for strategy, metrics in results.items():

        print(
            f"{strategy:<25}"
            f"{metrics['Recall@1']:>12.4f}"
            f"{metrics['Recall@3']:>12.4f}"
            f"{metrics['Recall@5']:>12.4f}"
            f"{metrics['MRR@5']:>12.4f}"
        )

    print("-" * 75)

    # -----------------------------------------------------
    # Determine winner
    # -----------------------------------------------------

    best_strategy = max(
        results,
        key=lambda strategy: (
            results[strategy]["Recall@5"],
            results[strategy]["MRR@5"],
        ),
    )

    print(
        f"\nBEST STRATEGY: "
        f"{best_strategy}"
    )

    print(
        f"Recall@5: "
        f"{results[best_strategy]['Recall@5']:.4f}"
    )

    print(
        f"MRR@5: "
        f"{results[best_strategy]['MRR@5']:.4f}"
    )

    print(
        "\nBenchmark complete."
    )


if __name__ == "__main__":
    main()