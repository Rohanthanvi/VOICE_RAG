"""
Stage 3 — Hybrid Retrieval + Multilingual Reranking

Pipeline:

Query
  ↓
Dense Retrieval + BM25
  ↓
RRF Fusion
  ↓
Top 20 Candidates
  ↓
BGE Reranker
  ↓
Top 5 Final Results
"""

from hybrid_retrieval import hybrid_retrieve
from reranker import rerank


RERANK_CANDIDATES = 10
FINAL_TOP_K = 5


def reranked_retrieve(
    query: str,
    top_k: int = FINAL_TOP_K,
):
    """
    Hybrid retrieval followed by multilingual reranking.
    """

    # Get a larger candidate pool first.
    candidates = hybrid_retrieve(
        query,
        top_k=RERANK_CANDIDATES,
    )

    # Rerank candidates using the multilingual
    # cross-encoder.
    results = rerank(
        query,
        candidates,
        top_k=top_k,
    )

    return results


if __name__ == "__main__":

    import sys

    query = (
        " ".join(sys.argv[1:])
        or "साइरीन क्या है"
    )

    print("\n" + "=" * 70)
    print("HYBRID + RERANKER")
    print("=" * 70)

    print(f"\nQuery: {query}")

    print("\nRetrieving hybrid candidates...")

    results = reranked_retrieve(
        query,
        top_k=FINAL_TOP_K,
    )

    if not results:
        print("\nNo results found.")
        raise SystemExit(0)

    print("\nFinal reranked results:")
    print("-" * 70)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        strategy = result[
            "metadata"
        ].get(
            "strategy",
            "unknown",
        )

        print(
            f"\n#{rank} "
            f"[{strategy}] "
            f"reranker="
            f"{result['reranker_score']:.4f}"
        )

        print(
            result["text"][:500]
        )
