"""
Hacker House Goa — Production E5 Retrieval

Fast semantic retrieval using:
    intfloat/multilingual-e5-small

Production index:
    chroma_db_e5
    msmarco_hi_chunks_e5

Important:
- Model is loaded only once.
- Chroma collection is loaded only once.
- E5 uses the required "query:" prefix.
- Returns distance + confidence information.
- Designed for the fast path in harness.py.

Run:
    python src/retrieval_e5.py "साइरीन क्या है"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = PROJECT_DIR / "chroma_db_e5"

COLLECTION_NAME = "msmarco_hi_chunks_e5"

MODEL_NAME = "intfloat/multilingual-e5-small"


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

DEFAULT_TOP_K = 8

# E5 distance is cosine distance because embeddings are normalized.
#
# Lower distance = better match.
#
# This threshold is intentionally conservative.
# We will tune it using benchmark results.
FAST_PATH_DISTANCE = 0.35


# ============================================================
# CACHED COMPONENTS
# ============================================================

_model: SentenceTransformer | None = None
_collection = None


# ============================================================
# LOAD MODEL
# ============================================================

def get_model() -> SentenceTransformer:

    global _model

    if _model is None:

        print(
            f"Loading E5 model: {MODEL_NAME}"
        )

        _model = SentenceTransformer(
            MODEL_NAME
        )

        print("E5 model loaded.")

    return _model


# ============================================================
# LOAD CHROMA
# ============================================================

def get_collection():

    global _collection

    if _collection is None:

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        _collection = client.get_collection(
            name=COLLECTION_NAME
        )

        print(
            f"E5 collection loaded: "
            f"{_collection.count():,} chunks"
        )

    return _collection


# ============================================================
# WARMUP
# ============================================================

def warmup() -> None:
    """
    Load the model and Chroma collection into memory.

    Call once when the application starts.
    """

    print("=" * 70)
    print("E5 RETRIEVAL WARMUP")
    print("=" * 70)

    t0 = time.perf_counter()

    model = get_model()
    collection = get_collection()

    # Real embedding warmup.
    model.encode(
        ["query: साइरीन क्या है"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Real Chroma query warmup.
    query_embedding = model.encode(
        ["query: साइरीन क्या है"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()

    collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    elapsed = (
        time.perf_counter() - t0
    ) * 1000

    print(
        f"Warmup complete: {elapsed:.1f} ms"
    )

    print(
        "E5 model and Chroma remain loaded."
    )


# ============================================================
# CONFIDENCE
# ============================================================

def distance_to_confidence(
    distance: float,
) -> float:
    """
    Convert Chroma cosine distance into a simple
    normalized confidence score.

    This is NOT a calibrated probability.

    1.0 = very close
    0.0 = weak match
    """

    if distance is None:
        return 0.0

    # Conservative linear mapping.
    confidence = 1.0 - (
        distance / 1.0
    )

    return max(
        0.0,
        min(1.0, confidence),
    )


def is_confident(
    distance: float,
) -> bool:
    """
    Fast-path decision.

    True:
        E5 result is strong enough for direct
        extractive answering.

    False:
        harness should use BM25 fallback.
    """

    return (
        distance <= FAST_PATH_DISTANCE
    )


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:

    if not query or not query.strip():
        return []

    query = query.strip()

    model = get_model()
    collection = get_collection()

    # --------------------------------------------------------
    # E5 QUERY FORMAT
    # --------------------------------------------------------

    query_text = f"query: {query}"

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()

    # --------------------------------------------------------
    # CHROMA SEARCH
    # --------------------------------------------------------

    result = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = (
        result.get("documents", [[]])[0]
    )

    metadatas = (
        result.get("metadatas", [[]])[0]
    )

    distances = (
        result.get("distances", [[]])[0]
    )

    ids = (
        result.get("ids", [[]])[0]
    )

    output = []

    for i, document in enumerate(
        documents
    ):

        distance = float(
            distances[i]
        ) if i < len(distances) else 1.0

        metadata = (
            metadatas[i]
            if i < len(metadatas)
            else {}
        )

        chunk_id = (
            ids[i]
            if i < len(ids)
            else metadata.get(
                "chunk_id",
                "",
            )
        )

        output.append(
            {
                "chunk_id": chunk_id,

                "text": document,

                "metadata": metadata,

                "distance": distance,

                "confidence":
                    distance_to_confidence(
                        distance
                    ),

                "e5_rank": i + 1,
            }
        )

    return output


# ============================================================
# FAST RETRIEVAL
# ============================================================

def retrieve_fast(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """
    Retrieval helper used by the production harness.

    Returns both the retrieved results and the decision
    about whether the query is suitable for the fast path.
    """

    t0 = time.perf_counter()

    results = retrieve(
        query=query,
        top_k=top_k,
    )

    elapsed_ms = (
        time.perf_counter() - t0
    ) * 1000

    if results:

        best_distance = results[0][
            "distance"
        ]

        confidence = results[0][
            "confidence"
        ]

    else:

        best_distance = 1.0
        confidence = 0.0

    return {
        "results": results,

        "best_distance":
            best_distance,

        "confidence":
            confidence,

        "fast_path":
            is_confident(
                best_distance
            ),

        "latency_ms":
            elapsed_ms,
    }


# ============================================================
# CLI
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            'python src/retrieval_e5.py '
            '"साइरीन क्या है"'
        )

        sys.exit(1)

    query = " ".join(
        sys.argv[1:]
    )

    print("=" * 75)
    print(
        "HACKER HOUSE GOA — "
        "PRODUCTION E5 RETRIEVAL"
    )
    print("=" * 75)

    print(
        f"\nQuery: {query}"
    )

    print(
        "\nRunning E5 retrieval..."
    )

    t0 = time.perf_counter()

    results = retrieve(
        query,
        top_k=DEFAULT_TOP_K,
    )

    elapsed = (
        time.perf_counter() - t0
    ) * 1000

    print(
        f"\nRetrieval latency: "
        f"{elapsed:.1f} ms"
    )

    print(
        "\nResults:"
    )

    print("-" * 75)

    for i, item in enumerate(
        results[:5],
        start=1,
    ):

        print(
            f"\n#{i} "
            f"[{item['metadata'].get('strategy', '')}] "
            f"distance={item['distance']:.4f} "
            f"confidence={item['confidence']:.3f}"
        )

        print(
            item["text"][:500]
        )

    if results:

        best = results[0]

        print(
            "\n" + "=" * 75
        )

        print(
            "FAST PATH DECISION"
        )

        print(
            f"Best distance : "
            f"{best['distance']:.4f}"
        )

        print(
            f"Confidence    : "
            f"{best['confidence']:.3f}"
        )

        if is_confident(
            best["distance"]
        ):

            print(
                "Decision      : "
                "FAST EXTRACTIVE PATH"
            )

        else:

            print(
                "Decision      : "
                "BM25 FALLBACK"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()