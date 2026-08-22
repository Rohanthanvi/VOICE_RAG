"""
Day 3 (part 2) — Retrieval

Retrieves relevant Hindi passages from ChromaDB using
EmbeddingGemma-300M.

Features:
- FP32 inference to avoid NaN embeddings.
- encode_query() for query embeddings.
- Normalized embeddings.
- ChromaDB similarity search.
- Candidate over-fetching.
- Exact + near-duplicate removal.
- Strategy filtering.
- Safe embedding validation.
"""

from pathlib import Path
import math
import re

import numpy as np


CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "msmarco_hi_chunks"
MODEL_NAME = "google/embeddinggemma-300m"

_model = None
_collection = None


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

def get_model():
    global _model

    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(
            MODEL_NAME,
            model_kwargs={"torch_dtype": "float32"},
        )

    return _model


# ---------------------------------------------------------
# CHROMA
# ---------------------------------------------------------

def get_collection():
    global _collection

    if _collection is None:
        import chromadb

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        _collection = client.get_collection(
            name=COLLECTION_NAME
        )

    return _collection


# ---------------------------------------------------------
# WARMUP
# ---------------------------------------------------------

def warmup() -> None:
    """
    Force model and ChromaDB collection to load.
    """
    get_model()
    get_collection()


# ---------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for duplicate detection.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\u0900-\u097F]", "", text)
    return text.strip()


# ---------------------------------------------------------
# DUPLICATE DETECTION
# ---------------------------------------------------------

def is_duplicate(
    text: str,
    seen_texts: set[str],
    similarity_threshold: float = 0.92,
) -> bool:
    """
    Detect exact and near-duplicate passages.

    Exact duplicates are removed using normalized text.

    Near duplicates are detected using token Jaccard similarity.
    """

    normalized = normalize_text(text)

    if not normalized:
        return True

    # Exact duplicate
    if normalized in seen_texts:
        return True

    # Token-level near duplicate detection
    current_tokens = set(normalized.split())

    if not current_tokens:
        return True

    for previous in seen_texts:

        previous_tokens = set(previous.split())

        if not previous_tokens:
            continue

        intersection = len(
            current_tokens & previous_tokens
        )

        union = len(
            current_tokens | previous_tokens
        )

        if union == 0:
            continue

        similarity = intersection / union

        if similarity >= similarity_threshold:
            return True

    return False


# ---------------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------------

def retrieve(
    query_text: str,
    top_k: int = 5,
    strategy_filter: str | None = None,
) -> list[dict]:
    """
    Return the top_k unique and relevant chunks.

    ChromaDB retrieves extra candidates first so that
    duplicate passages can be removed without reducing
    the final number of results.
    """

    if not query_text or not query_text.strip():
        raise ValueError("Query cannot be empty.")

    model = get_model()
    collection = get_collection()

    # -----------------------------------------------------
    # 1. Query embedding
    # -----------------------------------------------------

    query_embedding = model.encode_query(
        query_text.strip(),
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # -----------------------------------------------------
    # 2. Validate embedding
    # -----------------------------------------------------

    if np.isnan(query_embedding).any():
        raise ValueError(
            "Query embedding contains NaN values."
        )

    if np.isinf(query_embedding).any():
        raise ValueError(
            "Query embedding contains Inf values."
        )

    norm = np.linalg.norm(query_embedding)

    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(
            f"Invalid query embedding norm: {norm}"
        )

    query_embedding = query_embedding.tolist()

    # -----------------------------------------------------
    # 3. Metadata filter
    # -----------------------------------------------------

    where = None

    if strategy_filter:
        where = {
            "strategy": strategy_filter
        }

    # -----------------------------------------------------
    # 4. Over-fetch candidates
    # -----------------------------------------------------
    #
    # If top_k = 5, retrieve 20 candidates.
    # This gives us enough candidates after deduplication.
    #

    candidate_k = max(top_k * 4, 20)

    # Do not ask Chroma for more than the collection contains.
    collection_count = collection.count()

    candidate_k = min(
        candidate_k,
        collection_count
    )

    if candidate_k == 0:
        return []

    # -----------------------------------------------------
    # 5. ChromaDB search
    # -----------------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        where=where,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # -----------------------------------------------------
    # 6. Deduplicate
    # -----------------------------------------------------

    chunks = []
    seen_texts = set()

    for i in range(len(ids)):

        distance = distances[i]

        # Ignore invalid distances
        if not math.isfinite(distance):
            continue

        text = documents[i]

        # Remove exact/near duplicates
        if is_duplicate(
            text,
            seen_texts,
            similarity_threshold=0.92,
        ):
            continue

        normalized = normalize_text(text)
        seen_texts.add(normalized)

        chunks.append(
            {
                "chunk_id": ids[i],
                "text": text,
                "metadata": metadatas[i],
                "distance": float(distance),
            }
        )

        # Stop once we have enough unique results
        if len(chunks) >= top_k:
            break

    return chunks


# ---------------------------------------------------------
# COMMAND-LINE TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    import sys

    query = (
        " ".join(sys.argv[1:])
        or "भारत की राजधानी क्या है"
    )

    print(f"\nQuery: {query}")
    print("=" * 70)

    try:

        results = retrieve(
            query,
            top_k=5,
        )

        if not results:
            print("No valid retrieval results found.")
            sys.exit(0)

        for rank, chunk in enumerate(
            results,
            start=1,
        ):

            strategy = chunk[
                "metadata"
            ].get(
                "strategy",
                "unknown",
            )

            print(
                f"\n#{rank} "
                f"[{strategy}] "
                f"dist={chunk['distance']:.4f}"
            )

            print(
                chunk["text"][:500]
            )

    except Exception as e:

        print(
            f"\nRetrieval error: "
            f"{type(e).__name__}: {e}"
        )

        raise