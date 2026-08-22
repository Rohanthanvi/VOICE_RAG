"""
Day 4 — Hybrid Retrieval
Hacker House Goa — RAG_GOA_V2

Dense retrieval:
    EmbeddingGemma-300M + ChromaDB

Lexical retrieval:
    BM25

Fusion:
    Reciprocal Rank Fusion (RRF)

The existing ChromaDB index is NOT modified.
"""

from pathlib import Path
import json
import math
import re

import numpy as np


# =========================================================
# PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = PROJECT_DIR / "chroma_db"

COLLECTION_NAME = "msmarco_hi_chunks"

CORPUS_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "chunks_hi.jsonl"
)

MODEL_NAME = "google/embeddinggemma-300m"


# =========================================================
# CONFIGURATION
# =========================================================

DENSE_K = 20
BM25_K = 20
FINAL_K = 5

RRF_K = 60

DENSE_WEIGHT = 1.0
BM25_WEIGHT = 1.0


# =========================================================
# GLOBAL OBJECTS
# =========================================================

_model = None
_collection = None
_bm25 = None

_corpus_documents = []
_corpus_metadata = []
_corpus_ids = []


# =========================================================
# TEXT TOKENIZATION
# =========================================================

def tokenize(text: str) -> list[str]:
    """
    Tokenize Hindi/English text while preserving Unicode words.
    """

    text = text.lower()

    return re.findall(
        r"[\w\u0900-\u097F]+",
        text,
        flags=re.UNICODE,
    )


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

def get_model():

    global _model

    if _model is None:

        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(
            MODEL_NAME,
            model_kwargs={
                "torch_dtype": "float32"
            },
        )

    return _model


# =========================================================
# LOAD CHROMADB
# =========================================================

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


# =========================================================
# LOAD CORPUS FOR BM25
# =========================================================

def load_corpus():

    global _corpus_documents
    global _corpus_metadata
    global _corpus_ids

    if _corpus_documents:
        return

    print(
        f"Loading lexical corpus from:\n"
        f"{CORPUS_FILE}"
    )

    with open(
        CORPUS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            _corpus_ids.append(
                item["chunk_id"]
            )

            _corpus_documents.append(
                item["text"]
            )

            _corpus_metadata.append(
                item
            )

    print(
        f"Loaded "
        f"{len(_corpus_documents):,} "
        f"documents for BM25."
    )


# =========================================================
# BUILD BM25
# =========================================================

def get_bm25():

    global _bm25

    if _bm25 is None:

        load_corpus()

        try:

            from rank_bm25 import BM25Okapi

        except ImportError:

            raise ImportError(
                "\nrank_bm25 is not installed.\n"
                "Run:\n\n"
                "pip install rank-bm25\n"
            )

        tokenized_corpus = [
            tokenize(text)
            for text in _corpus_documents
        ]

        print(
            "Building BM25 index..."
        )

        _bm25 = BM25Okapi(
            tokenized_corpus
        )

        print(
            "BM25 index ready."
        )

    return _bm25


# =========================================================
# NORMALIZE TEXT
# =========================================================

def normalize_text(text: str) -> str:

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[^\w\s\u0900-\u097F]",
        "",
        text,
    )

    return text.strip()


# =========================================================
# DEDUPLICATION
# =========================================================

def deduplicate(results):

    seen = set()

    output = []

    for result in results:

        normalized = normalize_text(
            result["text"]
        )

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)

        output.append(result)

    return output


# =========================================================
# DENSE RETRIEVAL
# =========================================================

def dense_retrieve(
    query: str,
    top_k: int = DENSE_K,
    strategy_filter: str | None = None,
):

    model = get_model()

    collection = get_collection()

    query_embedding = model.encode_query(
        query.strip(),
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    if not np.isfinite(
        query_embedding
    ).all():

        raise ValueError(
            "Query embedding contains "
            "NaN or Inf values."
        )

    where = None

    if strategy_filter:

        where = {
            "strategy": strategy_filter
        }

    collection_count = (
        collection.count()
    )

    top_k = min(
        top_k,
        collection_count,
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        where=where,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    output = []

    for i in range(
        len(results["ids"][0])
    ):

        distance = results[
            "distances"
        ][0][i]

        if not math.isfinite(
            distance
        ):
            continue

        output.append(
            {
                "chunk_id": results[
                    "ids"
                ][0][i],

                "text": results[
                    "documents"
                ][0][i],

                "metadata": results[
                    "metadatas"
                ][0][i],

                "distance": float(
                    distance
                ),
            }
        )

    return deduplicate(output)


# =========================================================
# BM25 RETRIEVAL
# =========================================================

def bm25_retrieve(
    query: str,
    top_k: int = BM25_K,
):

    bm25 = get_bm25()

    query_tokens = tokenize(
        query
    )

    scores = bm25.get_scores(
        query_tokens
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        score = float(
            scores[index]
        )

        if not math.isfinite(
            score
        ):
            continue

        results.append(
            {
                "chunk_id":
                    _corpus_ids[index],

                "text":
                    _corpus_documents[index],

                "metadata":
                    _corpus_metadata[index],

                "bm25_score":
                    score,
            }
        )

    return deduplicate(
        results
    )


# =========================================================
# RRF
# =========================================================

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    final_k=FINAL_K,
):

    fused = {}

    # -----------------------------------------------------
    # Dense ranking
    # -----------------------------------------------------

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):

        chunk_id = result[
            "chunk_id"
        ]

        if chunk_id not in fused:

            fused[chunk_id] = {
                "chunk_id": chunk_id,
                "text": result["text"],
                "metadata": result[
                    "metadata"
                ],
                "rrf_score": 0.0,
            }

        fused[
            chunk_id
        ]["rrf_score"] += (
            DENSE_WEIGHT
            / (RRF_K + rank)
        )

    # -----------------------------------------------------
    # BM25 ranking
    # -----------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):

        chunk_id = result[
            "chunk_id"
        ]

        if chunk_id not in fused:

            fused[chunk_id] = {
                "chunk_id": chunk_id,
                "text": result["text"],
                "metadata": result[
                    "metadata"
                ],
                "rrf_score": 0.0,
            }

        fused[
            chunk_id
        ]["rrf_score"] += (
            BM25_WEIGHT
            / (RRF_K + rank)
        )

    # -----------------------------------------------------
    # Sort by RRF score
    # -----------------------------------------------------

    ranked = sorted(
        fused.values(),
        key=lambda x: x[
            "rrf_score"
        ],
        reverse=True,
    )

    return ranked[:final_k]


# =========================================================
# HYBRID RETRIEVAL
# =========================================================

def hybrid_retrieve(
    query: str,
    top_k: int = FINAL_K,
    strategy_filter: str | None = None,
):

    dense_results = dense_retrieve(
        query,
        top_k=DENSE_K,
        strategy_filter=strategy_filter,
    )

    bm25_results = bm25_retrieve(
        query,
        top_k=BM25_K,
    )

    fused_results = (
        reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            final_k=top_k,
        )
    )

    return fused_results


# =========================================================
# CLI TEST
# =========================================================

if __name__ == "__main__":

    import sys

    query = (
        " ".join(sys.argv[1:])
        or "साइरीन क्या है"
    )

    print("\n" + "=" * 70)

    print(
        "HACKER HOUSE GOA — HYBRID RETRIEVAL"
    )

    print("=" * 70)

    print(
        f"\nQuery: {query}"
    )

    print(
        "\nRunning dense retrieval..."
    )

    dense = dense_retrieve(
        query,
        top_k=DENSE_K,
    )

    print(
        f"Dense candidates: "
        f"{len(dense)}"
    )

    print(
        "\nRunning BM25..."
    )

    bm25 = bm25_retrieve(
        query,
        top_k=BM25_K,
    )

    print(
        f"BM25 candidates: "
        f"{len(bm25)}"
    )

    print(
        "\nRunning RRF fusion..."
    )

    results = reciprocal_rank_fusion(
        dense,
        bm25,
        final_k=FINAL_K,
    )

    print(
        "\n" + "-" * 70
    )

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
            f"RRF={result['rrf_score']:.6f}"
        )

        print(
            result["text"][:500]
        )

    print(
        "\n" + "=" * 70
    )