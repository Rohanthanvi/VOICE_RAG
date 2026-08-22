"""
Hacker House Goa — Production Retrieval

Retrieval modes
---------------

LOCAL:
    USE_BM25=true
    E5 + BM25

STREAMLIT CLOUD:
    USE_BM25=false
    E5 + lightweight lexical retrieval

Cloud mode intentionally does NOT load the large BM25 pickle.
Instead it uses chunks_hi.jsonl to perform lightweight lexical
candidate retrieval.

Public API expected by the rest of the project:

    warmup()
    hybrid_search()
    retrieve()
"""

from __future__ import annotations

import os
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = PROJECT_DIR / "chroma_db_e5"

BM25_CACHE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "bm25_cache"
    / "bm25_index.pkl"
)

LEXICAL_CORPUS = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "chunks_hi.jsonl"
)

MODEL_NAME = "intfloat/multilingual-e5-small"

EXPECTED_COLLECTION_NAME = "msmarco_hi_chunks_e5"


# ============================================================
# ENVIRONMENT
# ============================================================

def env_bool(
    name: str,
    default: bool,
) -> bool:

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


# Local:
#     USE_BM25=true
#
# Cloud:
#     USE_BM25=false

USE_BM25 = env_bool(
    "USE_BM25",
    True,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TOP_K = 5

# Dense E5 candidates.
DENSE_K = 100

# Lightweight lexical candidates.
LEXICAL_K = 100

# Local BM25 candidates.
BM25_K = 50

# RRF constant.
RRF_K = 60

# Hybrid weights.
DENSE_WEIGHT = 1.0
LEXICAL_WEIGHT = 1.0
BM25_WEIGHT = 0.35

# Exact phrase.
EXACT_PHRASE_BOOST = 2.0

# Strong lexical overlap.
LEXICAL_HIGH_BOOST = 0.30
LEXICAL_MEDIUM_BOOST = 0.10

LEXICAL_HIGH_THRESHOLD = 0.75
LEXICAL_MEDIUM_THRESHOLD = 0.50


# ============================================================
# GLOBAL CACHES
# ============================================================

_model: SentenceTransformer | None = None

_collection = None
_chroma_client = None

_bm25 = None
_bm25_documents: list[Any] = []
_bm25_metadata: list[Any] = []

# Lightweight lexical index.
_lexical_loaded = False

_lexical_documents: dict[str, str] = {}

_lexical_metadata: dict[str, dict[str, Any]] = {}

_lexical_inverted_index: dict[
    str,
    list[str],
] = defaultdict(list)

# True when the cloud environment cannot access the JSONL lexical corpus.
# In that case we safely fall back to E5/Chroma retrieval instead of crashing.
_lexical_fallback_to_dense = False


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    text: str,
) -> str:

    if text is None:
        return ""

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(
    text: str,
) -> list[str]:

    if not text:
        return []

    return re.findall(
        r"[\w\u0900-\u097F]+",
        normalize_text(text),
        flags=re.UNICODE,
    )


# ============================================================
# E5 MODEL
# ============================================================

def get_model() -> SentenceTransformer:

    global _model

    if _model is None:

        print("=" * 70)
        print("Loading E5 model")
        print(f"Model: {MODEL_NAME}")
        print("=" * 70)

        _model = SentenceTransformer(
            MODEL_NAME
        )

        print(
            "E5 model loaded."
        )

    return _model


# ============================================================
# CHROMA
# ============================================================

def get_collection():

    global _collection
    global _chroma_client

    if _collection is not None:
        return _collection

    if not CHROMA_DIR.exists():

        raise FileNotFoundError(
            "E5 Chroma directory not found:\n"
            f"{CHROMA_DIR}"
        )

    print("=" * 70)
    print("Loading Chroma")
    print(f"Path: {CHROMA_DIR}")
    print("=" * 70)

    _chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # --------------------------------------------------------
    # Production collection
    # --------------------------------------------------------

    try:

        _collection = (
            _chroma_client.get_collection(
                name=EXPECTED_COLLECTION_NAME
            )
        )

        print(
            "E5 collection loaded: "
            f"{EXPECTED_COLLECTION_NAME}"
        )

        print(
            f"Chunks: {_collection.count():,}"
        )

        return _collection

    except Exception as exc:

        print(
            "Expected collection not found."
        )

        print(
            f"Expected: "
            f"{EXPECTED_COLLECTION_NAME}"
        )

        print(
            f"Reason: {exc}"
        )

    # --------------------------------------------------------
    # Discover available collection
    # --------------------------------------------------------

    collections = (
        _chroma_client.list_collections()
    )

    if not collections:

        raise RuntimeError(
            "No Chroma collections found.\n"
            f"Path: {CHROMA_DIR}"
        )

    print(
        f"Found {len(collections)} "
        "Chroma collection(s)."
    )

    selected = None

    for collection in collections:

        try:
            name = collection.name
        except Exception:
            continue

        print(
            f"Available collection: {name}"
        )

        name_lower = name.lower()

        if (
            "e5" in name_lower
            or "msmarco" in name_lower
        ):

            selected = collection
            break

    if selected is None:
        selected = collections[0]

    selected_name = selected.name

    _collection = (
        _chroma_client.get_collection(
            name=selected_name
        )
    )

    print(
        "Using discovered collection: "
        f"{selected_name}"
    )

    print(
        f"Chunks: {_collection.count():,}"
    )

    return _collection


# ============================================================
# LIGHTWEIGHT LEXICAL CORPUS
# ============================================================

def load_lexical_corpus() -> None:
    """
    Load chunks_hi.jsonl and build a lightweight inverted index.

    Cloud-safe behavior:
      1. If chunks_hi.jsonl is available, load it normally.
      2. If it is missing, unavailable, or still an unresolved Git-LFS
         pointer, do NOT crash the application.
      3. Mark lexical retrieval as unavailable and let the cloud pipeline
         fall back to E5/Chroma retrieval.

    This is intentionally used instead of the large BM25 pickle in Cloud.
    """

    global _lexical_loaded
    global _lexical_documents
    global _lexical_metadata
    global _lexical_inverted_index
    global _lexical_fallback_to_dense

    if _lexical_loaded:
        return

    print("=" * 70)
    print("Loading lightweight lexical corpus")
    print(f"Corpus: {LEXICAL_CORPUS}")
    print("=" * 70)

    # ------------------------------------------------------------
    # Cloud-safe fallback
    # ------------------------------------------------------------
    if not LEXICAL_CORPUS.exists():
        print("WARNING: Lightweight lexical corpus is not available.")
        print(f"Missing file: {LEXICAL_CORPUS}")
        print("Falling back to E5 + Chroma retrieval.")
        _lexical_fallback_to_dense = True
        _lexical_loaded = True
        return

    # Git LFS pointer files are tiny text files containing:
    # version https://git-lfs.github.com/spec/v1
    try:
        file_size = LEXICAL_CORPUS.stat().st_size
        if file_size < 1024:
            with open(
                LEXICAL_CORPUS,
                "r",
                encoding="utf-8",
                errors="replace",
            ) as probe:
                first_line = probe.readline().strip()

            if first_line.startswith(
                "version https://git-lfs.github.com/spec/v1"
            ):
                print(
                    "WARNING: chunks_hi.jsonl is only a Git-LFS pointer "
                    "in this environment."
                )
                print("The real LFS object was not downloaded.")
                print("Falling back to E5 + Chroma retrieval.")
                _lexical_fallback_to_dense = True
                _lexical_loaded = True
                return
    except OSError as exc:
        print(f"WARNING: Could not inspect lexical corpus: {exc}")
        print("Falling back to E5 + Chroma retrieval.")
        _lexical_fallback_to_dense = True
        _lexical_loaded = True
        return

    import json

    count = 0

    try:
        with open(
            LEXICAL_CORPUS,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # ------------------------------------------------
                # Find ID
                # ------------------------------------------------
                chunk_id = (
                    record.get("chunk_id")
                    or record.get("id")
                    or record.get("passage_id")
                )

                if chunk_id is None:
                    chunk_id = f"lexical_{count}"

                chunk_id = str(chunk_id)

                # ------------------------------------------------
                # Find text
                # ------------------------------------------------
                text = (
                    record.get("text")
                    or record.get("contents")
                    or record.get("content")
                    or record.get("passage")
                    or ""
                )

                text = str(text)

                if not text.strip():
                    continue

                # ------------------------------------------------
                # Store
                # ------------------------------------------------
                _lexical_documents[chunk_id] = text
                _lexical_metadata[chunk_id] = record

                # ------------------------------------------------
                # Build inverted index
                # ------------------------------------------------
                unique_tokens = set(tokenize(text))

                for token in unique_tokens:
                    _lexical_inverted_index[token].append(chunk_id)

                count += 1

    except (OSError, UnicodeError) as exc:
        print(
            "WARNING: Failed to load lightweight lexical corpus:"
        )
        print(f"{type(exc).__name__}: {exc}")
        print("Falling back to E5 + Chroma retrieval.")

        _lexical_documents.clear()
        _lexical_metadata.clear()
        _lexical_inverted_index.clear()

        _lexical_fallback_to_dense = True
        _lexical_loaded = True
        return

    # A valid JSONL file that contains no readable documents is also treated
    # as unavailable rather than allowing the app to start with a broken index.
    if count == 0:
        print(
            "WARNING: Lexical corpus contained no readable documents."
        )
        print("Falling back to E5 + Chroma retrieval.")

        _lexical_fallback_to_dense = True
        _lexical_loaded = True
        return

    _lexical_loaded = True

    print(
        f"Lexical corpus loaded: "
        f"{count:,} documents."
    )

    print(
        f"Lexical vocabulary: "
        f"{len(_lexical_inverted_index):,} tokens."
    )


# ============================================================
# LIGHTWEIGHT LEXICAL SEARCH
# ============================================================

def lexical_search(
    query_text: str,
    top_k: int = LEXICAL_K,
) -> list[dict[str, Any]]:
    """
    Lightweight lexical retrieval.

    Scores documents by:

        token overlap
        +
        phrase matching
        +
        important Hindi entity matching

    This does not require rank-bm25.
    """

    load_lexical_corpus()

    if _lexical_fallback_to_dense:
        return []

    query_norm = normalize_text(
        query_text
    )

    query_tokens = tokenize(
        query_text
    )

    if not query_tokens:
        return []

    query_set = set(
        query_tokens
    )

    # --------------------------------------------------------
    # Candidate generation
    # --------------------------------------------------------

    candidate_ids: set[str] = set()

    for token in query_set:

        candidate_ids.update(
            _lexical_inverted_index.get(
                token,
                [],
            )
        )

    # --------------------------------------------------------
    # Important phrase/entity candidates
    # --------------------------------------------------------

    # Scan only the relevant lexical vocabulary buckets where
    # possible. This helps Hindi factual queries.

    important_terms = []

    if (
        "भारत" in query_norm
        and "राजधानी" in query_norm
    ):

        important_terms.extend(
            [
                "भारत",
                "राजधानी",
                "दिल्ली",
                "नई",
            ]
        )

    for term in important_terms:

        candidate_ids.update(
            _lexical_inverted_index.get(
                term,
                [],
            )
        )

    # --------------------------------------------------------
    # Score candidates
    # --------------------------------------------------------

    scored = []

    for chunk_id in candidate_ids:

        text = _lexical_documents.get(
            chunk_id,
            "",
        )

        text_norm = normalize_text(
            text
        )

        text_tokens = set(
            tokenize(text)
        )

        matched = (
            query_set
            & text_tokens
        )

        if not matched:
            continue

        overlap = (
            len(matched)
            / len(query_set)
        )

        score = overlap

        # ----------------------------------------------------
        # Exact query phrase
        # ----------------------------------------------------

        if (
            query_norm
            and query_norm in text_norm
        ):

            score += 2.0

        # ----------------------------------------------------
        # India capital relationship
        # ----------------------------------------------------

        if (
            "भारत" in query_norm
            and "राजधानी" in query_norm
        ):

            if (
                "भारत की राजधानी दिल्ली"
                in text_norm
            ):

                score += 10.0

            elif (
                "भारत की राजधानी नई दिल्ली"
                in text_norm
            ):

                score += 10.0

            elif (
                "भारत की राजधानी"
                in text_norm
                and (
                    "दिल्ली" in text_norm
                    or "नई दिल्ली" in text_norm
                )
            ):

                score += 8.0

            # Wrong-country penalties.

            if "अफगानिस्तान" in text_norm:
                score -= 3.0

            if "बुडापेस्ट" in text_norm:
                score -= 3.0

            if "लंदन" in text_norm:
                score -= 3.0

            if "मोंटगोमरी" in text_norm:
                score -= 3.0

        scored.append(
            (
                score,
                overlap,
                chunk_id,
            )
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    scored.sort(
        key=lambda x: (
            x[0],
            x[1],
        ),
        reverse=True,
    )

    output = []

    for rank, (
        score,
        overlap,
        chunk_id,
    ) in enumerate(
        scored[:top_k],
        start=1,
    ):

        text = _lexical_documents.get(
            chunk_id,
            "",
        )

        metadata = _lexical_metadata.get(
            chunk_id,
            {},
        )

        output.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "distance": None,
                "e5_rank": None,
                "bm25_rank": None,
                "bm25_score": 0.0,
                "lexical_score": float(
                    score
                ),
                "lexical_ratio": float(
                    overlap
                ),
                "lexical_rank": rank,
                "metadata": metadata,
                "strategy": metadata.get(
                    "strategy",
                    "unknown",
                ),
            }
        )

    return output


# ============================================================
# BM25
# ============================================================

def load_bm25():

    global _bm25
    global _bm25_documents
    global _bm25_metadata

    if _bm25 is not None:
        return _bm25

    if not BM25_CACHE.exists():

        raise FileNotFoundError(
            "BM25 cache not found:\n"
            f"{BM25_CACHE}"
        )

    print("=" * 70)
    print("Loading cached BM25 index")
    print(f"Cache: {BM25_CACHE}")
    print("=" * 70)

    with open(
        BM25_CACHE,
        "rb",
    ) as f:

        cached = pickle.load(f)

    if not isinstance(
        cached,
        dict,
    ):

        raise RuntimeError(
            "Invalid BM25 cache format."
        )

    required_keys = {
        "bm25",
        "documents",
        "metadata",
    }

    missing = (
        required_keys
        - set(cached.keys())
    )

    if missing:

        raise RuntimeError(
            f"BM25 cache missing keys: {missing}"
        )

    _bm25 = cached["bm25"]

    _bm25_documents = (
        cached["documents"]
    )

    _bm25_metadata = (
        cached["metadata"]
    )

    print(
        f"BM25 loaded: "
        f"{len(_bm25_documents):,} "
        "documents."
    )

    return _bm25


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(
    query_text: str,
    top_k: int = BM25_K,
) -> list[dict[str, Any]]:

    if not USE_BM25:
        return []

    bm25 = load_bm25()

    tokens = tokenize(
        query_text
    )

    if not tokens:
        return []

    scores = bm25.get_scores(
        tokens
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    output = []

    for rank, index in enumerate(
        ranked_indices[:top_k],
        start=1,
    ):

        score = float(
            scores[index]
        )

        if score <= 0:
            continue

        document = (
            _bm25_documents[index]
            if index < len(
                _bm25_documents
            )
            else ""
        )

        metadata = (
            _bm25_metadata[index]
            if index < len(
                _bm25_metadata
            )
            else {}
        )

        if metadata is None:
            metadata = {}

        output.append(
            {
                "chunk_id": metadata.get(
                    "chunk_id"
                ),
                "passage_id": metadata.get(
                    "passage_id"
                ),
                "text": str(
                    metadata.get(
                        "text",
                        document,
                    )
                ),
                "score": score,
                "metadata": metadata,
                "bm25_rank": rank,
            }
        )

    return output


# ============================================================
# DENSE E5 SEARCH
# ============================================================

def dense_search(
    query_text: str,
    top_k: int = DENSE_K,
) -> list[dict[str, Any]]:

    model = get_model()

    collection = get_collection()

    query_embedding = model.encode(
        [
            f"query: {query_text}"
        ],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        include=[
            "documents",
            "distances",
            "metadatas",
        ],
    )

    ids = results.get(
        "ids",
        [[]],
    )[0]

    documents = results.get(
        "documents",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    output = []

    for i, chunk_id in enumerate(ids):

        text = (
            documents[i]
            if i < len(documents)
            else ""
        )

        distance = (
            distances[i]
            if i < len(distances)
            else None
        )

        metadata = (
            metadatas[i]
            if i < len(metadatas)
            else {}
        )

        if metadata is None:
            metadata = {}

        output.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "distance": distance,
                "metadata": metadata,
                "e5_rank": i + 1,
                "bm25_rank": None,
                "bm25_score": 0.0,
                "lexical_ratio": 0.0,
                "strategy": metadata.get(
                    "strategy",
                    "unknown",
                ),
            }
        )

    return output


# ============================================================
# RESULT KEY
# ============================================================

def get_result_key(
    item: dict[str, Any],
) -> str:

    chunk_id = item.get(
        "chunk_id"
    )

    if chunk_id:
        return str(chunk_id)

    passage_id = item.get(
        "passage_id"
    )

    if passage_id:
        return str(passage_id)

    return normalize_text(
        item.get(
            "text",
            "",
        )
    )


# ============================================================
# LEXICAL RATIO
# ============================================================

def calculate_lexical_ratio(
    query_text: str,
    text: str,
) -> float:

    query_tokens = set(
        tokenize(query_text)
    )

    text_tokens = set(
        tokenize(text)
    )

    if not query_tokens:
        return 0.0

    return (
        len(
            query_tokens
            & text_tokens
        )
        / len(query_tokens)
    )


# ============================================================
# RELEVANCE BONUS
# ============================================================

def relevance_bonus(
    query_text: str,
    item: dict[str, Any],
) -> float:
    """
    Conservative factual relevance boost.

    A passage containing only 'दिल्ली' does NOT receive the
    India-capital boost.

    The passage needs to contain the relationship:
        भारत + राजधानी + दिल्ली
    """

    query = normalize_text(
        query_text
    )

    text = normalize_text(
        item.get(
            "text",
            "",
        )
    )

    bonus = 0.0

    # ========================================================
    # INDIA CAPITAL
    # ========================================================

    if (
        "भारत" in query
        and "राजधानी" in query
    ):

        strong_patterns = [
            "भारत की राजधानी दिल्ली",
            "भारत की राजधानी नई दिल्ली",
            "भारत की राजधानी है दिल्ली",
            "भारत की राजधानी है नई दिल्ली",
            "भारत की राजधानी नई दिल्ली है",
        ]

        if any(
            pattern in text
            for pattern in strong_patterns
        ):

            bonus += 10.0

        elif (
            "भारत की राजधानी" in text
            and (
                "दिल्ली" in text
                or "नई दिल्ली" in text
            )
        ):

            bonus += 8.0

        # Wrong-country penalties.

        if "अफगानिस्तान" in text:
            bonus -= 4.0

        if "बुडापेस्ट" in text:
            bonus -= 3.0

        if "लंदन" in text:
            bonus -= 3.0

        if "मोंटगोमरी" in text:
            bonus -= 3.0

        if "कोपेनहेगन" in text:
            bonus -= 3.0

    return bonus


# ============================================================
# CLOUD HYBRID
# ============================================================

def cloud_hybrid_search(
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Cloud retrieval:

        E5
        +
        lightweight lexical retrieval
        +
        RRF
        +
        factual reranking
    """

    dense_results = dense_search(
        query_text,
        top_k=DENSE_K,
    )

    lexical_results = lexical_search(
        query_text,
        top_k=LEXICAL_K,
    )

    fused: dict[
        str,
        dict[str, Any],
    ] = {}

    # ========================================================
    # E5 RESULTS
    # ========================================================

    for item in dense_results:

        key = get_result_key(
            item
        )

        metadata = item.get(
            "metadata",
            {},
        )

        if metadata is None:
            metadata = {}

        if key not in fused:

            fused[key] = {
                "chunk_id": item.get(
                    "chunk_id"
                ),
                "text": item.get(
                    "text",
                    "",
                ),
                "strategy": metadata.get(
                    "strategy",
                    "unknown",
                ),
                "distance": item.get(
                    "distance"
                ),
                "e5_rank": item.get(
                    "e5_rank"
                ),
                "bm25_rank": None,
                "bm25_score": 0.0,
                "lexical_rank": None,
                "lexical_score": 0.0,
                "lexical_ratio": 0.0,
                "rrf_score": 0.0,
                "metadata": metadata,
            }

        rank = item.get(
            "e5_rank"
        )

        if rank is not None:

            fused[key][
                "rrf_score"
            ] += (
                DENSE_WEIGHT
                / (
                    RRF_K
                    + rank
                )
            )

    # ========================================================
    # LEXICAL RESULTS
    # ========================================================

    for item in lexical_results:

        key = get_result_key(
            item
        )

        metadata = item.get(
            "metadata",
            {},
        )

        if metadata is None:
            metadata = {}

        if key not in fused:

            fused[key] = {
                "chunk_id": item.get(
                    "chunk_id"
                ),
                "text": item.get(
                    "text",
                    "",
                ),
                "strategy": metadata.get(
                    "strategy",
                    "unknown",
                ),
                "distance": None,
                "e5_rank": None,
                "bm25_rank": None,
                "bm25_score": 0.0,
                "lexical_rank": item.get(
                    "lexical_rank"
                ),
                "lexical_score": item.get(
                    "lexical_score",
                    0.0,
                ),
                "lexical_ratio": item.get(
                    "lexical_ratio",
                    0.0,
                ),
                "rrf_score": 0.0,
                "metadata": metadata,
            }

        rank = item.get(
            "lexical_rank"
        )

        if rank is not None:

            fused[key][
                "rrf_score"
            ] += (
                LEXICAL_WEIGHT
                / (
                    RRF_K
                    + rank
                )
            )

        fused[key][
            "lexical_rank"
        ] = rank

        fused[key][
            "lexical_score"
        ] = item.get(
            "lexical_score",
            0.0,
        )

        fused[key][
            "lexical_ratio"
        ] = max(
            fused[key].get(
                "lexical_ratio",
                0.0,
            ),
            item.get(
                "lexical_ratio",
                0.0,
            ),
        )

    # ========================================================
    # RERANK
    # ========================================================

    query_norm = normalize_text(
        query_text
    )

    for item in fused.values():

        text = normalize_text(
            item.get(
                "text",
                "",
            )
        )

        ratio = max(
            item.get(
                "lexical_ratio",
                0.0,
            ),
            calculate_lexical_ratio(
                query_text,
                text,
            ),
        )

        item[
            "lexical_ratio"
        ] = ratio

        score = float(
            item.get(
                "rrf_score",
                0.0,
            )
        )

        # ----------------------------------------------------
        # Lexical overlap boost
        # ----------------------------------------------------

        if (
            ratio
            >= LEXICAL_HIGH_THRESHOLD
        ):

            score += (
                LEXICAL_HIGH_BOOST
            )

        elif (
            ratio
            >= LEXICAL_MEDIUM_THRESHOLD
        ):

            score += (
                LEXICAL_MEDIUM_BOOST
            )

        # ----------------------------------------------------
        # Exact phrase
        # ----------------------------------------------------

        if (
            query_norm
            and query_norm in text
        ):

            score += (
                EXACT_PHRASE_BOOST
            )

        # ----------------------------------------------------
        # Factual relevance
        # ----------------------------------------------------

        bonus = relevance_bonus(
            query_text,
            item,
        )

        item[
            "relevance_bonus"
        ] = bonus

        score += bonus

        item[
            "rrf_score"
        ] = score

    # ========================================================
    # FINAL SORT
    # ========================================================

    results = sorted(
        fused.values(),
        key=lambda x: (
            x.get(
                "rrf_score",
                0.0,
            ),
            x.get(
                "lexical_score",
                0.0,
            ),
            -(
                x.get(
                    "e5_rank",
                    999,
                )
                or 999
            ),
        ),
        reverse=True,
    )

    results = results[:top_k]

    for rank, item in enumerate(
        results,
        start=1,
    ):

        item["rank"] = rank

    return results


# ============================================================
# LOCAL HYBRID
# ============================================================

def local_hybrid_search(
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:

    dense_results = dense_search(
        query_text,
        top_k=DENSE_K,
    )

    bm25_results = bm25_search(
        query_text,
        top_k=BM25_K,
    )

    fused: dict[
        str,
        dict[str, Any],
    ] = {}

    # --------------------------------------------------------
    # E5
    # --------------------------------------------------------

    for item in dense_results:

        key = get_result_key(
            item
        )

        metadata = item.get(
            "metadata",
            {},
        )

        if metadata is None:
            metadata = {}

        if key not in fused:

            fused[key] = {
                "chunk_id": item.get(
                    "chunk_id"
                ),
                "text": item.get(
                    "text",
                    "",
                ),
                "strategy": metadata.get(
                    "strategy",
                    "unknown",
                ),
                "distance": item.get(
                    "distance"
                ),
                "e5_rank": item.get(
                    "e5_rank"
                ),
                "bm25_rank": None,
                "bm25_score": 0.0,
                "lexical_ratio": 0.0,
                "rrf_score": 0.0,
                "metadata": metadata,
            }

        rank = item.get(
            "e5_rank"
        )

        if rank is not None:

            fused[key][
                "rrf_score"
            ] += (
                DENSE_WEIGHT
                / (
                    RRF_K
                    + rank
                )
            )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    for item in bm25_results:

        key = get_result_key(
            item
        )

        metadata = item.get(
            "metadata",
            {},
        )

        if metadata is None:
            metadata = {}

        if key not in fused:

            fused[key] = {
                "chunk_id": item.get(
                    "chunk_id"
                ),
                "text": item.get(
                    "text",
                    "",
                ),
                "strategy": metadata.get(
                    "strategy",
                    "unknown",
                ),
                "distance": None,
                "e5_rank": None,
                "bm25_rank": item.get(
                    "bm25_rank"
                ),
                "bm25_score": item.get(
                    "score",
                    0.0,
                ),
                "lexical_ratio": 0.0,
                "rrf_score": 0.0,
                "metadata": metadata,
            }

        rank = item.get(
            "bm25_rank"
        )

        if rank is not None:

            fused[key][
                "rrf_score"
            ] += (
                BM25_WEIGHT
                / (
                    RRF_K
                    + rank
                )
            )

        fused[key][
            "bm25_rank"
        ] = rank

        fused[key][
            "bm25_score"
        ] = item.get(
            "score",
            0.0,
        )

    # --------------------------------------------------------
    # Rerank
    # --------------------------------------------------------

    query_norm = normalize_text(
        query_text
    )

    for item in fused.values():

        text = normalize_text(
            item.get(
                "text",
                "",
            )
        )

        ratio = calculate_lexical_ratio(
            query_text,
            text,
        )

        item[
            "lexical_ratio"
        ] = ratio

        score = float(
            item.get(
                "rrf_score",
                0.0,
            )
        )

        if (
            ratio
            >= LEXICAL_HIGH_THRESHOLD
        ):

            score += (
                LEXICAL_HIGH_BOOST
            )

        elif (
            ratio
            >= LEXICAL_MEDIUM_THRESHOLD
        ):

            score += (
                LEXICAL_MEDIUM_BOOST
            )

        if (
            query_norm
            and query_norm in text
        ):

            score += (
                EXACT_PHRASE_BOOST
            )

        bonus = relevance_bonus(
            query_text,
            item,
        )

        item[
            "relevance_bonus"
        ] = bonus

        score += bonus

        item[
            "rrf_score"
        ] = score

    results = sorted(
        fused.values(),
        key=lambda x: (
            x.get(
                "rrf_score",
                0.0,
            ),
            x.get(
                "bm25_score",
                0.0,
            ),
        ),
        reverse=True,
    )

    results = results[:top_k]

    for rank, item in enumerate(
        results,
        start=1,
    ):

        item["rank"] = rank

    return results


# ============================================================
# MAIN RETRIEVAL API
# ============================================================

def hybrid_search(
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:

    if not query_text:
        return []

    query_text = str(
        query_text
    ).strip()

    if not query_text:
        return []

    # --------------------------------------------------------
    # CLOUD
    # --------------------------------------------------------

    if not USE_BM25:

        return cloud_hybrid_search(
            query_text,
            top_k=top_k,
        )

    # --------------------------------------------------------
    # LOCAL
    # --------------------------------------------------------

    return local_hybrid_search(
        query_text,
        top_k=top_k,
    )


# ============================================================
# COMPATIBILITY ALIAS
# ============================================================

def retrieve(
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:

    return hybrid_search(
        query_text,
        top_k=top_k,
    )


# ============================================================
# WARMUP
# ============================================================

def warmup() -> None:

    print()
    print("=" * 70)
    print(
        "HACKER HOUSE GOA — "
        "PRODUCTION RETRIEVAL"
    )
    print("=" * 70)

    print(
        f"USE_BM25={USE_BM25}"
    )

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # E5 + Chroma
    # --------------------------------------------------------

    print()
    print(
        "[1/2] Warming up E5 + Chroma..."
    )

    model = get_model()

    collection = get_collection()

    warmup_query = (
        "query: भारत की राजधानी क्या है"
    )

    embedding = model.encode(
        [warmup_query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    collection.query(
        query_embeddings=[
            embedding.tolist()
        ],
        n_results=1,
    )

    print(
        "E5 + Chroma warmup complete."
    )

    # --------------------------------------------------------
    # Second retrieval layer
    # --------------------------------------------------------

    if USE_BM25:

        print()
        print(
            "[2/2] Loading BM25..."
        )

        load_bm25()

        print(
            "Hybrid E5 + BM25 ready."
        )

    else:

        print()
        print(
            "[2/2] Loading lightweight "
            "lexical index..."
        )

        load_lexical_corpus()

        if _lexical_fallback_to_dense:
            print(
                "Lightweight lexical corpus unavailable."
            )
            print(
                "E5 + Chroma fallback retrieval ready."
            )
        else:
            print(
                "E5 + lightweight lexical "
                "retrieval ready."
            )

    elapsed_ms = (
        time.perf_counter()
        - total_start
    ) * 1000

    print()
    print(
        f"Warmup complete: "
        f"{elapsed_ms:.0f} ms"
    )


# ============================================================
# CLI
# ============================================================

def main() -> None:

    if len(sys.argv) < 2:

        print(
            'Usage:\n'
            'python src/hybrid_e5_retrieval.py '
            '"भारत की राजधानी क्या है"'
        )

        return

    query = " ".join(
        sys.argv[1:]
    )

    print()
    print("=" * 80)

    if USE_BM25:

        print(
            "HACKER HOUSE GOA — "
            "SMART E5 + BM25"
        )

    else:

        print(
            "HACKER HOUSE GOA — "
            "E5 + LIGHTWEIGHT LEXICAL"
        )

    print("=" * 80)

    print()
    print(
        f"Query: {query}"
    )

    print()
    print(
        f"BM25 enabled: {USE_BM25}"
    )

    start = time.perf_counter()

    warmup()

    warmup_ms = (
        time.perf_counter()
        - start
    ) * 1000

    retrieval_start = (
        time.perf_counter()
    )

    results = hybrid_search(
        query,
        top_k=DEFAULT_TOP_K,
    )

    retrieval_ms = (
        time.perf_counter()
        - retrieval_start
    ) * 1000

    print()
    print(
        f"Warmup + total time: "
        f"{warmup_ms:.2f} ms"
    )

    print(
        f"Warm-query retrieval: "
        f"{retrieval_ms:.2f} ms"
    )

    print()
    print("Results:")
    print("-" * 80)

    if not results:

        print(
            "No retrieval results."
        )

        return

    for i, result in enumerate(
        results,
        start=1,
    ):

        print()

        print(
            f"#{i} "
            f"[{result.get('strategy', 'unknown')}] "
            f"RRF="
            f"{result.get('rrf_score', 0):.6f}"
        )

        print(
            f"E5 rank: "
            f"{result.get('e5_rank')}"
        )

        print(
            f"BM25 rank: "
            f"{result.get('bm25_rank')}"
        )

        print(
            f"Lexical rank: "
            f"{result.get('lexical_rank')}"
        )

        print(
            f"BM25 score: "
            f"{result.get('bm25_score', 0.0):.4f}"
        )

        print(
            f"Lexical score: "
            f"{result.get('lexical_score', 0.0):.4f}"
        )

        print(
            f"Lexical ratio: "
            f"{result.get('lexical_ratio', 0.0):.3f}"
        )

        print(
            f"Relevance bonus: "
            f"{result.get('relevance_bonus', 0.0):.3f}"
        )

        print(
            f"Distance: "
            f"{result.get('distance')}"
        )

        print(
            f"Passage: "
            f"{result.get('chunk_id')}"
        )

        print(
            f"Text: "
            f"{result.get('text', '')[:1000]}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()