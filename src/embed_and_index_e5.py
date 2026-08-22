"""
Hacker House Goa — E5-small FULL production index

Creates a full ChromaDB index using:
    intfloat/multilingual-e5-small

IMPORTANT:
- Existing EmbeddingGemma index is NOT modified.
- Existing E5 test indexes are NOT modified.
- Documents are embedded in batches to avoid loading all embeddings
  into RAM at once.
- E5 passage prefix is used during indexing:
      passage: <text>

Production index:
    chroma_db_e5/
    msmarco_hi_chunks_e5

Run full index:
    python src/embed_and_index_e5.py

Optional test:
    python src/embed_and_index_e5.py --limit 10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "chunks_hi.jsonl"
)

# NEW production E5 index
CHROMA_DIR = (
    PROJECT_DIR
    / "chroma_db_e5"
)

COLLECTION_NAME = (
    "msmarco_hi_chunks_e5"
)

MODEL_NAME = (
    "intfloat/multilingual-e5-small"
)


# ============================================================
# SETTINGS
# ============================================================

# Number of documents embedded by the model at once.
# 32 is safe for CPU/RAM.
EMBED_BATCH_SIZE = 32

# Number of vectors written to Chroma at once.
WRITE_BATCH_SIZE = 1000


# ============================================================
# MODEL
# ============================================================

def load_model():

    print(
        "\nLoading multilingual-e5-small..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "E5-small loaded."
    )

    return model


# ============================================================
# CHROMA
# ============================================================

def get_collection():

    print(
        "\nOpening ChromaDB..."
    )

    print(
        f"Directory:\n{CHROMA_DIR}"
    )

    print(
        f"Collection:\n{COLLECTION_NAME}"
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


# ============================================================
# BATCH GENERATOR
# ============================================================

def read_batches(
    limit: int | None,
    batch_size: int,
):

    batch = []

    count = 0

    with open(
        CHUNKS_FILE,
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            chunk = json.loads(line)

            batch.append(chunk)

            count += 1

            if len(batch) >= batch_size:

                yield batch

                batch = []

            if (
                limit is not None
                and count >= limit
            ):
                break

    if batch:
        yield batch


# ============================================================
# MAIN
# ============================================================

def main(limit: int | None):

    print("=" * 80)
    print("HACKER HOUSE GOA — E5 SMALL FULL INDEX")
    print("=" * 80)

    print(
        f"\nSource:\n{CHUNKS_FILE}"
    )

    if limit is None:

        print(
            "\nMode: FULL CORPUS"
        )

    else:

        print(
            f"\nMode: TEST — first {limit:,} chunks"
        )

    print(
        f"\nProduction Chroma directory:\n"
        f"{CHROMA_DIR}"
    )

    print(
        f"\nProduction collection:\n"
        f"{COLLECTION_NAME}"
    )

    print(
        "\nExisting EmbeddingGemma index:"
    )

    print(
        "UNCHANGED"
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # CHROMA
    # ========================================================

    collection = get_collection()

    # ========================================================
    # INDEX
    # ========================================================

    print(
        "\nStarting batch indexing..."
    )

    total_processed = 0

    batch_iterator = read_batches(
        limit=limit,
        batch_size=WRITE_BATCH_SIZE,
    )

    for chunks in tqdm(
        batch_iterator,
        desc="Indexing batches",
    ):

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        texts = [
            c["text"]
            for c in chunks
        ]

        # ----------------------------------------------------
        # E5 DOCUMENT PREFIX
        # ----------------------------------------------------

        passage_texts = [
            f"passage: {text}"
            for text in texts
        ]

        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------

        embeddings = model.encode(
            passage_texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # ----------------------------------------------------
        # CHROMA METADATA
        # ----------------------------------------------------

        metadatas = []

        for c in chunks:

            metadatas.append(
                {
                    "passage_id": str(
                        c.get(
                            "passage_id",
                            "",
                        )
                    ),

                    "parent_passage_id": str(
                        c.get(
                            "parent_passage_id",
                            "",
                        )
                    ),

                    "strategy": str(
                        c.get(
                            "strategy",
                            "",
                        )
                    ),

                    "ever_selected": bool(
                        c.get(
                            "ever_selected",
                            False,
                        )
                    ),

                    "linked_query_count": int(
                        c.get(
                            "linked_query_count",
                            0,
                        )
                    ),

                    "char_len": int(
                        c.get(
                            "char_len",
                            len(c["text"]),
                        )
                    ),
                }
            )

        # ----------------------------------------------------
        # WRITE TO CHROMA
        # ----------------------------------------------------

        collection.upsert(

            ids=[
                str(
                    c["chunk_id"]
                )
                for c in chunks
            ],

            documents=texts,

            embeddings=embeddings.tolist(),

            metadatas=metadatas,
        )

        total_processed += len(
            chunks
        )

        # Explicitly release the large embedding array
        del embeddings

    # ========================================================
    # FINAL
    # ========================================================

    print("\n")
    print("=" * 80)
    print("E5 PRODUCTION INDEX COMPLETE")
    print("=" * 80)

    print(
        f"\nChunks processed: "
        f"{total_processed:,}"
    )

    print(
        f"Collection count: "
        f"{collection.count():,}"
    )

    print(
        f"\nIndex directory:"
    )

    print(
        CHROMA_DIR
    )

    print(
        f"\nCollection:"
    )

    print(
        COLLECTION_NAME
    )

    print(
        "\nE5 document format:"
    )

    print(
        "passage: <document>"
    )

    print(
        "\nExisting indexes were NOT modified."
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of chunks to index. "
            "Omit this argument for the full corpus."
        ),
    )

    args = parser.parse_args()

    main(
        args.limit
    )