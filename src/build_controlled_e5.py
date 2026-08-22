"""
Hacker House Goa — Controlled E5 Test Index

Creates a fair 10K passage-level E5 index.

This experiment:
- uses passage_pool_hi.jsonl
- selects 10,000 passages
- creates E5 embeddings
- writes a NEW ChromaDB index
- does NOT modify the existing EmbeddingGemma index

Run:
    python src/build_controlled_e5.py --limit 10000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parent.parent

PASSAGE_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "passage_pool_hi.jsonl"
)

CHROMA_DIR = (
    PROJECT_DIR
    / "chroma_db_e5_controlled"
)

COLLECTION_NAME = "msmarco_hi_passage_e5_controlled"

MODEL_NAME = "intfloat/multilingual-e5-small"

RANDOM_SEED = 42

BATCH_SIZE = 32

WRITE_BATCH = 1000


def load_passages(limit: int):

    print("\nLoading passage pool...")

    passages = []

    with open(
        PASSAGE_FILE,
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            passages.append(
                json.loads(line)
            )

            if len(passages) >= limit:
                break

    print(
        f"Loaded {len(passages):,} passages."
    )

    return passages


def main(limit: int):

    print("=" * 80)
    print("HACKER HOUSE GOA — CONTROLLED E5 INDEX")
    print("=" * 80)

    print(
        f"\nSource:\n{PASSAGE_FILE}"
    )

    print(
        f"\nNumber of passages: {limit:,}"
    )

    print(
        f"\nNew Chroma directory:\n{CHROMA_DIR}"
    )

    print(
        f"\nCollection:\n{COLLECTION_NAME}"
    )

    # ========================================================
    # LOAD PASSAGES
    # ========================================================

    passages = load_passages(limit)

    if not passages:
        raise RuntimeError(
            "No passages were loaded."
        )

    # ========================================================
    # LOAD E5
    # ========================================================

    print(
        "\nLoading multilingual-e5-small..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "E5-small loaded."
    )

    # ========================================================
    # PREPARE PASSAGES
    # ========================================================

    texts = [
        p["text"]
        for p in passages
    ]

    # E5 passage prefix
    e5_texts = [
        f"passage: {text}"
        for text in texts
    ]

    # ========================================================
    # EMBEDDING
    # ========================================================

    print(
        "\nEmbedding passages..."
    )

    embeddings = model.encode(
        e5_texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(
        "\nEmbedding complete."
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # ========================================================
    # CHROMADB
    # ========================================================

    print(
        "\nCreating ChromaDB..."
    )

    import chromadb

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # ========================================================
    # WRITE
    # ========================================================

    print(
        "\nWriting vectors..."
    )

    for start in tqdm(
        range(
            0,
            len(passages),
            WRITE_BATCH,
        ),
        desc="Indexing",
    ):

        end = min(
            start + WRITE_BATCH,
            len(passages),
        )

        batch = passages[start:end]

        collection.upsert(
            ids=[
                p["passage_id"]
                for p in batch
            ],

            documents=[
                p["text"]
                for p in batch
            ],

            embeddings=embeddings[
                start:end
            ].tolist(),

            metadatas=[
                {
                    "passage_id": p["passage_id"],
                    "ever_selected": p.get(
                        "ever_selected",
                        False,
                    ),
                    "linked_query_count": len(
                        p.get(
                            "seen_with_query_ids",
                            [],
                        )
                    ),
                }
                for p in batch
            ],
        )

    # ========================================================
    # DONE
    # ========================================================

    print("\n" + "=" * 80)
    print("CONTROLLED E5 INDEX COMPLETE")
    print("=" * 80)

    print(
        f"\nIndexed passages: "
        f"{collection.count():,}"
    )

    print(
        f"\nDirectory:\n{CHROMA_DIR}"
    )

    print(
        "\nExisting EmbeddingGemma index was NOT modified."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
    )

    args = parser.parse_args()

    main(args.limit)