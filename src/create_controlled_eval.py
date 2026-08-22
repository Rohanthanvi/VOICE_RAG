"""
Hacker House Goa — Controlled Evaluation Set

Creates an evaluation set containing ONLY queries whose relevant
passage exists inside the controlled 10K passage index.

This makes the E5 vs EmbeddingGemma comparison fair.
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "msmarco_hi_train_5000.jsonl"
)

PASSAGE_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "passage_pool_hi.jsonl"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "controlled_eval_10k.jsonl"
)

LIMIT = 10000


def main():

    print("=" * 80)
    print("CREATING CONTROLLED EVALUATION SET")
    print("=" * 80)

    # ========================================================
    # FIRST 10K PASSAGES
    # ========================================================

    passage_ids = set()

    print(
        "\nLoading first "
        f"{LIMIT:,} passages..."
    )

    with open(
        PASSAGE_FILE,
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            passage_ids.add(
                row["passage_id"]
            )

            if len(passage_ids) >= LIMIT:
                break

    print(
        f"Controlled corpus: "
        f"{len(passage_ids):,} passages"
    )

    # ========================================================
    # RELEVANCE MAP
    # ========================================================

    query_to_relevant = {}

    print(
        "\nBuilding relevance map..."
    )

    with open(
        PASSAGE_FILE,
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            passage_id = row[
                "passage_id"
            ]

            if passage_id not in passage_ids:
                continue

            for query_id in row.get(
                "seen_with_query_ids",
                [],
            ):

                query_id = int(query_id)

                query_to_relevant.setdefault(
                    query_id,
                    set(),
                ).add(
                    passage_id
                )

    print(
        f"Queries with relevant passages: "
        f"{len(query_to_relevant):,}"
    )

    # ========================================================
    # LOAD QUERIES
    # ========================================================

    selected = []

    with open(
        RAW_FILE,
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            query_id = None

            for key in (
                "query_id",
                "id",
                "qid",
            ):

                if key in row:
                    query_id = int(
                        row[key]
                    )
                    break

            if query_id is None:
                continue

            if query_id not in query_to_relevant:
                continue

            selected.append(
                {
                    "query_id": query_id,
                    "query": row.get(
                        "query",
                        row.get(
                            "text",
                            "",
                        ),
                    ),
                    "relevant_passage_ids": sorted(
                        query_to_relevant[
                            query_id
                        ]
                    ),
                }
            )

    # ========================================================
    # WRITE
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        for row in selected:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"\nControlled queries: "
        f"{len(selected):,}"
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()