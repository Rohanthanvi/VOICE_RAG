"""
Day 1 (wrap-up) — Build the Hindi passage pool

What this does:
Takes the extracted rows from extract_dataset.py (each row = one query +
~10 candidate passages) and flips the data around: instead of being
organized by query, we end up with a flat, deduplicated pool of unique
passages. THIS pool is what gets chunked and embedded into ChromaDB in
Day 2 — it's the actual searchable knowledge base.

Why dedupe: the same passage (same web paragraph) often appears as a
candidate under multiple different queries. Without deduping, the vector
DB would store the same text several times, wasting space and skewing
retrieval toward accidentally-duplicated content.

Why keep BOTH selected and non-selected passages: the task wants real
retrieval, not a lookup table of pre-marked correct answers. A pool made
only of "correct" passages would make retrieval trivially easy. Keeping
the non-selected ones too means the system has to actually distinguish
relevant from irrelevant based on the query.

Run:
    python src/build_corpus.py --in_file data/raw/msmarco_hi_train_5000.jsonl
"""

import argparse
import hashlib
import json
from pathlib import Path

from tqdm import tqdm

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def passage_id(text: str) -> str:
    """Stable ID derived from the text itself, so the exact same passage
    text always maps to the exact same ID no matter which query it showed
    up under — that's what makes deduplication possible."""
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


def main(in_file: str, out_tag: str):
    in_path = Path(in_file)
    pool: dict[str, dict] = {}  # passage_id -> passage record

    with open(in_path, encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Building passage pool"):
        row = json.loads(line)
        query_id = row["query_id"]
        selected_texts = set(row["selected_passages_hi"])

        for text in row["all_passages_hi"]:
            text = text.strip()
            if not text:
                continue
            pid = passage_id(text)
            was_selected = text in selected_texts

            if pid not in pool:
                pool[pid] = {
                    "passage_id": pid,
                    "text": text,
                    "seen_with_query_ids": [query_id],
                    "ever_selected": was_selected,
                }
            else:
                # Same passage text seen again under a different query —
                # just record the extra query_id, don't duplicate the text.
                pool[pid]["seen_with_query_ids"].append(query_id)
                pool[pid]["ever_selected"] = pool[pid]["ever_selected"] or was_selected

    out_path = PROCESSED_DIR / f"passage_pool_{out_tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as fout:
        for record in pool.values():
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    n_selected = sum(1 for r in pool.values() if r["ever_selected"])
    print(f"\nRead {len(lines)} query rows.")
    print(f"Unique passages after dedup: {len(pool)}")
    print(f"  - ever marked relevant (ever_selected=True): {n_selected}")
    print(f"  - never marked relevant (distractors): {len(pool) - n_selected}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True, help="path to the extracted jsonl from extract_dataset.py")
    parser.add_argument("--out_tag", type=str, default="hi", help="short tag used only for the output filename")
    args = parser.parse_args()
    main(args.in_file, args.out_tag)