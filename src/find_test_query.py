"""
Utility — find test queries that are actually answerable

Since the production ChromaDB index only contains a random 3000-passage
sample (not the full 49,499-passage pool), most of the original 5000
queries' correct passages simply aren't in the index. Testing with an
arbitrary question (like general "capital of India" trivia) has a real
chance of hitting a gap in the sample, not a retrieval failure.

This script reproduces the exact same random sample (same seed=42) used
by embed_and_index.py, then finds original queries whose selected passage
IS in that sample — so you get a fair test question with a known-correct
answer actually sitting in the index.

Run:
    python src/find_test_query.py
"""

import json
import random
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RANDOM_SEED = 42
SAMPLE_SIZE = 3000


def main():
    with open(PROCESSED_DIR / "passage_pool_hi.jsonl", encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]

    random.seed(RANDOM_SEED)
    sampled = random.sample(passages, SAMPLE_SIZE)
    sampled_ids = {p["passage_id"] for p in sampled}

    with open(RAW_DIR / "msmarco_hi_train_5000.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    def text_to_id(text: str) -> str:
        import hashlib
        return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]

    findable = []
    for row in rows:
        for passage_text in row["selected_passages_hi"]:
            if text_to_id(passage_text) in sampled_ids:
                findable.append({"query": row["query"], "answer": row["answer"], "passage": passage_text})
                break

    print(f"{len(findable)} of {len(rows)} original queries have their answer passage in the 3000-sample index.\n")
    print("Try these (their answer passage IS indexed, so retrieval should actually find it):\n")
    for item in findable[:10]:
        print(f"Q: {item['query']}")
        print(f"Expected answer: {item['answer']}")
        print(f"Source passage: {item['passage'][:150]}\n")


if __name__ == "__main__":
    main()