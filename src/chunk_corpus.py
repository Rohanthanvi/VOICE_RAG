"""
Day 2 (part 1) — Chunk the Hindi passage pool, multiple strategies

What this does:
Reads data/processed/passage_pool_hi.jsonl (the deduped passages from Day 1)
and produces multiple chunk representations of each passage. Every chunk
also carries metadata (ever_selected, how many queries linked to this
passage, character length) — that's the "metadata-aware" piece: it's not a
separate splitting method, it's something every chunk gets, so retrieval
can later filter or boost by these fields regardless of which strategy
produced the chunk.

Strategies implemented here (text-only, no embedding model needed):
  1. passage_level     — whole passage as one chunk (baseline)
  2. fixed_size_overlap — word-window chunks with overlap, for long passages
  3. parent_child       — each sentence is its own small "child" chunk, but
                           tagged with parent_passage_id so the full passage
                           can be pulled back as context after retrieval

A 4th strategy, semantic_sentence (embedding-similarity-based grouping),
needs the embedding model loaded — that happens in embed_and_index.py
(Day 2 part 2), since it doesn't make sense to load a model twice.

Run:
    python src/chunk_corpus.py --in_file data/processed/passage_pool_hi.jsonl
"""

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Hindi sentence-ending punctuation includes the danda (।) alongside
# standard Latin punctuation, since these passages mix Hindi and
# transliterated/borrowed terms.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[।.!?])\s+")
MIN_SENTENCE_WORDS = 3  # sentences shorter than this get merged into the previous one


def split_sentences(text: str) -> list[str]:
    raw = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    merged: list[str] = []
    for s in raw:
        if merged and len(s.split()) < MIN_SENTENCE_WORDS:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged if merged else [text.strip()]


def fixed_size_chunks(text: str, window: int = 40, overlap: int = 10) -> list[str]:
    words = text.split()
    if len(words) <= window:
        return [text]
    step = window - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + window]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + window >= len(words):
            break
    return chunks


def make_chunk(passage: dict, strategy: str, text: str, idx: int, parent_passage_id: str | None = None) -> dict:
    return {
        "chunk_id": f"{passage['passage_id']}_{strategy}_{idx}",
        "passage_id": passage["passage_id"],
        "parent_passage_id": parent_passage_id or passage["passage_id"],
        "strategy": strategy,
        "text": text,
        # metadata-aware fields — carried on every chunk regardless of strategy
        "ever_selected": passage["ever_selected"],
        "linked_query_count": len(passage["seen_with_query_ids"]),
        "char_len": len(text),
    }


def main(in_file: str, out_tag: str):
    in_path = Path(in_file)
    with open(in_path, encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]

    all_chunks = []
    counts = {"passage_level": 0, "fixed_size_overlap": 0, "parent_child": 0}

    for passage in tqdm(passages, desc="Chunking"):
        text = passage["text"]

        # Strategy 1: passage-level (baseline)
        c = make_chunk(passage, "passage_level", text, 0)
        all_chunks.append(c)
        counts["passage_level"] += 1

        # Strategy 2: fixed-size with overlap
        for i, piece in enumerate(fixed_size_chunks(text)):
            c = make_chunk(passage, "fixed_size_overlap", piece, i)
            all_chunks.append(c)
            counts["fixed_size_overlap"] += 1

        # Strategy 3: parent-child (sentence-level children, linked to parent passage)
        for i, sentence in enumerate(split_sentences(text)):
            c = make_chunk(passage, "parent_child", sentence, i, parent_passage_id=passage["passage_id"])
            all_chunks.append(c)
            counts["parent_child"] += 1

    out_path = PROCESSED_DIR / f"chunks_{out_tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as fout:
        for c in all_chunks:
            fout.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nChunked {len(passages)} passages -> {len(all_chunks)} total chunks")
    for strategy, n in counts.items():
        print(f"  - {strategy}: {n}")
    print(f"Saved -> {out_path}")
    print("\nSample chunks (one per strategy):")
    seen_strategies = set()
    for c in all_chunks:
        if c["strategy"] not in seen_strategies:
            seen_strategies.add(c["strategy"])
            print(f"\n[{c['strategy']}] {json.dumps(c, ensure_ascii=False, indent=2)[:400]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True, help="path to passage_pool_hi.jsonl from Day 1")
    parser.add_argument("--out_tag", type=str, default="hi", help="short tag used only for the output filename")
    args = parser.parse_args()
    main(args.in_file, args.out_tag)