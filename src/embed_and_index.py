"""
Day 2 (part 2) — Semantic chunking, embedding, and ChromaDB indexing

What this does:
1. Adds the 4th chunking strategy — semantic_sentence — which needed the
   embedding model, so it lives here rather than in chunk_corpus.py.
   Logic: split each passage into sentences, embed each sentence, then
   greedily merge ADJACENT sentences into one chunk as long as they stay
   semantically similar (cosine similarity above a threshold) and the
   merged chunk doesn't exceed a word cap. A topic shift (similarity drops)
   or hitting the size cap starts a new chunk. This is what makes it
   "semantic" rather than fixed-size: boundaries are decided by meaning,
   not a fixed word count.
2. Loads the chunks from chunk_corpus.py (passage_level, fixed_size_overlap,
   parent_child) and combines them with the new semantic_sentence chunks.
3. Embeds every chunk's text with EmbeddingGemma-300M.
4. Writes everything into a persistent ChromaDB collection, with each
   chunk's strategy + relevance metadata attached, so retrieval can later
   filter or compare by strategy.

IMPORTANT — one-time setup before running:
EmbeddingGemma is a gated Hugging Face model. Before this script will work:
  1. Visit https://huggingface.co/google/embeddinggemma-300m and accept the license.
  2. Run `huggingface-cli login` in your terminal (needs a Hugging Face access token).

Run (use --limit while testing, drop it for the full run):
    python src/embed_and_index.py --limit 500
    python src/embed_and_index.py --sample_passages 3000

NOTE ON CPU-ONLY MACHINES: embedding all 4 strategies across the full
~49,500-passage pool produces ~475,000 chunks, which takes many hours
without a GPU. Default behavior here embeds only passage_level +
semantic_sentence (the two most retrieval-useful, least redundant
strategies) and supports --sample_passages to cap the corpus size.
Use --strategies to include fixed_size_overlap / parent_child on a small
--limit run for strategy comparison instead of full-scale indexing.
"""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
RANDOM_SEED = 42

SENTENCE_SPLIT_RE = re.compile(r"(?<=[।.!?])\s+")
MIN_SENTENCE_WORDS = 3


def split_sentences(text: str) -> list[str]:
    raw = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    merged: list[str] = []
    for s in raw:
        if merged and len(s.split()) < MIN_SENTENCE_WORDS:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged if merged else [text.strip()]


def semantic_chunks(text: str, model, sim_threshold: float = 0.55, max_words: int = 60) -> list[str]:
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return sentences

    embs = model.encode(sentences, normalize_embeddings=True)
    chunks, current, current_words = [], [sentences[0]], len(sentences[0].split())

    for i in range(1, len(sentences)):
        sim = float(np.dot(embs[i - 1], embs[i]))  # normalized vectors -> dot product = cosine similarity
        words = len(sentences[i].split())
        if sim >= sim_threshold and current_words + words <= max_words:
            current.append(sentences[i])
            current_words += words
        else:
            chunks.append(" ".join(current))
            current, current_words = [sentences[i]], words
    chunks.append(" ".join(current))
    return chunks


def load_model():
    from sentence_transformers import SentenceTransformer
    print("Loading EmbeddingGemma-300M (first run downloads it, needs HF login + accepted license) ...")
    return SentenceTransformer("google/embeddinggemma-300m")


def main(limit: int | None, sample_passages: int | None, strategies: set[str], batch_size: int,
         collection_name: str, sim_threshold: float, max_words: int):
    model = load_model()

    # --- Step 1: load existing chunks from Day 2 part 1, keep only requested strategies ---
    chunks_path = PROCESSED_DIR / "chunks_hi.jsonl"
    with open(chunks_path, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if json.loads(line)["strategy"] in strategies]
    print(f"Loaded {len(chunks)} chunks from {chunks_path.name} matching strategies={sorted(strategies)}")

    # --- Step 2: load the passage pool, optionally subsample it ---
    pool_path = PROCESSED_DIR / "passage_pool_hi.jsonl"
    with open(pool_path, encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]

    if sample_passages and sample_passages < len(passages):
        random.seed(RANDOM_SEED)
        passages = random.sample(passages, sample_passages)
        print(f"--sample_passages set: randomly sampled {sample_passages} of {len(passages)} passages (seed={RANDOM_SEED})")
    if limit:
        passages = passages[:limit]
        print(f"--limit set: only processing first {limit} passages for this run")

    allowed_ids = {p["passage_id"] for p in passages}
    # keep only chunks belonging to the (possibly subsampled) passage set
    chunks = [c for c in chunks if c["passage_id"] in allowed_ids]

    # --- Step 3: build semantic_sentence chunks, only if requested ---
    if "semantic_sentence" in strategies:
        semantic_count = 0
        for passage in tqdm(passages, desc="Semantic chunking"):
            pieces = semantic_chunks(passage["text"], model, sim_threshold, max_words)
            for i, piece in enumerate(pieces):
                chunks.append({
                    "chunk_id": f"{passage['passage_id']}_semantic_sentence_{i}",
                    "passage_id": passage["passage_id"],
                    "parent_passage_id": passage["passage_id"],
                    "strategy": "semantic_sentence",
                    "text": piece,
                    "ever_selected": passage["ever_selected"],
                    "linked_query_count": len(passage["seen_with_query_ids"]),
                    "char_len": len(piece),
                })
                semantic_count += 1
        print(f"Built {semantic_count} semantic_sentence chunks")

    print(f"\nTotal chunks to embed and index: {len(chunks)}")

    # --- Step 3: embed everything ---
    texts = [c["text"] for c in chunks]
    print("Embedding all chunks (this is the slow part) ...")
    embeddings = model.encode_document(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # --- Step 4: store in ChromaDB ---
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=collection_name)

    print(f"Writing to ChromaDB collection '{collection_name}' at {CHROMA_DIR} ...")
    write_batch = 5000
    for start in tqdm(range(0, len(chunks), write_batch), desc="Indexing"):
        end = start + write_batch
        batch_chunks = chunks[start:end]
        collection.upsert(
            ids=[c["chunk_id"] for c in batch_chunks],
            documents=[c["text"] for c in batch_chunks],
            embeddings=embeddings[start:end].tolist(),
            metadatas=[{
                "passage_id": c["passage_id"],
                "parent_passage_id": c["parent_passage_id"],
                "strategy": c["strategy"],
                "ever_selected": c["ever_selected"],
                "linked_query_count": c["linked_query_count"],
                "char_len": c["char_len"],
            } for c in batch_chunks],
        )

    print(f"\nDone. Collection '{collection_name}' now has {collection.count()} chunks.")
    from collections import Counter
    strategy_counts = Counter(c["strategy"] for c in chunks)
    for strategy, n in strategy_counts.items():
        print(f"  - {strategy}: {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process first N passages after sampling (for a fast test run)")
    parser.add_argument("--sample_passages", type=int, default=None, help="randomly subsample the passage pool to N passages before chunking (recommended for CPU-only machines)")
    parser.add_argument("--strategies", type=str, default="passage_level,semantic_sentence",
                         help="comma-separated list from: passage_level,fixed_size_overlap,parent_child,semantic_sentence")
    parser.add_argument("--batch_size", type=int, default=64, help="embedding batch size")
    parser.add_argument("--collection_name", type=str, default="msmarco_hi_chunks")
    parser.add_argument("--sim_threshold", type=float, default=0.55, help="cosine similarity cutoff for merging sentences in semantic_sentence")
    parser.add_argument("--max_words", type=int, default=60, help="max words per semantic_sentence chunk")
    args = parser.parse_args()
    strategies = set(s.strip() for s in args.strategies.split(","))
    main(args.limit, args.sample_passages, strategies, args.batch_size, args.collection_name, args.sim_threshold, args.max_words)