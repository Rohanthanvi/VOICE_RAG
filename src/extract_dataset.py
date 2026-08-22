"""
Day 1 — Dataset extraction for RAGInGoa

What this does:
1. Downloads the real per-language parquet file directly (e.g.
   "train/hintrain.parquet" for Hindi) using huggingface_hub. The dataset
   card's usage example (`load_dataset(..., "hi", ...)`) and its listed
   ".jsonl" filenames are both stale — the repo actually stores one parquet
   file per language inside train/ and validation/ folders.
2. Reads it with plain `pyarrow.parquet.ParquetFile` (NOT `datasets` /
   `pyarrow.dataset`) — the datasets library's streaming path uses
   pyarrow.dataset's scanner internally, which hits a known bug
   ("Nested data conversions not implemented for chunked array outputs")
   on this file's nested passage columns. ParquetFile.iter_batches() reads
   row groups directly and avoids that scanner entirely.
3. Filters to rows that actually have a selected passage (is_selected == 1
   somewhere) — rows with no selected passage are useless for RAG since
   there's no "ground truth" context to retrieve.
4. Flattens each row into a simpler shape: one query + its selected passages.
5. Saves the first N qualifying examples locally as JSONL so Day 2
   (chunking) can work offline, fast, without touching parquet/Arrow again.

Note: this downloads the full ~3.7GB file once (cached by huggingface_hub
afterwards, so it only happens the first time you run this).

Run:
    python src/extract_dataset.py --n 5000 --lang_prefix hin --out_tag hi
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def flatten_example(example: dict) -> dict | None:
    """Pull out just what we need: query, answer, and the passages marked
    as actually relevant (is_selected == 1). Returns None if there's no
    selected passage — nothing to retrieve, so we skip it."""
    passages = example["passages"]
    selected_idxs = [i for i, s in enumerate(passages["is_selected"]) if s == 1]
    if not selected_idxs:
        return None

    return {
        "query_id": example["query_id"],
        "query_type": example.get("query_type"),
        "query": example["query"],           # Hindi query
        "answer": example.get("Answer"),      # Hindi answer
        "eng_query": example.get("Eng_Query"),    # kept for debugging/eval
        "eng_answer": example.get("Eng_Answer"),
        "selected_passages_hi": [passages["Translated_passages"][i] for i in selected_idxs],
        "all_passages_hi": passages["Translated_passages"],
        "all_passages_en": passages.get("English_passages"),
    }


def main(n: int, split: str, lang_prefix: str, out_tag: str):
    split_folder = "train" if split == "train" else "validation"
    suffix = "train" if split == "train" else "val"
    filename = f"{split_folder}/{lang_prefix}{suffix}.parquet"

    print(f"Downloading {filename} from ai4bharat/MSMARCO-XI (one-time, ~3.7GB, cached after) ...")
    local_path = hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI",
        repo_type="dataset",
        filename=filename,
    )
    print(f"Downloaded to {local_path}")

    out_path = RAW_DIR / f"msmarco_{out_tag}_{split}_{n}.jsonl"
    kept = 0
    scanned = 0

    pf = pq.ParquetFile(local_path)
    with open(out_path, "w", encoding="utf-8") as fout:
        pbar = tqdm(total=n, desc="Extracting")
        for batch in pf.iter_batches(batch_size=500):
            for example in batch.to_pylist():
                scanned += 1
                flat = flatten_example(example)
                if flat is None:
                    continue
                fout.write(json.dumps(flat, ensure_ascii=False) + "\n")
                kept += 1
                pbar.update(1)
                if kept >= n:
                    break
            if kept >= n:
                break
        pbar.close()

    print(f"\nScanned {scanned} rows to find {kept} rows with a selected passage.")
    print(f"Saved {kept} examples -> {out_path}")
    print("Sample record:")
    with open(out_path, encoding="utf-8") as f:
        print(json.dumps(json.loads(f.readline()), ensure_ascii=False, indent=2)[:800])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="number of examples to extract")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"])
    parser.add_argument("--lang_prefix", type=str, default="hin", help="filename prefix on the Hub, e.g. hin, tam, tel, ben, guj (note: Gujarati is 'guj' not 'gu')")
    parser.add_argument("--out_tag", type=str, default="hi", help="short tag used only for the output filename")
    args = parser.parse_args()
    main(args.n, args.split, args.lang_prefix, args.out_tag)