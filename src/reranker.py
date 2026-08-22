"""
Stage 3 — Multilingual Reranker

Hacker House Goa — RAG_GOA_V2

Uses:
    BAAI/bge-reranker-v2-m3

The reranker receives:
    (query, passage)

and produces a relevance score.

It is used AFTER hybrid retrieval.
"""

from __future__ import annotations

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


MODEL_NAME = "BAAI/bge-reranker-v2-m3"

_model = None
_tokenizer = None
_device = None


def get_reranker():

    global _model
    global _tokenizer
    global _device

    if _model is None:

        print(
            f"Loading reranker: {MODEL_NAME}"
        )

        _device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        print(
            f"Reranker device: {_device}"
        )

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )

        _model.to(_device)

        _model.eval()

        print(
            "Reranker loaded."
        )

    return _tokenizer, _model, _device


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    batch_size: int = 16,
):
    """
    Rerank candidate passages against the query.

    Args:
        query:
            User query.

        candidates:
            Candidate dictionaries containing at least
            a 'text' field.

        top_k:
            Number of final results.

        batch_size:
            Number of query/passage pairs processed together.
    """

    if not candidates:
        return []

    tokenizer, model, device = get_reranker()

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    scores = []

    with torch.no_grad():

        for start in range(
            0,
            len(pairs),
            batch_size,
        ):

            batch_pairs = pairs[
                start:start + batch_size
            ]

            encoded = tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            outputs = model(
                **encoded
            )

            logits = outputs.logits

            # Standard single-score reranker
            batch_scores = (
                logits.view(-1)
                .float()
                .cpu()
                .tolist()
            )

            scores.extend(
                batch_scores
            )

    reranked = []

    for candidate, score in zip(
        candidates,
        scores,
    ):

        result = dict(candidate)

        result["reranker_score"] = float(
            score
        )

        reranked.append(result)

    reranked.sort(
        key=lambda x: x["reranker_score"],
        reverse=True,
    )

    return reranked[:top_k]


if __name__ == "__main__":

    query = "साइरीन क्या है"

    candidates = [
        {
            "chunk_id": "1",
            "text": (
                "साइरीन लीबिया में स्थित "
                "एक प्राचीन यूनानी और रोमन शहर था।"
            ),
            "metadata": {
                "strategy": "test"
            },
        },
        {
            "chunk_id": "2",
            "text": (
                "साइरीन एक लड़कियों के नाम "
                "के रूप में भी प्रयोग किया जाता है।"
            ),
            "metadata": {
                "strategy": "test"
            },
        },
    ]

    results = rerank(
        query,
        candidates,
        top_k=2,
    )

    print(
        "\nReranked results:"
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n#{rank} "
            f"score={result['reranker_score']:.4f}"
        )

        print(
            result["text"]
        )