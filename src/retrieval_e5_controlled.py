from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = (
    PROJECT_DIR
    / "chroma_db_e5_controlled"
)

COLLECTION_NAME = (
    "msmarco_hi_passage_e5_controlled"
)

MODEL_NAME = (
    "intfloat/multilingual-e5-small"
)


_model = None
_collection = None


def get_model():

    global _model

    if _model is None:

        _model = SentenceTransformer(
            MODEL_NAME
        )

    return _model


def get_collection():

    global _collection

    if _collection is None:

        import chromadb

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        _collection = client.get_collection(
            name=COLLECTION_NAME
        )

    return _collection


def retrieve(
    query_text: str,
    top_k: int = 5,
):

    model = get_model()

    collection = get_collection()

    query_embedding = model.encode(
        [f"query: {query_text}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
    )

    output = []

    for i in range(
        len(results["ids"][0])
    ):

        output.append(
            {
                "passage_id": results[
                    "ids"
                ][0][i],

                "text": results[
                    "documents"
                ][0][i],

                "distance": results[
                    "distances"
                ][0][i],
            }
        )

    return output


if __name__ == "__main__":

    import sys

    query = (
        " ".join(sys.argv[1:])
        or "भारत की राजधानी क्या है"
    )

    print(
        f"\nQuery: {query}"
    )

    results = retrieve(
        query,
        top_k=5,
    )

    for i, result in enumerate(
        results,
        1,
    ):

        print(
            f"\n#{i} "
            f"distance="
            f"{result['distance']:.4f}"
        )

        print(
            result["text"][:500]
        )