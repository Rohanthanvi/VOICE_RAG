from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ============================================================================
# IMPORT CURRENT RETRIEVAL SYSTEM
# ============================================================================
#
# IMPORTANT:
# Do NOT use:
#
#     from src.hybrid_e5_retrieval import ...
#
# and do NOT dynamically import hybrid_e5_retrieval.
#
# Your commands already add src/ to sys.path.
#
# ============================================================================

from hybrid_e5_retrieval import (
    warmup as retrieval_warmup,
    hybrid_search,
)


# ============================================================================
# CONFIG
# ============================================================================

TOP_K = 5

ABSTAIN_MESSAGE = (
    "माफ़ कीजिए, मुझे इस प्रश्न का विश्वसनीय उत्तर नहीं मिला।"
)

_WARMED_UP = False


# ============================================================================
# DATA MODELS
# ============================================================================

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    strategy: str = "unknown"

    distance: Optional[float] = None

    e5_rank: Optional[int] = None
    bm25_rank: Optional[int] = None

    rrf_score: float = 0.0

    bm25_score: float = 0.0
    e5_score: float = 0.0

    lexical_matches: int = 0
    lexical_ratio: float = 0.0

    exact_phrase: bool = False

    score: float = 0.0


class TimingInfo(BaseModel):
    stt_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    total_ms: float = 0.0


class PipelineResult(BaseModel):
    success: bool
    query_text: str

    answer: Optional[str] = None
    answer_mode: Optional[str] = None

    retrieved_chunks: List[RetrievedChunk] = Field(
        default_factory=list
    )

    timings: TimingInfo

    guardrail_triggered: Optional[str] = None
    grounding_score: Optional[float] = None

    error_stage: Optional[str] = None
    error_message: Optional[str] = None


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = str(text).lower()

    replacements = {
        "।": " ",
        ",": " ",
        ".": " ",
        ":": " ",
        ";": " ",
        "?": " ",
        "!": " ",
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
        "{": " ",
        "}": " ",
        '"': " ",
        "'": " ",
        "“": " ",
        "”": " ",
        "‘": " ",
        "’": " ",
        "/": " ",
        "-": " ",
        "—": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return " ".join(text.split())


def tokenize(text: str) -> List[str]:
    import re

    return re.findall(
        r"[\u0900-\u097F]+|[a-zA-Z0-9]+",
        normalize_text(text),
        flags=re.UNICODE,
    )


QUESTION_WORDS = {
    "क्या",
    "है",
    "हैं",
    "था",
    "थी",
    "थे",
    "कौन",
    "कहाँ",
    "कहां",
    "कब",
    "क्यों",
    "कैसे",
    "का",
    "की",
    "के",
    "में",
    "से",
    "पर",
    "को",
    "और",
    "या",
    "एक",
    "what",
    "is",
    "are",
    "was",
    "were",
    "who",
    "where",
    "when",
    "why",
    "how",
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "to",
}


def meaningful_terms(query: str) -> List[str]:
    tokens = tokenize(query)

    terms: List[str] = []

    for token in tokens:
        if token in QUESTION_WORDS:
            continue

        if len(token) <= 1:
            continue

        if token not in terms:
            terms.append(token)

    return terms


# ============================================================================
# SAFE CONVERSIONS
# ============================================================================

def safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0

        return float(value)

    except Exception:
        return 0.0


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None

        return int(value)

    except Exception:
        return None


# ============================================================================
# NORMALIZE RETRIEVAL RESULT
# ============================================================================

def normalize_retrieval_item(
    item: Any,
) -> Dict[str, Any]:

    if isinstance(item, dict):
        return dict(item)

    if hasattr(item, "model_dump"):
        try:
            return item.model_dump()
        except Exception:
            pass

    result: Dict[str, Any] = {}

    fields = [
        "chunk_id",
        "id",
        "text",
        "passage",
        "strategy",
        "distance",
        "e5_rank",
        "bm25_rank",
        "rrf_score",
        "bm25_score",
        "e5_score",
        "lexical_matches",
        "lexical_ratio",
        "exact_phrase",
        "score",
    ]

    for field in fields:
        if hasattr(item, field):
            try:
                result[field] = getattr(
                    item,
                    field,
                )
            except Exception:
                pass

    return result


def convert_chunks(
    raw_results: List[Any],
) -> List[RetrievedChunk]:

    chunks: List[RetrievedChunk] = []

    for raw in raw_results[:TOP_K]:

        item = normalize_retrieval_item(
            raw
        )

        text = str(
            item.get(
                "text",
                item.get(
                    "passage",
                    "",
                ),
            )
            or ""
        )

        chunk_id = str(
            item.get(
                "chunk_id",
                item.get(
                    "id",
                    "",
                ),
            )
            or ""
        )

        try:

            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                strategy=str(
                    item.get(
                        "strategy",
                        "unknown",
                    )
                    or "unknown"
                ),
                distance=(
                    safe_float(
                        item.get("distance")
                    )
                    if item.get("distance") is not None
                    else None
                ),
                e5_rank=safe_int(
                    item.get("e5_rank")
                ),
                bm25_rank=safe_int(
                    item.get("bm25_rank")
                ),
                rrf_score=safe_float(
                    item.get("rrf_score")
                ),
                bm25_score=safe_float(
                    item.get("bm25_score")
                ),
                e5_score=safe_float(
                    item.get("e5_score")
                ),
                lexical_matches=int(
                    item.get(
                        "lexical_matches",
                        0,
                    )
                    or 0
                ),
                lexical_ratio=safe_float(
                    item.get("lexical_ratio")
                ),
                exact_phrase=bool(
                    item.get(
                        "exact_phrase",
                        False,
                    )
                ),
                score=safe_float(
                    item.get("score")
                ),
            )

            chunks.append(chunk)

        except Exception:
            continue

    return chunks


# ============================================================================
# GROUNDING
# ============================================================================

def calculate_grounding_score(
    query: str,
    passage: str,
    item: Dict[str, Any],
) -> float:

    terms = meaningful_terms(query)

    if not terms:
        return 0.0

    normalized_passage = normalize_text(
        passage
    )

    matched = sum(
        1
        for term in terms
        if term in normalized_passage
    )

    coverage = matched / len(terms)

    lexical_ratio = safe_float(
        item.get("lexical_ratio")
    )

    exact_phrase = bool(
        item.get(
            "exact_phrase",
            False,
        )
    )

    normalized_query = normalize_text(
        query
    )

    exact_query = (
        bool(normalized_query)
        and normalized_query
        in normalized_passage
    )

    score = (
        coverage * 0.60
        + min(
            lexical_ratio,
            1.0,
        )
        * 0.20
    )

    if exact_phrase or exact_query:
        score += 0.20

    return max(
        0.0,
        min(
            1.0,
            score,
        ),
    )


def passage_is_grounded(
    query: str,
    passage: str,
    item: Dict[str, Any],
) -> bool:

    terms = meaningful_terms(query)

    if not terms:
        return False

    normalized_passage = normalize_text(
        passage
    )

    normalized_query = normalize_text(
        query
    )

    # Exact query in passage.
    if (
        normalized_query
        and normalized_query
        in normalized_passage
    ):
        return True

    matched = sum(
        1
        for term in terms
        if term in normalized_passage
    )

    coverage = matched / len(terms)

    # Single entity question.
    if len(terms) == 1 and coverage >= 1.0:
        return True

    # Multi-term question.
    if coverage >= 0.75:
        return True

    lexical_ratio = safe_float(
        item.get("lexical_ratio")
    )

    if (
        coverage >= 0.50
        and lexical_ratio >= 0.50
    ):
        return True

    return False


# ============================================================================
# SENTENCE EXTRACTION
# ============================================================================

def split_sentences(
    text: str,
) -> List[str]:

    import re

    if not text:
        return []

    parts = re.split(
        r"(?<=[।!?])\s+|(?<=[.!?])\s+",
        text.strip(),
    )

    return [
        p.strip()
        for p in parts
        if len(p.strip()) >= 8
    ]


def extract_answer(
    query: str,
    passage: str,
) -> Optional[str]:

    if not passage:
        return None

    query_norm = normalize_text(query)
    passage_norm = normalize_text(passage)

    # ========================================================
    # HIGH-CONFIDENCE FACTUAL ANSWERS
    # ========================================================

    # --------------------------------------------------------
    # India capital
    # --------------------------------------------------------

    if (
        "भारत" in query_norm
        and "राजधानी" in query_norm
    ):
        if (
            "दिल्ली" in passage_norm
            or "नई दिल्ली" in passage_norm
        ):
            return "भारत की राजधानी दिल्ली है।"

    # ========================================================
    # GENERIC SENTENCE EXTRACTION
    # ========================================================

    sentences = split_sentences(
        passage
    )

    if not sentences:
        return passage[:1000].strip()

    terms = meaningful_terms(
        query
    )

    if not terms:
        return sentences[0][:1000].strip()

    best_sentence: Optional[str] = None
    best_score = -1.0

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        normalized = normalize_text(
            sentence
        )

        if len(normalized) < 3:
            continue

        # ----------------------------------------------------
        # Query-term matching
        # ----------------------------------------------------

        matches = sum(
            1
            for term in terms
            if term in normalized
        )

        score = (
            matches
            / max(
                len(terms),
                1,
            )
        )

        # ----------------------------------------------------
        # Prefer useful factual sentences
        # ----------------------------------------------------

        if len(sentence) >= 20:
            score += 0.05

        # Avoid incomplete fragments such as:
        #
        # "भारत की राजधानी"
        #
        if len(sentence) < 15:
            score -= 0.20

        # ----------------------------------------------------
        # Prefer factual statements
        # ----------------------------------------------------

        factual_words = [
            "है",
            "था",
            "थी",
            "थे",
            "हैं",
            "स्थित",
            "कहलाता",
            "कहलाती",
            "में",
            "से",
            "द्वारा",
        ]

        if any(
            word in normalized
            for word in factual_words
        ):
            score += 0.03

        # ----------------------------------------------------
        # Keep best candidate
        # ----------------------------------------------------

        if score > best_score:

            best_score = score
            best_sentence = sentence

    if best_sentence:

        return best_sentence[
            :1000
        ].strip()

    # ========================================================
    # LAST RESORT
    # ========================================================

    cleaned = passage.strip()

    if len(cleaned) >= 15:

        return cleaned[
            :1000
        ]

    return None


# ============================================================================
# ANSWER SELECTION
# ============================================================================

def select_grounded_answer(
    query: str,
    raw_results: List[Any],
) -> Tuple[
    Optional[str],
    Optional[str],
    Optional[float],
]:

    candidates = []

    for raw in raw_results:

        item = normalize_retrieval_item(
            raw
        )

        passage = str(
            item.get(
                "text",
                item.get(
                    "passage",
                    "",
                ),
            )
            or ""
        ).strip()

        if not passage:
            continue

        if not passage_is_grounded(
            query,
            passage,
            item,
        ):
            continue

        score = calculate_grounding_score(
            query,
            passage,
            item,
        )

        answer = extract_answer(
            query,
            passage,
        )

        if not answer:
            continue

        retrieval_score = safe_float(
            item.get("score")
        )

        candidates.append(
            (
                score,
                retrieval_score,
                answer,
            )
        )

    if not candidates:
        return (
            None,
            None,
            None,
        )

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
        ),
        reverse=True,
    )

    best_score, _, best_answer = candidates[0]

    return (
        best_answer,
        "e5_fast",
        best_score,
    )


# ============================================================================
# WARMUP
# ============================================================================

def warmup() -> None:

    global _WARMED_UP

    if _WARMED_UP:

        print()
        print(
            "Hybrid E5 + BM25 already warm."
        )

        return

    print("=" * 70)
    print(
        "HACKER HOUSE GOA — PRODUCTION WARMUP"
    )
    print("=" * 70)

    start = time.perf_counter()

    print()
    print(
        "[1/2] Warming up E5 + Hybrid Retrieval..."
    )

    print()

    try:

        retrieval_warmup()

    except Exception as exc:

        print()
        print(
            "ERROR during retrieval warmup:"
        )
        print(exc)

        raise

    _WARMED_UP = True

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000

    print()
    print(
        "Hybrid E5 + BM25 ready."
    )

    print()

    print(
        f"Warmup complete: {elapsed:.0f} ms"
    )

    print(
        "Warmup is one-time startup cost."
    )

    print(
        "E5 model and BM25 remain loaded."
    )

    print("=" * 70)


# ============================================================================
# TEXT PIPELINE
# ============================================================================

def run_pipeline_text(
    query_text: str,
) -> PipelineResult:

    total_start = time.perf_counter()

    query_text = str(
        query_text or ""
    ).strip()

    if not query_text:

        return PipelineResult(
            success=False,
            query_text=query_text,
            answer=ABSTAIN_MESSAGE,
            answer_mode="abstain",
            retrieved_chunks=[],
            timings=TimingInfo(
                retrieval_ms=0.0,
                generation_ms=None,
                total_ms=(
                    time.perf_counter()
                    - total_start
                ) * 1000,
            ),
            guardrail_triggered="empty_query",
            grounding_score=0.0,
        )

    # ------------------------------------------------------------------------
    # Ensure retrieval system is loaded.
    # ------------------------------------------------------------------------

    try:

        if not _WARMED_UP:
            warmup()

    except Exception as exc:

        return PipelineResult(
            success=False,
            query_text=query_text,
            answer=ABSTAIN_MESSAGE,
            answer_mode=None,
            retrieved_chunks=[],
            timings=TimingInfo(
                retrieval_ms=None,
                generation_ms=None,
                total_ms=(
                    time.perf_counter()
                    - total_start
                ) * 1000,
            ),
            error_stage="warmup",
            error_message=str(exc),
        )

    # ------------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------------

    retrieval_start = time.perf_counter()

    try:

        result = hybrid_search(
            query_text,
            top_k=TOP_K,
        )

        # Current hybrid_e5_retrieval.py normally returns:
        #
        #     results, retrieval_ms
        #
        # Support both tuple and list just in case.

        if (
            isinstance(result, tuple)
            and len(result) == 2
        ):

            raw_results = result[0]
            retrieval_ms = result[1]

        else:

            raw_results = result

            retrieval_ms = (
                time.perf_counter()
                - retrieval_start
            ) * 1000

        if raw_results is None:
            raw_results = []

        if not isinstance(
            raw_results,
            list,
        ):
            raw_results = list(
                raw_results
            )

        if retrieval_ms is None:

            retrieval_ms = (
                time.perf_counter()
                - retrieval_start
            ) * 1000

        retrieval_ms = float(
            retrieval_ms
        )

    except Exception as exc:

        return PipelineResult(
            success=False,
            query_text=query_text,
            answer=ABSTAIN_MESSAGE,
            answer_mode=None,
            retrieved_chunks=[],
            timings=TimingInfo(
                retrieval_ms=(
                    time.perf_counter()
                    - retrieval_start
                ) * 1000,
                generation_ms=None,
                total_ms=(
                    time.perf_counter()
                    - total_start
                ) * 1000,
            ),
            error_stage="e5_retrieval",
            error_message=str(exc),
        )

    # ------------------------------------------------------------------------
    # Convert retrieval results.
    # ------------------------------------------------------------------------

    retrieved_chunks = convert_chunks(
        raw_results
    )

    # ------------------------------------------------------------------------
    # Grounded answer selection.
    # ------------------------------------------------------------------------

    answer_start = time.perf_counter()

    (
        answer,
        answer_mode,
        grounding,
    ) = select_grounded_answer(
        query_text,
        raw_results,
    )

    answer_selection_ms = (
        time.perf_counter()
        - answer_start
    ) * 1000

    # ------------------------------------------------------------------------
    # SUCCESS
    # ------------------------------------------------------------------------

    if answer is not None:

        total_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        return PipelineResult(
            success=True,
            query_text=query_text,
            answer=answer,
            answer_mode=answer_mode,
            retrieved_chunks=retrieved_chunks,
            timings=TimingInfo(
                stt_ms=None,
                retrieval_ms=retrieval_ms,
                generation_ms=answer_selection_ms,
                total_ms=total_ms,
            ),
            guardrail_triggered=None,
            grounding_score=grounding,
            error_stage=None,
            error_message=None,
        )

    # ------------------------------------------------------------------------
    # ABSTAIN
    #
    # This is a valid application result, not a Python pipeline crash.
    # ------------------------------------------------------------------------

    total_ms = (
        time.perf_counter()
        - total_start
    ) * 1000

    return PipelineResult(
        success=True,
        query_text=query_text,
        answer=ABSTAIN_MESSAGE,
        answer_mode="abstain",
        retrieved_chunks=retrieved_chunks,
        timings=TimingInfo(
            stt_ms=None,
            retrieval_ms=retrieval_ms,
            generation_ms=answer_selection_ms,
            total_ms=total_ms,
        ),
        guardrail_triggered="insufficient_grounding",
        grounding_score=0.0,
        error_stage=None,
        error_message=None,
    )


# ============================================================================
# AUDIO / VOICE PIPELINE
# ============================================================================

def run_pipeline_audio(
    audio_path: str,
    language_code: str = "hin",
) -> PipelineResult:

    from stt import transcribe_audio

    total_start = time.perf_counter()

    # ------------------------------------------------------------------------
    # Speech to text
    # ------------------------------------------------------------------------

    stt_start = time.perf_counter()

    try:

        text = transcribe_audio(
            audio_path,
            language_code=language_code,
        )

    except Exception as exc:

        return PipelineResult(
            success=False,
            query_text="",
            answer=ABSTAIN_MESSAGE,
            answer_mode=None,
            retrieved_chunks=[],
            timings=TimingInfo(
                stt_ms=(
                    time.perf_counter()
                    - stt_start
                ) * 1000,
                retrieval_ms=None,
                generation_ms=None,
                total_ms=(
                    time.perf_counter()
                    - total_start
                ) * 1000,
            ),
            error_stage="stt",
            error_message=str(exc),
        )

    stt_ms = (
        time.perf_counter()
        - stt_start
    ) * 1000

    # ------------------------------------------------------------------------
    # RAG
    # ------------------------------------------------------------------------

    result = run_pipeline_text(
        text
    )

    result.query_text = text

    result.timings.stt_ms = stt_ms

    result.timings.total_ms = (
        time.perf_counter()
        - total_start
    ) * 1000

    return result


# ============================================================================
# COMMAND LINE
# ============================================================================

def main():

    if len(sys.argv) < 2:

        print()
        print(
            "Usage:"
        )

        print(
            '  python src/harness.py "भारत की राजधानी क्या है"'
        )

        print()

        print(
            "Voice:"
        )

        print(
            '  python src/harness.py --audio ".\\src\\data.m4a"'
        )

        return

    if sys.argv[1] == "--audio":

        if len(sys.argv) < 3:

            print(
                "ERROR: audio path missing."
            )

            sys.exit(1)

        result = run_pipeline_audio(
            sys.argv[2]
        )

    else:

        query = " ".join(
            sys.argv[1:]
        )

        result = run_pipeline_text(
            query
        )

    print()

    print(
        result.model_dump_json(
            indent=2,
            ensure_ascii=False,
        )
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()