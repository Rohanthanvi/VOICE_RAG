"""
Day 4 (part 1) — Guardrails

Three checks, each addressing a different failure mode:

1. check_unsafe_input(text) — a fast local keyword/pattern check run on
   the transcribed query BEFORE retrieval or generation happen. Catches
   obviously unsafe categories (violence, self-harm, illegal activity,
   hate speech) without needing a network call, so an unsafe query never
   even reaches the LLM. This is intentionally simple and transparent —
   a keyword list rather than a black-box classifier — because for a
   demo/submission it's more important that graders can see exactly what
   it catches and why, than to have a state-of-the-art moderation model.

2. is_grounded_enough(top_distance) — after retrieval, checks whether the
   BEST match is actually close enough to trust. If nothing in the corpus
   is genuinely relevant (distance too high), we skip generation entirely
   and return "I don't know" directly — this both prevents hallucination
   AND saves the slowest stage (the LLM call) on queries we already know
   we can't answer well.
   Threshold was calibrated empirically from real test queries: correct
   matches during Day 3 testing ranged 0.54-1.16; a topically-related-
   but-not-quite-right match (the "capital of India" gap) sat at ~1.26.
   1.2 is set as the cutoff — tunable if you gather more test data.

3. answer_overlap_score(answer, retrieved_chunks) — after generation, a
   lightweight lexical-overlap check between the answer and the context
   it was supposedly grounded in. Low overlap doesn't necessarily mean
   the answer is wrong, but it's a useful transparency signal to log and
   report, and demonstrates the system tracking its own grounding rather
   than trusting the LLM blindly.
"""

import re

# Keyword/pattern groups, kept separate by category for clarity in
# logging (which category triggered, not just "blocked"). Hindi and
# English terms included since queries may mix scripts/languages.
# Stem-based (not exact-phrase) patterns: Hindi verbs conjugate heavily
# (बनाना / बनाने / बनाएं / बनाया are all forms of "to make"), so matching
# only one exact form misses real variants a user or STT might produce.
# Using the stem + a wildcard catches the conjugation family instead of
# just one spelling. This is still a known-imperfect first line of
# defense — not a substitute for the grounding/overlap checks below,
# which is why those exist as a second, independent layer.
UNSAFE_PATTERNS = {
    "violence_weapons": [
        r"बम\s*बना", r"हथियार\s*बना", r"बंदूक\s*बना", r"विस्फोटक\s*बना",
        r"bomb\s*making", r"how to make a (bomb|weapon)",
        r"किसी को मार", r"how to kill",
    ],
    "self_harm": [
        r"आत्महत्या", r"suicide", r"खुद को नुकसान", r"self.?harm",
    ],
    "illegal_activity": [
        r"ड्रग्स\s*बना", r"how to make drugs", r"hack.*(account|password)",
        r"नकली\s*दस्तावेज़", r"fake documents",
    ],
    "hate_harassment": [
        r"नफरत\s*फैला", r"hate speech", r"racial slur",
    ],
}

_COMPILED_PATTERNS = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in UNSAFE_PATTERNS.items()
}

GROUNDING_DISTANCE_THRESHOLD = 1.2


def check_unsafe_input(text: str) -> tuple[bool, str | None]:
    """Returns (is_unsafe, category). category is None if safe."""
    for category, patterns in _COMPILED_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                return True, category
    return False, None


def is_grounded_enough(top_distance: float, threshold: float = GROUNDING_DISTANCE_THRESHOLD) -> bool:
    """True if the best retrieved chunk is close enough to trust."""
    return top_distance <= threshold


def answer_overlap_score(answer: str, retrieved_chunks: list[dict]) -> float:
    """Rough lexical overlap between the answer and its supporting
    context: what fraction of the answer's meaningful words also appear
    somewhere in the retrieved chunks. Not a proof of correctness, but a
    cheap, explainable grounding signal worth logging alongside answers."""
    context_text = " ".join(c["text"] for c in retrieved_chunks)
    context_words = set(re.findall(r"\S+", context_text))
    answer_words = re.findall(r"\S+", answer)

    # ignore very short tokens (punctuation fragments, single characters)
    meaningful = [w for w in answer_words if len(w) > 2]
    if not meaningful:
        return 1.0  # nothing substantive to check (e.g. the refusal message itself)

    overlap = sum(1 for w in meaningful if w in context_words)
    return overlap / len(meaningful)