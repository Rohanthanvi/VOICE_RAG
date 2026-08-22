"""
RAGInGoa — Hindi Voice-Enabled RAG

Streamlit deployment app for:
Voice/Text → STT → Hybrid E5 + BM25 Retrieval
→ Grounded Answer → Guardrails → Latency

Run locally:
    streamlit run src/streamlit_app.py

Deploy:
    Streamlit Community Cloud
    Main file:
        src/streamlit_app.py
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st


# ============================================================================
# PATH SETUP
# ============================================================================

SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================================
# IMPORT PRODUCTION HARNESS
# ============================================================================

from harness import (  # noqa: E402
    run_pipeline_audio,
    run_pipeline_text,
    warmup,
)


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="RAGInGoa — Hindi Voice RAG",
    page_icon="🎙️",
    layout="wide",
)


# ============================================================================
# HEADER
# ============================================================================

st.title("🎙️ RAGInGoa — Hindi Voice-Enabled RAG")

st.caption(
    "HH Goa 2026 · Voice → ElevenLabs STT → "
    "Hybrid E5 + BM25 Retrieval → Grounded Answer"
)


# ============================================================================
# WARMUP
# ============================================================================

@st.cache_resource
def _warmup():
    """
    Load the E5 model, ChromaDB and BM25 index once per Streamlit process.

    Streamlit cache_resource keeps these heavy objects alive across
    reruns and users.
    """
    warmup()
    return True


with st.spinner(
    "Loading E5 model + ChromaDB + BM25 index "
    "(first load only)..."
):
    _warmup()


# ============================================================================
# HELPERS
# ============================================================================

def format_distance(distance):
    """Safely format optional retrieval distance."""
    if distance is None:
        return "—"

    try:
        return f"{float(distance):.3f}"
    except Exception:
        return "—"


def display_result(result, include_stt=True):
    """Display a PipelineResult in a consistent format."""

    if not result.success:
        st.error(
            f"Pipeline failed at stage "
            f"'{result.error_stage}': "
            f"{result.error_message}"
        )
        return

    # ------------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------------

    st.subheader("Question")

    st.write(result.query_text)

    # ------------------------------------------------------------------------
    # Answer
    # ------------------------------------------------------------------------

    st.subheader("Answer")

    if result.guardrail_triggered:
        st.warning(
            f"Guardrail triggered: "
            f"{result.guardrail_triggered}"
        )

    st.success(result.answer)

    # ------------------------------------------------------------------------
    # Retrieved Sources
    # ------------------------------------------------------------------------

    with st.expander(
        f"Retrieved sources ({len(result.retrieved_chunks)})"
    ):
        if not result.retrieved_chunks:
            st.write("No retrieval results.")
        else:
            for index, chunk in enumerate(
                result.retrieved_chunks,
                start=1,
            ):
                st.markdown(
                    f"### Source #{index}"
                )

                st.markdown(
                    f"**Strategy:** `{chunk.strategy}`"
                )

                st.markdown(
                    f"**Distance:** "
                    f"`{format_distance(chunk.distance)}`"
                )

                if chunk.e5_rank is not None:
                    st.markdown(
                        f"**E5 rank:** `{chunk.e5_rank}`"
                    )

                if chunk.bm25_rank is not None:
                    st.markdown(
                        f"**BM25 rank:** `{chunk.bm25_rank}`"
                    )

                st.markdown(
                    f"**BM25 score:** "
                    f"`{chunk.bm25_score:.4f}`"
                )

                st.write(chunk.text)

                if index < len(
                    result.retrieved_chunks
                ):
                    st.divider()

    # ------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------

    with st.expander("Pipeline metrics"):

        timing = result.timings

        if include_stt:
            if timing.stt_ms is not None:
                st.metric(
                    "STT",
                    f"{timing.stt_ms:.0f} ms",
                )
            else:
                st.metric(
                    "STT",
                    "—",
                )

        if timing.retrieval_ms is not None:
            st.metric(
                "Retrieval",
                f"{timing.retrieval_ms:.0f} ms",
            )

        if timing.generation_ms is not None:
            st.metric(
                "Answer selection",
                f"{timing.generation_ms:.0f} ms",
            )

        st.metric(
            "Total",
            f"{timing.total_ms:.0f} ms",
        )

        if result.grounding_score is not None:
            st.metric(
                "Grounding score",
                f"{result.grounding_score:.2f}",
            )

        if result.answer_mode:
            st.write(
                f"**Answer mode:** `{result.answer_mode}`"
            )


# ============================================================================
# TABS
# ============================================================================

tab_voice, tab_text = st.tabs(
    [
        "🎤 Voice question",
        "⌨️ Text question",
    ]
)


# ============================================================================
# VOICE TAB
# ============================================================================

with tab_voice:

    st.subheader(
        "Ask a question in Hindi"
    )

    audio_value = st.audio_input(
        "Record your question"
    )

    if audio_value is not None:

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as tmp:

            tmp.write(
                audio_value.getvalue()
            )

            tmp_path = tmp.name

        try:

            with st.spinner(
                "Transcribing → retrieving → answering..."
            ):

                result = run_pipeline_audio(
                    tmp_path,
                    language_code="hin",
                )

            display_result(
                result,
                include_stt=True,
            )

        finally:

            try:
                Path(tmp_path).unlink(
                    missing_ok=True
                )
            except Exception:
                pass


# ============================================================================
# TEXT TAB
# ============================================================================

with tab_text:

    st.subheader(
        "Test without voice"
    )

    query_text = st.text_input(
        "Type a Hindi question",
        placeholder="भारत की राजधानी क्या है?",
    )

    ask = st.button(
        "Ask",
        type="primary",
        use_container_width=True,
    )

    if ask:

        if not query_text.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Retrieving and generating answer..."
            ):

                result = run_pipeline_text(
                    query_text
                )

            display_result(
                result,
                include_stt=False,
            )


# ============================================================================
# FOOTER
# ============================================================================

st.divider()

st.caption(
    "RAGInGoa · Hindi Voice RAG · "
    "E5 + BM25 Hybrid Retrieval · "
    "Grounded Answering"
)