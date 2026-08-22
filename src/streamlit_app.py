"""
RAGInGoa — Production Hindi Voice RAG

Streamlit production application.

Pipeline:

    Voice
      ↓
    ElevenLabs STT
      ↓
    Hindi query
      ↓
    E5 + BM25 Hybrid Retrieval
      ↓
    Grounded Answer
      ↓
    Guardrail
      ↓
    Result

Text mode is also available for fast testing.

Run locally:

    streamlit run src/streamlit_app.py

Required Streamlit secrets:

    GROQ_API_KEY
    ELEVENLABS_API_KEY
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT PRODUCTION PIPELINE
# ============================================================

from harness import (  # noqa: E402
    run_pipeline_audio,
    run_pipeline_text,
    warmup,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAGInGoa — Hindi Voice RAG",
    page_icon="🎙️",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🎙️ RAGInGoa — Hindi Voice RAG")

st.caption(
    "Hacker House Goa 2026 · "
    "Hindi Voice → STT → E5 + BM25 Retrieval → "
    "Grounded Answer"
)


# ============================================================
# PRODUCTION WARMUP
# ============================================================

@st.cache_resource(show_spinner=False)
def _warmup():
    """
    Load the production E5 model, Chroma collection
    and BM25 index once per Streamlit process.

    Streamlit cache_resource ensures that the expensive
    model/index loading is not repeated for every user
    interaction.
    """

    warmup()

    return True


# ============================================================
# STARTUP
# ============================================================

with st.spinner(
    "Loading E5 model + Chroma + BM25 index..."
):
    _warmup()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_distance(distance):
    """
    Safely format retrieval distance.

    Some hybrid/BM25 results intentionally have
    distance=None.
    """

    if distance is None:
        return "—"

    try:
        return f"{float(distance):.3f}"
    except (TypeError, ValueError):
        return "—"


def display_result(result, show_stt: bool = False):
    """
    Display a PipelineResult in the Streamlit UI.
    """

    if not result.success:
        st.error(
            f"Pipeline failed at stage "
            f"'{result.error_stage}': "
            f"{result.error_message}"
        )
        return

    # --------------------------------------------------------
    # Transcription
    # --------------------------------------------------------

    if show_stt:
        st.subheader("🗣️ Transcribed Question")
        st.write(result.query_text)

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    st.subheader("💡 Answer")

    if result.guardrail_triggered:
        st.warning(
            f"Guardrail: {result.guardrail_triggered}"
        )

    st.write(result.answer)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Answer Mode",
            result.answer_mode or "—",
        )

    with col2:
        if result.grounding_score is not None:
            st.metric(
                "Grounding",
                f"{result.grounding_score:.2f}",
            )
        else:
            st.metric(
                "Grounding",
                "—",
            )

    with col3:
        st.metric(
            "Retrieved",
            len(result.retrieved_chunks),
        )

    # --------------------------------------------------------
    # Retrieved Sources
    # --------------------------------------------------------

    with st.expander(
        f"📚 Retrieved Sources "
        f"({len(result.retrieved_chunks)})"
    ):

        for index, chunk in enumerate(
            result.retrieved_chunks,
            start=1,
        ):

            st.markdown(
                f"### Source {index}"
            )

            st.markdown(
                f"**Strategy:** "
                f"`{chunk.strategy}`"
            )

            st.markdown(
                f"**Distance:** "
                f"`{format_distance(chunk.distance)}`"
            )

            if chunk.bm25_rank is not None:
                st.markdown(
                    f"**BM25 Rank:** "
                    f"`{chunk.bm25_rank}`"
                )

            if chunk.e5_rank is not None:
                st.markdown(
                    f"**E5 Rank:** "
                    f"`{chunk.e5_rank}`"
                )

            st.markdown(
                f"**RRF Score:** "
                f"`{chunk.rrf_score:.4f}`"
            )

            st.write(chunk.text)

            if index < len(
                result.retrieved_chunks
            ):
                st.divider()

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    with st.expander("⚡ Latency Breakdown"):

        timings = result.timings

        if timings.stt_ms is not None:
            st.write(
                f"STT: "
                f"{timings.stt_ms:.0f} ms"
            )

        if timings.retrieval_ms is not None:
            st.write(
                f"Retrieval: "
                f"{timings.retrieval_ms:.0f} ms"
            )

        if timings.generation_ms is not None:
            st.write(
                f"Generation: "
                f"{timings.generation_ms:.0f} ms"
            )

        st.write(
            f"**Total: "
            f"{timings.total_ms:.0f} ms**"
        )


# ============================================================
# TABS
# ============================================================

tab_voice, tab_text = st.tabs(
    [
        "🎤 Voice Question",
        "⌨️ Text Question",
    ]
)


# ============================================================
# VOICE TAB
# ============================================================

with tab_voice:

    st.subheader(
        "Ask a question in Hindi"
    )

    st.write(
        "Record your question and the system "
        "will transcribe it, retrieve evidence "
        "and generate a grounded answer."
    )

    audio_value = st.audio_input(
        "🎙️ Record your Hindi question"
    )

    if audio_value is not None:

        # ----------------------------------------------------
        # Save uploaded audio temporarily
        # ----------------------------------------------------

        suffix = ".wav"

        if hasattr(
            audio_value,
            "name",
        ):
            original_name = str(
                audio_value.name
            )

            detected_suffix = (
                Path(original_name).suffix.lower()
            )

            if detected_suffix:
                suffix = detected_suffix

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as tmp:

            tmp.write(
                audio_value.getvalue()
            )

            tmp_path = tmp.name

        # ----------------------------------------------------
        # Run Voice Pipeline
        # ----------------------------------------------------

        try:

            with st.spinner(
                "🎧 Transcribing → retrieving → generating..."
            ):

                result = run_pipeline_audio(
                    tmp_path,
                    language_code="hin",
                )

            display_result(
                result,
                show_stt=True,
            )

        finally:

            # ------------------------------------------------
            # Cleanup temporary audio
            # ------------------------------------------------

            try:
                Path(tmp_path).unlink(
                    missing_ok=True
                )
            except Exception:
                pass


# ============================================================
# TEXT TAB
# ============================================================

with tab_text:

    st.subheader(
        "Quick Text Test"
    )

    st.write(
        "Use this mode to test retrieval "
        "without speech-to-text."
    )

    query_text = st.text_input(
        "Hindi question",
        placeholder="भारत की राजधानी क्या है?",
    )

    ask_clicked = st.button(
        "🔎 Ask",
        type="primary",
    )

    if ask_clicked:

        if not query_text.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Retrieving and generating..."
            ):

                result = run_pipeline_text(
                    query_text.strip()
                )

            display_result(
                result,
                show_stt=False,
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "RAGInGoa · Production Hybrid E5 + BM25 · "
    "Hindi Voice RAG"
)