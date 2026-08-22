"""
Day 4 (part 3) — Streamlit deployment app

Wraps harness.py in a simple web UI: record/upload a Hindi voice question
(or type text directly for quick testing), run it through the full
pipeline, and display the answer along with retrieved sources, guardrail
status, and per-stage latency — transparency that's useful both for your
demo video and for anyone evaluating the submission.

Run locally:
    streamlit run src/streamlit_app.py

Deploy: push this repo to GitHub, then deploy on Streamlit Community
Cloud pointing at src/streamlit_app.py, with ELEVENLABS_API_KEY and
GROQ_API_KEY added as app secrets (Settings -> Secrets), not committed
to the repo.
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import run_pipeline, run_pipeline_text, warmup_retrieval  # noqa: E402

st.set_page_config(page_title="RAGInGoa — Hindi Voice RAG", page_icon="🎙️")
st.title("🎙️ RAGInGoa — Hindi Voice-Enabled RAG")
st.caption("HH Goa 2026 · Voice → ElevenLabs STT → ChromaDB retrieval → Groq generation, guardrailed and timed")


@st.cache_resource
def _warmup():
    """Loads the embedding model + ChromaDB collection once per server
    process (not once per user request) — Streamlit's cache_resource
    keeps this across reruns/users, which is exactly the warmup behavior
    we want in production."""
    warmup_retrieval()
    return True


with st.spinner("Loading model + index (first load only) ..."):
    _warmup()

tab_voice, tab_text = st.tabs(["🎤 Voice question", "⌨️ Text question (quick test)"])

with tab_voice:
    audio_value = st.audio_input("Ask your question in Hindi")
    if audio_value is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_value.getvalue())
            tmp_path = tmp.name

        with st.spinner("Transcribing, retrieving, and generating ..."):
            result = run_pipeline(tmp_path)

        if not result.success:
            st.error(f"Pipeline failed at stage '{result.error_stage}': {result.error_message}")
        else:
            st.subheader("Transcribed question")
            st.write(result.query_text)

            st.subheader("Answer")
            if result.guardrail_triggered:
                st.warning(f"Guardrail triggered: {result.guardrail_triggered}")
            st.write(result.answer)

            with st.expander(f"Retrieved sources ({len(result.retrieved_chunks)})"):
                for c in result.retrieved_chunks:
                    st.markdown(f"**[{c.strategy}]** (distance: {c.distance:.3f})")
                    st.write(c.text)
                    st.divider()

            with st.expander("Latency breakdown"):
                t = result.timings
                st.write(f"STT: {t.stt_ms:.0f}ms" if t.stt_ms else "STT: —")
                st.write(f"Retrieval: {t.retrieval_ms:.0f}ms" if t.retrieval_ms else "Retrieval: —")
                st.write(f"Generation: {t.generation_ms:.0f}ms" if t.generation_ms else "Generation: —")
                st.write(f"**Total: {t.total_ms:.0f}ms**" if t.total_ms else "Total: —")
                if result.grounding_score is not None:
                    st.write(f"Grounding overlap score: {result.grounding_score:.2f}")

with tab_text:
    query_text = st.text_input("Type a Hindi question directly (skips STT)")
    if st.button("Ask") and query_text:
        with st.spinner("Retrieving and generating ..."):
            result = run_pipeline_text(query_text)

        st.subheader("Answer")
        if result.guardrail_triggered:
            st.warning(f"Guardrail triggered: {result.guardrail_triggered}")
        st.write(result.answer)

        with st.expander(f"Retrieved sources ({len(result.retrieved_chunks)})"):
            for c in result.retrieved_chunks:
                st.markdown(f"**[{c.strategy}]** (distance: {c.distance:.3f})")
                st.write(c.text)
                st.divider()

        with st.expander("Latency breakdown"):
            t = result.timings
            st.write(f"Retrieval: {t.retrieval_ms:.0f}ms" if t.retrieval_ms else "Retrieval: —")
            st.write(f"Generation: {t.generation_ms:.0f}ms" if t.generation_ms else "Generation: —")
            st.write(f"**Total: {t.total_ms:.0f}ms**" if t.total_ms else "Total: —")