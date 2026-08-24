# 🎙️ RAGInGoa — Hindi Voice RAG

### Hacker House Goa 2026

**Hindi Voice → Speech-to-Text → E5 Retrieval + Lightweight Lexical Retrieval → Grounded Answer**

RAGInGoa is a production-oriented Hindi Voice Retrieval-Augmented Generation (RAG) system designed to answer questions from a large Hindi passage corpus.

The system combines:

- 🎙️ Voice input
- 🗣️ Speech-to-Text (STT)
- 🔎 Multilingual E5 semantic retrieval
- ⚡ Lightweight lexical retrieval
- 🔀 Hybrid ranking
- 🛡️ Grounding and guardrails
- 🤖 Grounded answer generation
- 📊 Retrieval and latency evaluation
- ☁️ Streamlit Cloud deployment

---

## 🌐 Live Demo

The application is deployed using Streamlit Cloud.

**Live application:**

`https://voicerag-kwfsigsmmsysj6dhyudkza.streamlit.app/`

> If the application is temporarily sleeping, the first request may take longer because the hosted environment needs to initialize the model and retrieval components.

---

# 🎯 Problem Statement

Most RAG systems are optimized primarily for English text.

RAGInGoa focuses on a Hindi voice-first retrieval pipeline where a user can ask a question naturally in Hindi and receive a concise grounded answer.

For example:

> भारत की राजधानी क्या है?

The system processes the question through:

```text
Voice
  ↓
Speech-to-Text
  ↓
Hindi Query
  ↓
Semantic Retrieval
  +
Lexical Retrieval
  ↓
Hybrid Ranking
  ↓
Relevant Hindi Passages
  ↓
Grounded Answer
