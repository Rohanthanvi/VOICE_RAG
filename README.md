# 🎙️ RAGInGoa — Hindi Voice RAG

### Hacker House Goa 2026

**Hindi Voice → Speech-to-Text → E5 Semantic Retrieval + Lightweight Lexical Retrieval → Hybrid Ranking → Grounded Answer**

RAGInGoa is a Hindi-first Voice Retrieval-Augmented Generation (RAG) system that allows users to ask questions naturally in Hindi using voice or text and receive concise, grounded answers from a large Hindi knowledge corpus.

## 🌐 Live Demo

🚀 **Live Application:**  
https://voicerag-kwfsigsmmsysj6dhyudkza.streamlit.app/

🔗 **GitHub Repository:**  
https://github.com/Rohanthanvi/VOICE_RAG

---

# 🎯 Project Overview

Traditional RAG systems often rely only on semantic vector search. However, semantic similarity can sometimes retrieve passages that are related to a query but do not contain the exact answer.

RAGInGoa addresses this by combining:

- 🎙️ Hindi Voice Input
- 🗣️ Speech-to-Text
- 🔎 Multilingual E5 Semantic Retrieval
- ⚡ Lightweight Lexical Retrieval
- 🔀 Reciprocal Rank Fusion (RRF)
- 🧠 Relevance Boosting
- 🛡️ Grounding and Guardrails
- 🤖 Grounded Answer Generation
- 📊 Retrieval Evaluation
- ⏱️ Latency Benchmarking
- ☁️ Streamlit Cloud Deployment

The result is an end-to-end Hindi Voice RAG pipeline.

---

# 🧠 System Architecture

```text
                    🎙️ USER VOICE
                          │
                          ▼
                 ┌─────────────────┐
                 │ Speech-to-Text  │
                 │      (STT)      │
                 └────────┬────────┘
                          │
                          ▼
                   Hindi Text Query
                          │
                ┌─────────┴─────────┐
                │                   │
                ▼                   ▼
        ┌──────────────┐     ┌──────────────┐
        │ E5 Semantic  │     │   Lexical    │
        │  Retrieval   │     │  Retrieval   │
        └──────┬───────┘     └──────┬───────┘
               │                    │
               ▼                    ▼
          ChromaDB             Token Matching
               │                    │
               └─────────┬──────────┘
                         ▼
                 Hybrid Ranking
                         │
                         ▼
                  RRF + Relevance
                     Boosting
                         │
                         ▼
                    Top-K Context
                         │
                         ▼
                  Grounded Answer
                     Generation
                         │
                         ▼
                  🗣️ Hindi Answer


🔬 Retrieval Stress Tests
भारत की राजधानी क्या है?

भारत की वित्तीय राजधानी कौन सी है?

किस देश की राजधानी बुडापेस्ट है?

डेनमार्क की राजधानी कहाँ है?

चिली की राजधानी कौन सी है?

दिल्ली में कौन-कौन सी भाषाएँ बोली जाती हैं?

भारत में सबसे बड़े शहर कौन से हैं?


Recommended Voice Demo

For the live demonstration, use natural Hindi speech.

Recommended questions:
भारत की राजधानी क्या है?

भारत की वित्तीय राजधानी कौन सी है?

दिल्ली में कौन सी भाषा सबसे ज्यादा बोली जाती है?

डेनमार्क की राजधानी क्या है?

चिली की राजधानी कौन सी है?

👨‍💻 Author

Rohan Narendra Thanvi

👨‍💻 Co-Author
Khyati Sinha

B.Tech — Computer Science & Engineering

Project: RAGInGoa — Hindi Voice RAG

If you find this project useful, consider giving the repository a ⭐ star.


