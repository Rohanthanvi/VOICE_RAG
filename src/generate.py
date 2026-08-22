"""
Day 3 (part 3) — Answer generation via Groq

Takes the user's query plus the retrieved chunks and asks a fast Groq
model to answer USING ONLY that context. The prompt explicitly instructs
the model to say it doesn't know rather than guess — this is a first,
basic grounding safeguard; Day 4 adds a proper post-hoc check that
verifies the answer actually traces back to the retrieved text.

Needs a GROQ_API_KEY environment variable (in the same .env file as
ELEVENLABS_API_KEY).
"""

import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client: Groq | None = None
MODEL = "openai/gpt-oss-20b"  # llama-3.1-8b-instant was deprecated by Groq (2026-06-17); this is Groq's recommended fast replacement

SYSTEM_PROMPT = """आप एक सहायक हैं जो केवल दिए गए संदर्भ (context) के आधार पर हिंदी में उत्तर देते हैं।
नियम:
1. केवल दिए गए संदर्भ में मौजूद जानकारी का उपयोग करें।
2. अगर संदर्भ में उत्तर नहीं है, तो स्पष्ट रूप से कहें "मुझे इसका उत्तर उपलब्ध जानकारी में नहीं मिला।"
3. जानकारी न बनाएं (hallucinate न करें)।
4. संक्षिप्त और स्पष्ट उत्तर दें।"""


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. Add it to a .env file at the project root.")
        _client = Groq(api_key=api_key)
    return _client


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(retrieved_chunks))

    user_prompt = f"""संदर्भ:
{context}

प्रश्न: {query}

उत्तर:"""

    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # low temperature: we want grounded, consistent answers, not creative ones
        max_tokens=300,
        reasoning_effort="low",   # gpt-oss is a reasoning model; low effort keeps latency down for this use case
        include_reasoning=False,  # keep only the final answer, not the chain-of-thought trace
    )
    answer = response.choices[0].message.content.strip()
    # defensive cleanup in case a reasoning trace slips through in <think> tags
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    return answer


if __name__ == "__main__":
    fake_chunks = [{"text": "भारत की राजधानी नई दिल्ली है।"}]
    print(generate_answer("भारत की राजधानी क्या है?", fake_chunks))