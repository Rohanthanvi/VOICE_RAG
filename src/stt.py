"""
Day 3 (part 1) — Speech-to-text via ElevenLabs

Takes a path to an audio file (the user's spoken question) and returns
the transcribed text. Kept as a single small function so the harness can
call it, time it, and retry it independently of the other stages.

Needs an ELEVENLABS_API_KEY environment variable (put it in a .env file
at the project root — python-dotenv loads it automatically).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

_client: ElevenLabs | None = None


def get_client() -> ElevenLabs:
    global _client
    if _client is None:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not set. Add it to a .env file at the project root.")
        _client = ElevenLabs(api_key=api_key)
    return _client


def transcribe_audio(audio_path: str, language_code: str = "hin") -> str:
    """Transcribe a spoken-question audio file to text.

    language_code hints the model toward Hindi; ElevenLabs' Scribe model
    can also auto-detect if you pass None, but hinting is faster and more
    reliable when you already know the expected language.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    client = get_client()
    with open(path, "rb") as f:
        result = client.speech_to_text.convert(
            file=f,
            model_id="scribe_v1",
            language_code=language_code,
        )
    return result.text.strip()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/stt.py <path_to_audio_file>")
        sys.exit(1)
    text = transcribe_audio(sys.argv[1])
    print(f"Transcribed text: {text}")