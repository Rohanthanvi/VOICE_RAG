from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "src"))

load_dotenv(PROJECT_DIR / ".env")


# ============================================================
# IMPORT EXISTING PIPELINE
# ============================================================

from harness import warmup, run_pipeline_text
from stt import transcribe_audio


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Hacker House Goa RAG API",
    description="Hindi voice/text RAG system using E5 + BM25",
    version="1.0.0",
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QueryRequest(BaseModel):
    query: str


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    """
    Load E5, Chroma and BM25 once when the server starts.
    """

    print("=" * 70)
    print("HACKER HOUSE GOA — API STARTUP")
    print("=" * 70)

    start = time.perf_counter()

    try:
        warmup()

        elapsed = (time.perf_counter() - start) * 1000

        print()
        print("=" * 70)
        print(f"READY — warmup completed in {elapsed:.2f} ms")
        print("=" * 70)

    except Exception as exc:
        print()
        print("=" * 70)
        print("STARTUP WARMUP FAILED")
        print("=" * 70)
        print(str(exc))
        raise


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Hacker House Goa RAG",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "rag-api",
    }


# ============================================================
# TEXT QUERY
# ============================================================

@app.post("/query")
def query(request: QueryRequest):

    query_text = request.query.strip()

    if not query_text:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    try:

        result = run_pipeline_text(query_text)

        return result.model_dump(
            mode="json"
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# VOICE QUERY
# ============================================================

@app.post("/voice")
async def voice(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Audio file is required.",
        )

    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    allowed_extensions = {
        ".wav",
        ".mp3",
        ".m4a",
        ".ogg",
        ".flac",
        ".webm",
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. "
                "Use WAV, MP3, M4A, OGG, FLAC or WEBM."
            ),
        )

    # --------------------------------------------------------
    # Temporary audio file
    # --------------------------------------------------------

    temp_dir = PROJECT_DIR / "temp_audio"
    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = int(
        time.time() * 1000
    )

    audio_path = (
        temp_dir
        / f"voice_{timestamp}{extension}"
    )

    try:

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded audio file is empty.",
            )

        audio_path.write_bytes(contents)

        # ----------------------------------------------------
        # STT
        # ----------------------------------------------------

        stt_start = time.perf_counter()

        text = transcribe_audio(
            str(audio_path),
            language_code="hin",
        )

        stt_ms = (
            time.perf_counter()
            - stt_start
        ) * 1000

        text = text.strip()

        if not text:

            raise HTTPException(
                status_code=422,
                detail="Speech could not be transcribed.",
            )

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        result = run_pipeline_text(text)

        response = result.model_dump(
            mode="json"
        )

        # Preserve actual STT timing in API response.

        if "timings" in response:

            response["timings"]["stt_ms"] = stt_ms

            total_ms = (
                response["timings"].get("total_ms")
            )

            if total_ms is not None:

                response["timings"]["total_ms"] = (
                    total_ms + stt_ms
                )

        return response

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:

        # ----------------------------------------------------
        # Cleanup temporary audio
        # ----------------------------------------------------

        try:

            if audio_path.exists():
                audio_path.unlink()

        except Exception:
            pass