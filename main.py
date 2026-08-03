"""
Fish TTS — FastAPI wrapper around self-hosted fishaudio/fish-speech.

Architecture:
  Coolify DNS -> :8080 (this app) -> 127.0.0.1:8081 (Fish Speech api_server)

Endpoints:
  GET  /health
  GET  /           (simple web UI)
  POST /v1/clone   (zero-shot: text + base64 reference audio + transcript)
  POST /v1/tts     (saved voice: text + voice_id under /voices)
  GET  /v1/voices  (list saved voices)
  POST /v1/voices  (save a recorded/uploaded reference voice)
"""

from __future__ import annotations

import base64
import shutil
import subprocess
import tempfile
import logging
import os
import re
import time
from pathlib import Path
from typing import Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.getenv("API_KEY", "").strip()
FISH_SPEECH_HOST = os.getenv("FISH_SPEECH_HOST", "127.0.0.1")
FISH_SPEECH_PORT = int(os.getenv("FISH_SPEECH_PORT", "8081"))
FISH_SPEECH_BASE = f"http://{FISH_SPEECH_HOST}:{FISH_SPEECH_PORT}"
VOICES_DIR = Path(os.getenv("VOICES_DIR", "/app/voices"))
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "600"))

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus", ".webm"}
VOICE_ID_RE = re.compile(r"^[a-zA-Z0-9\-_ ]+$")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fish-tts")

if not API_KEY:
    logger.warning("API_KEY is not set — /v1/* endpoints are open (dev mode)")
elif len(API_KEY) < 16:
    logger.warning("API_KEY looks short; use a long random secret in production")

security = HTTPBearer(auto_error=False)

app = FastAPI(
    title="Fish TTS Wrapper",
    version="1.0.0",
    description="Coolify-ready FastAPI gateway for fishaudio/fish-speech (S2-Pro).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CloneRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    reference_audio: str = Field(
        ...,
        description="Base64-encoded reference audio (wav/mp3/flac/ogg/...)",
    )
    reference_text: str = Field(
        ...,
        min_length=1,
        description="Exact transcript of the reference audio",
    )
    format: Literal["wav", "pcm", "mp3", "opus"] = "wav"
    temperature: float = Field(0.8, ge=0.1, le=1.0)
    top_p: float = Field(0.8, ge=0.1, le=1.0)
    repetition_penalty: float = Field(1.1, ge=0.9, le=2.0)
    chunk_length: int = Field(200, ge=100, le=1000)
    max_new_tokens: int = Field(1024, ge=0)
    seed: int | None = None
    normalize: bool = True
    use_memory_cache: Literal["on", "off"] = "off"

    @field_validator("reference_audio")
    @classmethod
    def strip_data_url(cls, v: str) -> str:
        v = v.strip()
        if "," in v and v.lower().startswith("data:"):
            v = v.split(",", 1)[1]
        return v


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    voice_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Folder name under /voices (maps to Fish Speech reference_id)",
    )
    format: Literal["wav", "pcm", "mp3", "opus"] = "wav"
    temperature: float = Field(0.8, ge=0.1, le=1.0)
    top_p: float = Field(0.8, ge=0.1, le=1.0)
    repetition_penalty: float = Field(1.1, ge=0.9, le=2.0)
    chunk_length: int = Field(200, ge=100, le=1000)
    max_new_tokens: int = Field(1024, ge=0)
    seed: int | None = None
    normalize: bool = True
    use_memory_cache: Literal["on", "off"] = "on"

    @field_validator("voice_id")
    @classmethod
    def validate_voice_id(cls, v: str) -> str:
        v = v.strip()
        if not VOICE_ID_RE.match(v):
            raise ValueError(
                "voice_id may only contain letters, numbers, hyphens, underscores, and spaces"
            )
        return v


class SaveVoiceRequest(BaseModel):
    voice_id: str = Field(..., min_length=1, max_length=255)
    reference_text: str = Field(..., min_length=1)
    reference_audio: str = Field(..., description="Base64-encoded reference audio")
    filename: str = Field("sample.webm", description="Original filename (used for extension hint)")

    @field_validator("voice_id")
    @classmethod
    def validate_voice_id(cls, v: str) -> str:
        v = v.strip()
        if not VOICE_ID_RE.match(v):
            raise ValueError(
                "voice_id may only contain letters, numbers, hyphens, underscores, and spaces"
            )
        return v

    @field_validator("reference_audio")
    @classmethod
    def strip_data_url(cls, v: str) -> str:
        v = v.strip()
        if "," in v and v.lower().startswith("data:"):
            v = v.split(",", 1)[1]
        return v


# ---------------------------------------------------------------------------
# Auth + logging middleware
# ---------------------------------------------------------------------------


async def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> None:
    """Enforce Bearer API key when API_KEY is configured."""
    if not API_KEY:
        # Dev-friendly: allow unauthenticated access if unset (log a warning once per process).
        return
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "%s %s -> %s (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _content_type(audio_format: str) -> str:
    return {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "pcm": "application/octet-stream",
    }.get(audio_format, "application/octet-stream")


def _decode_reference_audio(b64: str) -> bytes:
    try:
        audio = base64.b64decode(b64, validate=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid base64 reference_audio: {exc}") from exc
    if not audio:
        raise HTTPException(status_code=400, detail="reference_audio decoded to empty bytes")
    return audio


def _assert_voice_exists(voice_id: str) -> Path:
    voice_dir = VOICES_DIR / voice_id
    if not voice_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"voice_id '{voice_id}' not found under {VOICES_DIR}",
        )

    has_audio = False
    has_lab = False
    for path in voice_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            has_audio = True
            if path.with_suffix(".lab").is_file():
                has_lab = True
                break
        if path.suffix == ".lab":
            has_lab = True

    if not has_audio or not has_lab:
        raise HTTPException(
            status_code=400,
            detail=(
                f"voice_id '{voice_id}' is incomplete. Expected at least one audio file "
                f"and a matching .lab transcript inside {voice_dir}"
            ),
        )
    return voice_dir


async def _proxy_tts(payload: dict) -> Response:
    url = f"{FISH_SPEECH_BASE}/v1/tts"
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
            upstream = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
    except httpx.ConnectError as exc:
        logger.exception("Fish Speech backend unreachable at %s", url)
        raise HTTPException(
            status_code=503,
            detail="Fish Speech backend is unavailable",
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Fish Speech backend timed out") from exc

    if upstream.status_code != 200:
        detail: object
        try:
            detail = upstream.json()
        except Exception:  # noqa: BLE001
            detail = upstream.text
        logger.error("Upstream TTS failed (%s): %s", upstream.status_code, detail)
        raise HTTPException(status_code=upstream.status_code, detail=detail)

    audio_format = payload.get("format", "wav")
    return Response(
        content=upstream.content,
        media_type=_content_type(audio_format),
        headers={
            "Content-Disposition": f'inline; filename="speech.{audio_format}"',
            "X-Engine": "fish-speech",
        },
    )


def _to_wav_bytes(audio_bytes: bytes, filename_hint: str = "sample.webm") -> bytes:
    """Normalize browser recordings (often webm/opus) to wav via ffmpeg."""
    suffix = Path(filename_hint).suffix.lower() or ".webm"
    if suffix == ".wav":
        return audio_bytes

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        # Best-effort: store original bytes; Fish Speech may still decode via torchaudio.
        return audio_bytes

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"input{suffix}"
        dst = Path(tmp) / "output.wav"
        src.write_bytes(audio_bytes)
        proc = subprocess.run(
            [ffmpeg, "-y", "-i", str(src), "-ac", "1", "-ar", "44100", str(dst)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not dst.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Failed to convert audio to wav: {proc.stderr[-500:]}",
            )
        return dst.read_bytes()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Coolify / orchestrator health probe."""
    backend_ok = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{FISH_SPEECH_BASE}/v1/health")
            backend_ok = resp.status_code == 200
    except Exception as exc:  # noqa: BLE001
        logger.warning("Backend health check failed: %s", exc)

    payload = {"status": "ok" if backend_ok else "starting", "backend": backend_ok}
    return JSONResponse(content=payload, status_code=200 if backend_ok else 503)


@app.post("/v1/clone", dependencies=[Depends(require_api_key)])
async def clone_voice(body: CloneRequest):
    """
    Zero-shot clone: synthesize `text` using inline reference audio + transcript.
    Returns raw audio bytes.
    """
    audio_bytes = _to_wav_bytes(_decode_reference_audio(body.reference_audio), "reference.webm")
    # Fish Speech ServeReferenceAudio accepts base64 strings in JSON and decodes them.
    payload = {
        "text": body.text,
        "references": [
            {
                "audio": base64.b64encode(audio_bytes).decode("ascii"),
                "text": body.reference_text,
            }
        ],
        "reference_id": None,
        "format": body.format,
        "temperature": body.temperature,
        "top_p": body.top_p,
        "repetition_penalty": body.repetition_penalty,
        "chunk_length": body.chunk_length,
        "max_new_tokens": body.max_new_tokens,
        "seed": body.seed,
        "normalize": body.normalize,
        "use_memory_cache": body.use_memory_cache,
        "streaming": False,
    }
    logger.info(
        "clone request: text_len=%d ref_audio_bytes=%d format=%s",
        len(body.text),
        len(audio_bytes),
        body.format,
    )
    return await _proxy_tts(payload)


@app.post("/v1/tts", dependencies=[Depends(require_api_key)])
async def text_to_speech(body: TTSRequest):
    """
    Synthesize using a persisted voice pack under /voices/{voice_id}.
    Expected files: sample.wav (or other audio) + sample.lab transcript.
    """
    _assert_voice_exists(body.voice_id)
    payload = {
        "text": body.text,
        "references": [],
        "reference_id": body.voice_id,
        "format": body.format,
        "temperature": body.temperature,
        "top_p": body.top_p,
        "repetition_penalty": body.repetition_penalty,
        "chunk_length": body.chunk_length,
        "max_new_tokens": body.max_new_tokens,
        "seed": body.seed,
        "normalize": body.normalize,
        "use_memory_cache": body.use_memory_cache,
        "streaming": False,
    }
    logger.info(
        "tts request: voice_id=%s text_len=%d format=%s",
        body.voice_id,
        len(body.text),
        body.format,
    )
    return await _proxy_tts(payload)


@app.get("/v1/voices", dependencies=[Depends(require_api_key)])
async def list_voices():
    """List available voice_id folders under /voices (handy for frontends)."""
    if not VOICES_DIR.exists():
        return {"voices": []}

    voices: list[str] = []
    for entry in sorted(VOICES_DIR.iterdir()):
        if not entry.is_dir():
            continue
        try:
            _assert_voice_exists(entry.name)
        except HTTPException:
            continue
        voices.append(entry.name)
    return {"voices": voices}


@app.post("/v1/voices", dependencies=[Depends(require_api_key)])
async def save_voice(body: SaveVoiceRequest):
    """Persist a reference voice pack under /voices/{voice_id}."""
    voice_dir = VOICES_DIR / body.voice_id
    if voice_dir.exists():
        raise HTTPException(status_code=409, detail=f"voice_id '{body.voice_id}' already exists")

    raw = _decode_reference_audio(body.reference_audio)
    wav_bytes = _to_wav_bytes(raw, body.filename)

    try:
        voice_dir.mkdir(parents=True, exist_ok=False)
        (voice_dir / "sample.wav").write_bytes(wav_bytes)
        (voice_dir / "sample.lab").write_text(body.reference_text, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        if voice_dir.exists():
            shutil.rmtree(voice_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to save voice: {exc}") from exc

    logger.info("saved voice_id=%s bytes=%d", body.voice_id, len(wav_bytes))
    return {"success": True, "voice_id": body.voice_id}


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/")
async def ui_index():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index)


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
