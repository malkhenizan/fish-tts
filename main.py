"""
Fish TTS — FastAPI wrapper around self-hosted fishaudio/fish-speech.

Architecture:
  Coolify DNS -> :8080 (this app) -> 127.0.0.1:8081 (Fish Speech api_server)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
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
MAX_REFERENCE_AUDIO_BYTES = int(os.getenv("MAX_REFERENCE_AUDIO_BYTES", str(20 * 1024 * 1024)))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "5000"))
UPSTREAM_RETRIES = int(os.getenv("UPSTREAM_RETRIES", "1"))

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

# Shared HTTP client (created in lifespan)
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global http_client
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(REQUEST_TIMEOUT_S, connect=10.0),
        limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
    )
    logger.info("wrapper ready voices_dir=%s", VOICES_DIR)
    try:
        yield
    finally:
        if http_client is not None:
            await http_client.aclose()
            http_client = None


app = FastAPI(
    title="Fish TTS Wrapper",
    version="1.1.0",
    description="Coolify-ready FastAPI gateway for fishaudio/fish-speech (S2-Pro).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Single-GPU job gate
# ---------------------------------------------------------------------------


class JobGate:
    """Reject overlapping TTS jobs so two GPU inferences never run together."""

    def __init__(self) -> None:
        self._state = asyncio.Lock()
        self.busy = False
        self.kind: str | None = None
        self.started_at: float | None = None

    def snapshot(self) -> dict[str, Any]:
        started = self.started_at
        return {
            "busy": self.busy,
            "job": self.kind,
            "started_at": started,
            "running_for_s": (round(time.time() - started, 1) if started else 0),
        }

    @asynccontextmanager
    async def run(self, kind: str):
        async with self._state:
            if self.busy:
                snap = self.snapshot()
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "busy",
                        "message": (
                            "Another generation job is already running. "
                            "Wait for it to finish, then retry."
                        ),
                        **snap,
                    },
                    headers={"Retry-After": "5"},
                )
            self.busy = True
            self.kind = kind
            self.started_at = time.time()
            logger.info("job acquired kind=%s", kind)
        try:
            yield
        finally:
            async with self._state:
                logger.info(
                    "job released kind=%s duration=%.1fs",
                    kind,
                    (time.time() - (self.started_at or time.time())),
                )
                self.busy = False
                self.kind = None
                self.started_at = None


job_gate = JobGate()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _strip_data_url(v: str) -> str:
    v = v.strip()
    if v.lower().startswith("data:") and "," in v:
        return v.split(",", 1)[1]
    return v


class CloneRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    reference_audio: str = Field(..., min_length=1)
    reference_text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
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
        return _strip_data_url(v)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    voice_id: str = Field(..., min_length=1, max_length=255)
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
    reference_text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    reference_audio: str = Field(..., min_length=1)
    filename: str = Field("sample.webm", max_length=255)

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
        return _strip_data_url(v)

    @field_validator("filename")
    @classmethod
    def safe_filename(cls, v: str) -> str:
        name = Path(v).name.strip() or "sample.webm"
        if name in {".", ".."} or "/" in name or "\\" in name:
            return "sample.webm"
        return name


# ---------------------------------------------------------------------------
# Auth + logging
# ---------------------------------------------------------------------------


async def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> None:
    if not API_KEY:
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
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled error path=%s", request.url.path)
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    # Keep health/status quieter in logs
    if request.url.path not in {"/health", "/v1/status"}:
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
    if len(audio) > MAX_REFERENCE_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"reference_audio too large ({len(audio)} bytes). "
                f"Max is {MAX_REFERENCE_AUDIO_BYTES} bytes."
            ),
        )
    return audio


def _assert_voice_exists(voice_id: str) -> Path:
    if not VOICE_ID_RE.match(voice_id):
        raise HTTPException(status_code=400, detail="Invalid voice_id")

    voice_dir = (VOICES_DIR / voice_id).resolve()
    try:
        voice_dir.relative_to(VOICES_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid voice_id path") from exc

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


async def _backend_healthy() -> bool:
    client = http_client
    if client is None:
        return False
    try:
        resp = await client.get(f"{FISH_SPEECH_BASE}/v1/health", timeout=2.0)
        return resp.status_code == 200
    except Exception:  # noqa: BLE001
        return False


async def _proxy_tts(payload: dict) -> Response:
    client = http_client
    if client is None:
        raise HTTPException(status_code=503, detail="HTTP client not ready")

    url = f"{FISH_SPEECH_BASE}/v1/tts"
    last_exc: Exception | None = None
    upstream: httpx.Response | None = None

    attempts = max(1, UPSTREAM_RETRIES + 1)
    for attempt in range(1, attempts + 1):
        try:
            upstream = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            last_exc = None
            break
        except httpx.ConnectError as exc:
            last_exc = exc
            logger.warning("upstream connect failed attempt=%s/%s: %s", attempt, attempts, exc)
            if attempt < attempts:
                await asyncio.sleep(min(2 * attempt, 5))
                continue
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail="Fish Speech backend timed out") from exc

    if last_exc is not None or upstream is None:
        logger.exception("Fish Speech backend unreachable at %s", url)
        raise HTTPException(
            status_code=503,
            detail="Fish Speech backend is unavailable",
        ) from last_exc

    if upstream.status_code != 200:
        detail: object
        try:
            detail = upstream.json()
        except Exception:  # noqa: BLE001
            detail = upstream.text
        logger.error("Upstream TTS failed (%s): %s", upstream.status_code, detail)
        # Map upstream 5xx to 503 so clients can retry cleanly
        code = upstream.status_code if upstream.status_code < 500 else 503
        raise HTTPException(status_code=code, detail=detail)

    if not upstream.content:
        raise HTTPException(status_code=502, detail="Upstream returned empty audio")

    audio_format = payload.get("format", "wav")
    return Response(
        content=upstream.content,
        media_type=_content_type(audio_format),
        headers={
            "Content-Disposition": f'inline; filename="speech.{audio_format}"',
            "X-Engine": "fish-speech",
            "Cache-Control": "no-store",
        },
    )


def _to_wav_bytes(audio_bytes: bytes, filename_hint: str = "sample.webm") -> bytes:
    """Normalize browser recordings (often webm/opus) to wav via ffmpeg."""
    suffix = Path(filename_hint).suffix.lower() or ".webm"
    if suffix == ".wav":
        return audio_bytes

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found — storing/passing original audio bytes")
        return audio_bytes

    with tempfile.TemporaryDirectory(prefix="fish-tts-") as tmp:
        src = Path(tmp) / f"input{suffix}"
        dst = Path(tmp) / "output.wav"
        src.write_bytes(audio_bytes)
        proc = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src),
                "-ac",
                "1",
                "-ar",
                "44100",
                str(dst),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
            err = (proc.stderr or proc.stdout or "unknown ffmpeg error")[-500:]
            raise HTTPException(status_code=400, detail=f"Failed to convert audio to wav: {err}")
        return dst.read_bytes()


async def _require_backend_ready() -> None:
    if not await _backend_healthy():
        raise HTTPException(
            status_code=503,
            detail="Fish Speech backend is not ready yet. Retry shortly.",
            headers={"Retry-After": "10"},
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Coolify / orchestrator health probe."""
    backend_ok = await _backend_healthy()
    payload = {
        "status": "ok" if backend_ok else "starting",
        "backend": backend_ok,
        "job": job_gate.snapshot(),
    }
    return JSONResponse(content=payload, status_code=200 if backend_ok else 503)


@app.get("/v1/status")
async def job_status():
    backend_ok = await _backend_healthy()
    return {"ok": True, "backend": backend_ok, **job_gate.snapshot()}


@app.post("/v1/clone", dependencies=[Depends(require_api_key)])
async def clone_voice(body: CloneRequest):
    await _require_backend_ready()
    audio_bytes = _to_wav_bytes(_decode_reference_audio(body.reference_audio), "reference.webm")
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
    async with job_gate.run("clone"):
        return await _proxy_tts(payload)


@app.post("/v1/tts", dependencies=[Depends(require_api_key)])
async def text_to_speech(body: TTSRequest):
    await _require_backend_ready()
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
    async with job_gate.run("tts"):
        return await _proxy_tts(payload)


@app.get("/v1/voices", dependencies=[Depends(require_api_key)])
async def list_voices():
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
    """Persist a reference voice pack under /voices/{voice_id} (atomic write)."""
    voice_dir = VOICES_DIR / body.voice_id
    if voice_dir.exists():
        raise HTTPException(status_code=409, detail=f"voice_id '{body.voice_id}' already exists")

    raw = _decode_reference_audio(body.reference_audio)
    wav_bytes = await asyncio.to_thread(_to_wav_bytes, raw, body.filename)

    staging = VOICES_DIR / f".tmp-{body.voice_id}-{os.getpid()}"
    try:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=False)
        (staging / "sample.wav").write_bytes(wav_bytes)
        (staging / "sample.lab").write_text(body.reference_text, encoding="utf-8")
        # Atomic publish
        staging.rename(voice_dir)
    except FileExistsError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise HTTPException(status_code=409, detail=f"voice_id '{body.voice_id}' already exists") from exc
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(staging, ignore_errors=True)
        if voice_dir.exists() and not any(voice_dir.iterdir()):
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
    return FileResponse(index, headers={"Cache-Control": "no-cache"})


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
