# syntax=docker/dockerfile:1.7
#
# Coolify / NVIDIA RTX 3090 (24GB) optimized image.
#
# Build args (match host driver CUDA):
#   CUDA 12.6 -> UV_EXTRA=cu126  (default; broad driver compatibility)
#   CUDA 12.8 -> UV_EXTRA=cu128
#   CUDA 12.9 -> UV_EXTRA=cu129  (upstream fish-speech default)
#
# fish-speech is cloned at build time (Coolify-safe). The git submodule at
# ./fish-speech is for local development / tracking upstream updates.

ARG CUDA_VER=12.6.0
ARG UBUNTU_VER=24.04
ARG UV_EXTRA=cu126
ARG PY_VER=3.12
ARG UV_VERSION=0.8.15
ARG FISH_SPEECH_REF=main

FROM nvidia/cuda:${CUDA_VER}-cudnn-runtime-ubuntu${UBUNTU_VER}

ARG UV_EXTRA
ARG PY_VER
ARG UV_VERSION
ARG FISH_SPEECH_REF

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_EXTRA=${UV_EXTRA} \
    APP_DIR=/app \
    FISH_DIR=/app/fish-speech \
    CHECKPOINTS_DIR=/app/checkpoints \
    VOICES_DIR=/app/voices \
    PORT=8080 \
    FISH_SPEECH_HOST=127.0.0.1 \
    FISH_SPEECH_PORT=8081 \
    DEVICE=cuda \
    COMPILE=1 \
    HALF=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        ffmpeg \
        build-essential \
        cmake \
        python3 \
        python3-dev \
        python3-pip \
        libsox-dev \
        libasound-dev \
        portaudio19-dev \
        libportaudio2 \
        libportaudiocpp0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /uvx /bin/

WORKDIR /app

# Clone official fish-speech (pinned via FISH_SPEECH_REF).
RUN git clone --depth 1 --branch "${FISH_SPEECH_REF}" \
      https://github.com/fishaudio/fish-speech.git /app/fish-speech

WORKDIR /app/fish-speech

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python pin ${PY_VER} \
    && uv sync --extra ${UV_EXTRA} --frozen \
    && uv pip install "huggingface_hub>=0.26.0"

# Wrapper API — separate venv so FastAPI deps stay independent of fish-speech pins
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv \
    && uv pip install --python /app/.venv/bin/python -r /app/requirements.txt

COPY main.py /app/main.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh \
    && mkdir -p /app/checkpoints /app/voices \
    && ln -sfn /app/checkpoints /app/fish-speech/checkpoints \
    && ln -sfn /app/voices /app/fish-speech/references

ENV PATH="/app/.venv/bin:/app/fish-speech/.venv/bin:${PATH}" \
    VIRTUAL_ENV=/app/.venv

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

ENTRYPOINT ["/app/start.sh"]
