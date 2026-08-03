# syntax=docker/dockerfile:1.7
#
# Coolify / NVIDIA RTX 3090 (24GB) optimized image.
#
# IMPORTANT: Do NOT use ARG expansion in FROM lines.
# Coolify's BuildKit bake often passes empty --build-arg values, which
# override Dockerfile defaults and produce invalid image refs like:
#   nvidia/cuda:-cudnn-runtime-ubuntu
#
# To change CUDA/uv versions, edit the FROM tags below directly.
# UV_EXTRA must match the CUDA major (cu126 for CUDA 12.6).

FROM ghcr.io/astral-sh/uv:0.8.15 AS uv-bin

FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

ARG UV_EXTRA=cu126
ARG PY_VER=3.12
ARG FISH_SPEECH_REF=main

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

COPY --from=uv-bin /uv /uvx /bin/

WORKDIR /app

RUN git clone --depth 1 --branch "${FISH_SPEECH_REF}" \
      https://github.com/fishaudio/fish-speech.git /app/fish-speech

WORKDIR /app/fish-speech

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python pin ${PY_VER} \
    && uv sync --extra ${UV_EXTRA} --frozen \
    && uv pip install "huggingface_hub>=0.26.0"

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv \
    && uv pip install --python /app/.venv/bin/python -r /app/requirements.txt

COPY main.py /app/main.py
COPY start.sh /app/start.sh
COPY static /app/static
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
