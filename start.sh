#!/usr/bin/env bash
set -euo pipefail

log() { echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*"; }

APP_DIR="${APP_DIR:-/app}"
FISH_DIR="${FISH_DIR:-${APP_DIR}/fish-speech}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${APP_DIR}/checkpoints}"
VOICES_DIR="${VOICES_DIR:-${APP_DIR}/voices}"
S2_PRO_DIR="${CHECKPOINTS_DIR}/s2-pro"

PORT="${PORT:-8080}"
FISH_SPEECH_HOST="${FISH_SPEECH_HOST:-127.0.0.1}"
FISH_SPEECH_PORT="${FISH_SPEECH_PORT:-8081}"
DEVICE="${DEVICE:-cuda}"
COMPILE="${COMPILE:-1}"
HALF="${HALF:-0}"
SKIP_WEIGHT_DOWNLOAD="${SKIP_WEIGHT_DOWNLOAD:-0}"

mkdir -p "${CHECKPOINTS_DIR}" "${VOICES_DIR}"

# Fish Speech resolves checkpoints/ and references/ relative to its project root.
ln -sfn "${CHECKPOINTS_DIR}" "${FISH_DIR}/checkpoints"
ln -sfn "${VOICES_DIR}" "${FISH_DIR}/references"

weights_present() {
  # Treat the folder as ready only when core weight artifacts exist.
  if [[ ! -d "${S2_PRO_DIR}" ]]; then
    return 1
  fi
  # Non-empty directory with at least one model file.
  if [[ -z "$(find "${S2_PRO_DIR}" -type f \( -name "*.pth" -o -name "*.safetensors" -o -name "*.bin" -o -name "*.json" \) 2>/dev/null | head -n 1)" ]]; then
    return 1
  fi
  return 0
}

if ! weights_present; then
  if [[ "${SKIP_WEIGHT_DOWNLOAD}" == "1" ]]; then
    log "ERROR: checkpoints/s2-pro is empty and SKIP_WEIGHT_DOWNLOAD=1"
    exit 1
  fi
  log "checkpoints/s2-pro missing or empty — downloading fishaudio/s2-pro..."
  mkdir -p "${S2_PRO_DIR}"
  # huggingface-cli is a deprecated stub now; always use `hf`.
  if ! command -v hf >/dev/null 2>&1; then
    log "ERROR: hf CLI not found (install huggingface_hub)"
    exit 1
  fi
  hf download fishaudio/s2-pro --local-dir "${S2_PRO_DIR}"
  if ! weights_present; then
    log "ERROR: download finished but checkpoints/s2-pro still looks empty"
    exit 1
  fi
  log "Weight download complete."
else
  log "Found existing weights in ${S2_PRO_DIR}"
fi

EXTRA_ARGS=()
if [[ "${COMPILE}" == "1" || "${COMPILE}" == "true" ]]; then
  EXTRA_ARGS+=(--compile)
fi
if [[ "${HALF}" == "1" || "${HALF}" == "true" ]]; then
  EXTRA_ARGS+=(--half)
fi

cleanup() {
  log "Shutting down..."
  if [[ -n "${FISH_PID:-}" ]] && kill -0 "${FISH_PID}" 2>/dev/null; then
    kill "${FISH_PID}" 2>/dev/null || true
    wait "${FISH_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

log "Starting Fish Speech API on ${FISH_SPEECH_HOST}:${FISH_SPEECH_PORT} (device=${DEVICE})..."
cd "${FISH_DIR}"
uv run tools/api_server.py \
  --listen "${FISH_SPEECH_HOST}:${FISH_SPEECH_PORT}" \
  --llama-checkpoint-path "checkpoints/s2-pro" \
  --decoder-checkpoint-path "checkpoints/s2-pro/codec.pth" \
  --decoder-config-name "modded_dac_vq" \
  --device "${DEVICE}" \
  --workers 1 \
  "${EXTRA_ARGS[@]}" &
FISH_PID=$!

log "Waiting for Fish Speech backend health..."
ATTEMPTS=0
MAX_ATTEMPTS="${BACKEND_READY_ATTEMPTS:-180}"
until curl -fsS "http://${FISH_SPEECH_HOST}:${FISH_SPEECH_PORT}/v1/health" >/dev/null 2>&1; do
  if ! kill -0 "${FISH_PID}" 2>/dev/null; then
    log "ERROR: Fish Speech backend exited before becoming healthy"
    wait "${FISH_PID}" || true
    exit 1
  fi
  ATTEMPTS=$((ATTEMPTS + 1))
  if [[ "${ATTEMPTS}" -ge "${MAX_ATTEMPTS}" ]]; then
    log "ERROR: Timed out waiting for Fish Speech backend"
    exit 1
  fi
  sleep 2
done
log "Fish Speech backend is healthy."

log "Starting FastAPI wrapper on 0.0.0.0:${PORT}..."
cd "${APP_DIR}"
exec uvicorn main:app --host 0.0.0.0 --port "${PORT}" --workers 1 --log-level info
