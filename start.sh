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
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-3}"
BACKEND_READY_ATTEMPTS="${BACKEND_READY_ATTEMPTS:-240}"

mkdir -p "${CHECKPOINTS_DIR}" "${VOICES_DIR}"

# Fish Speech resolves checkpoints/ and references/ relative to its project root.
ln -sfn "${CHECKPOINTS_DIR}" "${FISH_DIR}/checkpoints"
ln -sfn "${VOICES_DIR}" "${FISH_DIR}/references"

weights_present() {
  # Require the decoder checkpoint — incomplete downloads must not count as ready.
  [[ -f "${S2_PRO_DIR}/codec.pth" ]] || return 1
  # And at least one model weight file.
  if [[ -z "$(find "${S2_PRO_DIR}" -type f \( -name "*.pth" -o -name "*.safetensors" -o -name "*.bin" \) 2>/dev/null | head -n 1)" ]]; then
    return 1
  fi
  return 0
}

download_weights() {
  local tmp_dir="${S2_PRO_DIR}.partial"
  local attempt

  if ! command -v hf >/dev/null 2>&1; then
    log "ERROR: hf CLI not found (install huggingface_hub)"
    return 1
  fi

  rm -rf "${tmp_dir}"
  mkdir -p "${tmp_dir}"

  for attempt in $(seq 1 "${DOWNLOAD_RETRIES}"); do
    log "Downloading fishaudio/s2-pro (attempt ${attempt}/${DOWNLOAD_RETRIES})..."
    if hf download fishaudio/s2-pro --local-dir "${tmp_dir}"; then
      if [[ -f "${tmp_dir}/codec.pth" ]]; then
        rm -rf "${S2_PRO_DIR}"
        mv "${tmp_dir}" "${S2_PRO_DIR}"
        log "Weight download complete."
        return 0
      fi
      log "Download finished but codec.pth missing — retrying..."
    else
      log "hf download failed on attempt ${attempt}"
    fi
    rm -rf "${tmp_dir}"
    mkdir -p "${tmp_dir}"
    sleep $((attempt * 5))
  done

  rm -rf "${tmp_dir}"
  log "ERROR: failed to download fishaudio/s2-pro after ${DOWNLOAD_RETRIES} attempts"
  return 1
}

if ! weights_present; then
  if [[ "${SKIP_WEIGHT_DOWNLOAD}" == "1" ]]; then
    log "ERROR: checkpoints/s2-pro is incomplete and SKIP_WEIGHT_DOWNLOAD=1"
    exit 1
  fi
  download_weights
  if ! weights_present; then
    log "ERROR: checkpoints/s2-pro still incomplete after download"
    exit 1
  fi
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

FISH_PID=""
WATCHDOG_PID=""
UVICORN_PID=""

cleanup() {
  log "Shutting down..."
  if [[ -n "${WATCHDOG_PID}" ]] && kill -0 "${WATCHDOG_PID}" 2>/dev/null; then
    kill "${WATCHDOG_PID}" 2>/dev/null || true
  fi
  if [[ -n "${UVICORN_PID}" ]] && kill -0 "${UVICORN_PID}" 2>/dev/null; then
    kill -TERM "${UVICORN_PID}" 2>/dev/null || true
  fi
  if [[ -n "${FISH_PID}" ]] && kill -0 "${FISH_PID}" 2>/dev/null; then
    kill -TERM "${FISH_PID}" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "${FISH_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "${FISH_PID}" 2>/dev/null || true
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

# If the backend dies after startup, exit so Coolify restarts the whole container.
(
  while kill -0 "${FISH_PID}" 2>/dev/null; do
    sleep 5
  done
  log "ERROR: Fish Speech backend exited unexpectedly — stopping wrapper"
  kill -TERM 1 2>/dev/null || kill -TERM $$ 2>/dev/null || true
) &
WATCHDOG_PID=$!

log "Waiting for Fish Speech backend health..."
ATTEMPTS=0
until curl -fsS --max-time 3 "http://${FISH_SPEECH_HOST}:${FISH_SPEECH_PORT}/v1/health" >/dev/null 2>&1; do
  if ! kill -0 "${FISH_PID}" 2>/dev/null; then
    log "ERROR: Fish Speech backend exited before becoming healthy"
    wait "${FISH_PID}" || true
    exit 1
  fi
  ATTEMPTS=$((ATTEMPTS + 1))
  if [[ "${ATTEMPTS}" -ge "${BACKEND_READY_ATTEMPTS}" ]]; then
    log "ERROR: Timed out waiting for Fish Speech backend"
    exit 1
  fi
  # Log progress every ~30s so Coolify logs aren't silent during model load.
  if (( ATTEMPTS % 15 == 0 )); then
    log "Still waiting for backend… (${ATTEMPTS}/${BACKEND_READY_ATTEMPTS})"
  fi
  sleep 2
done
log "Fish Speech backend is healthy."

log "Starting FastAPI wrapper on 0.0.0.0:${PORT}..."
cd "${APP_DIR}"
# Do not exec — keep this shell as PID 1 so trap/watchdog stay alive.
uvicorn main:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers 1 \
  --timeout-keep-alive 75 \
  --limit-concurrency 32 \
  --log-level info &
UVICORN_PID=$!

# If either child dies, tear down the other and exit for Coolify restart.
wait -n "${FISH_PID}" "${UVICORN_PID}" || true
log "A child process exited — shutting down"
if kill -0 "${UVICORN_PID}" 2>/dev/null; then
  kill -TERM "${UVICORN_PID}" 2>/dev/null || true
  wait "${UVICORN_PID}" 2>/dev/null || true
fi
exit 1
