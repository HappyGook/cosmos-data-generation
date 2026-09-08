#!/usr/bin/env bash
#
# Starts the Cosmos3-Nano vLLM-Omni server natively on the port, using the
# `cu128` torch backend.
# the cu128 build is only available via the
# native venv install path documented in the Cosmos3 cookbook:
#   https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/README.md#option-2-native-venv
#
# Prerequisites (one-time):
#   - NVIDIA driver + CUDA 12.x user-mode libraries
#   - `uv` installed (https://docs.astral.sh/uv/getting-started/installation/)
#   - HF auth for the gated Cosmos3 model repos:
#       uvx hf@latest auth login
#
# Usage:
#   ./start_cosmos3_server_native.sh
#   ./start_cosmos3_server_native.sh --tp 2 --port 8000
#
set -euo pipefail

# Config
VENV_DIR="${VENV_DIR:-./.venv-cosmos3}"
TORCH_BACKEND="${TORCH_BACKEND:-cu128}"       # cu128 for CUDA 12.x driver, cu130 for CUDA 13.x
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
INIT_TIMEOUT="${INIT_TIMEOUT:-1800}"
ALLOWED_MEDIA_PATH="${ALLOWED_MEDIA_PATH:-/}"
NO_GUARDRAILS="${NO_GUARDRAILS:-false}"       # set true to skip the gated guardrail model at startup
LOG_DIR="${LOG_DIR:-./server_logs}"
READY_POLL_INTERVAL_S=10

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_cosmos3_server_native.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --tp) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --backend) TORCH_BACKEND="$2"; shift 2 ;;
        --no-guardrails) NO_GUARDRAILS="true"; shift ;;
        *) log "unknown flag: $1"; exit 1 ;;
    esac
done

# Sanity checks
log "checking for nvidia-smi / GPU visibility..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi not found. Install the NVIDIA driver first."
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv | tee -a "$LOG_FILE"

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
log "detected $GPU_COUNT GPU(s), requesting tensor-parallel-size=$TENSOR_PARALLEL_SIZE"
if [[ "$GPU_COUNT" -lt "$TENSOR_PARALLEL_SIZE" ]]; then
    log "ERROR: tensor-parallel-size ($TENSOR_PARALLEL_SIZE) exceeds visible GPU count ($GPU_COUNT)"
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    log "ERROR: 'uv' not found. Install it: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Create / reuse the venv
if [[ ! -d "$VENV_DIR" ]]; then
    log "creating venv at $VENV_DIR"
    uv venv "$VENV_DIR" --python 3.13 --seed --managed-python 2>&1 | tee -a "$LOG_FILE"
else
    log "reusing existing venv at $VENV_DIR"
fi

# Install / upgrade vllm-omni against the requested torch backend, unless already present
if [[ ! -x "$VENV_DIR/bin/vllm" ]]; then
    log "installing vllm-omni (torch-backend=$TORCH_BACKEND) into $VENV_DIR — this can take a while"
    set +e
    "$VENV_DIR/bin/python" -m uv pip install --python "$VENV_DIR/bin/python" \
        --torch-backend="$TORCH_BACKEND" \
        "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@main" \
        2>&1 | tee -a "$LOG_FILE"
    INSTALL_STATUS=${PIPESTATUS[0]}
    set -e
    if [[ "$INSTALL_STATUS" -ne 0 ]]; then
        log "ERROR: vllm-omni install failed (exit $INSTALL_STATUS). See $LOG_FILE."
        exit 1
    fi
else
    log "vllm already installed at $VENV_DIR/bin/vllm — skipping install (delete $VENV_DIR to force reinstall)"
fi

# Verify torch actually sees the GPU(s) before attempting to serve
log "verifying torch/CUDA visibility in the venv..."
"$VENV_DIR/bin/python" - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

# Build the vllm serve command
SERVE_ARGS=(
    nvidia/Cosmos3-Nano
    --omni
    --model-class-name Cosmos3OmniDiffusersPipeline
    --allowed-local-media-path "$ALLOWED_MEDIA_PATH"
    --port "$PORT"
    --init-timeout "$INIT_TIMEOUT"
)
if [[ "$NO_GUARDRAILS" == "true" ]]; then
    SERVE_ARGS+=(--no-guardrails)
fi

log "starting vllm serve: ${SERVE_ARGS[*]}"

# Run in the background, capture PID, tee logs live
"$VENV_DIR/bin/vllm" serve "${SERVE_ARGS[@]}" >>"$LOG_FILE" 2>&1 &
SERVER_PID=$!
log "vllm serve started with PID $SERVER_PID, logging to $LOG_FILE"

log "polling http://localhost:${PORT}/v1/models every ${READY_POLL_INTERVAL_S}s (timeout ${INIT_TIMEOUT}s)..."
elapsed=0
while (( elapsed < INIT_TIMEOUT )); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        log "server is READY at http://localhost:${PORT}"
        log "full startup log: $LOG_FILE"
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "ERROR: vllm serve process exited unexpectedly. Last lines of $LOG_FILE:"
        tail -n 50 "$LOG_FILE" | tee -a "$LOG_FILE"
        exit 1
    fi
    sleep "$READY_POLL_INTERVAL_S"
    elapsed=$((elapsed + READY_POLL_INTERVAL_S))
    log "still waiting... (${elapsed}s elapsed)"
done

log "ERROR: server did not become ready within ${INIT_TIMEOUT}s"
kill "$SERVER_PID" 2>/dev/null || true
exit 1