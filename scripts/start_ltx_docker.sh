#!/usr/bin/env bash
#
# Starts an LTX-2.5 vLLM-Omni server in Docker and waits until it accepts
# requests. Same /v1/videos/sync endpoint as the Cosmos3 server, so main.py
# only needs `--backend ltx-<pipeline>` (same names as --pipeline here).
#
# Prerequisites (one-time):
#   - NVIDIA driver + nvidia-container-toolkit (`docker run --gpus all` works)
#   - Accept BOTH Lightricks licenses on Hugging Face (Lightricks/LTX-2.5 and
#     Lightricks/LTX-2.5-Diffusers), then `uvx hf@latest auth login` on the host
#     (or export HF_TOKEN)
#   - An image with vLLM >= 0.27 and recent vLLM-Omni main (override with IMAGE=)
#
# Pipelines (vLLM-Omni --model-class-name), fixed per server start:
#   distilled-one  LTX2DistilledOneStagePipeline   8 steps,  default 960x544
#   distilled-two  LTX2DistilledTwoStagePipeline   8+3 steps, default 1920x1088
#   full-one       LTX2Pipeline                    30 steps, default 960x544
#   full-two       LTX2TwoStagePipeline            30+3 steps, default 1920x1088
#
# Usage:
#   ./start_ltx_server.sh                                  # distilled-two
#   ./start_ltx_server.sh --pipeline distilled-one --fp8
#   ./start_ltx_server.sh --pipeline full-one --gpus 0,1 --tp 2 --port 8000
#
set -euo pipefail

# Config
CONTAINER_NAME="${CONTAINER_NAME:-ltx25-server}"
IMAGE="${IMAGE:-vllm/vllm-omni:latest}"
MODEL="${MODEL:-Lightricks/LTX-2.5-Diffusers}"
PIPELINE_KEY="${PIPELINE:-distilled-two}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-all}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
QUANT_FP8="${QUANT_FP8:-false}"            # online FP8 (Hopper/Blackwell only)
INIT_TIMEOUT="${INIT_TIMEOUT:-1800}"
VIDEO_SYNC_TIMEOUT="${VIDEO_SYNC_TIMEOUT:-3600}"
CREATE_START_TIMEOUT="${CREATE_START_TIMEOUT:-120}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(pwd)}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/home/iis/.cache/huggingface}"
LOG_DIR="${LOG_DIR:-./server_logs}"
READY_POLL_INTERVAL_S=10

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_ltx25_server.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

dump_container_state() {
    local name="$1"
    log "---- docker inspect state for '$name' ----"
    docker inspect "$name" \
        --format 'Status: {{.State.Status}} ExitCode: {{.State.ExitCode}} OOMKilled: {{.State.OOMKilled}} Error: {{.State.Error}}' \
        2>&1 | tee -a "$LOG_FILE" || log "  (container may not exist)"
    log "---- last 50 lines of container logs ----"
    docker logs --tail 50 "$name" 2>&1 | tee -a "$LOG_FILE" || log "  (no logs available)"
}

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pipeline) PIPELINE_KEY="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --tp) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --fp8) QUANT_FP8="true"; shift ;;
        *) log "unknown flag: $1"; exit 1 ;;
    esac
done

case "$PIPELINE_KEY" in
    distilled-one) MODEL_CLASS="LTX2DistilledOneStagePipeline" ;;
    distilled-two) MODEL_CLASS="LTX2DistilledTwoStagePipeline" ;;
    full-one)      MODEL_CLASS="LTX2Pipeline" ;;
    full-two)      MODEL_CLASS="LTX2TwoStagePipeline" ;;
    *) log "ERROR: unknown pipeline '$PIPELINE_KEY' (distilled-one|distilled-two|full-one|full-two)"; exit 1 ;;
esac

# Sanity checks
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi not found. Install the NVIDIA driver first."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv | tee -a "$LOG_FILE"
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "$GPU_COUNT" -lt "$TENSOR_PARALLEL_SIZE" ]]; then
    log "ERROR: tensor-parallel-size ($TENSOR_PARALLEL_SIZE) exceeds visible GPU count ($GPU_COUNT)"
    exit 1
fi

if [[ ! -d "$HF_CACHE_DIR" && -z "${HF_TOKEN:-}" ]]; then
    log "WARNING: $HF_CACHE_DIR not found and HF_TOKEN unset. The model is gated; run: uvx hf@latest auth login"
fi

# Remove a previous LTX container; refuse to start if another container
# (e.g. the Cosmos3 server) already publishes this port.
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log "removing existing container '$CONTAINER_NAME'"
    docker rm -f "$CONTAINER_NAME" 2>&1 | tee -a "$LOG_FILE" || true
fi
if docker ps --format '{{.Names}} {{.Ports}}' | grep -q ":${PORT}->"; then
    log "ERROR: port $PORT is already published by another container:"
    docker ps --format '  {{.Names}} {{.Ports}}' | grep ":${PORT}->" | tee -a "$LOG_FILE"
    log "Stop it first (e.g. docker stop cosmos3-nano-server) or use --port."
    exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    log "image '$IMAGE' not found locally, pulling (this can take a while)..."
    docker pull "$IMAGE" 2>&1 | tee -a "$LOG_FILE"
    if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
        log "ERROR: docker pull failed for '$IMAGE'"
        exit 1
    fi
fi

# Launch
EXTRA_ENV=()
[[ -n "${HF_TOKEN:-}" ]] && EXTRA_ENV+=(-e "HF_TOKEN=$HF_TOKEN")

SERVE_ARGS=(
    vllm serve "$MODEL"
    --omni
    --model-class-name "$MODEL_CLASS"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --allowed-local-media-path /workspace
    --port 8000
    --init-timeout "$INIT_TIMEOUT"
)
[[ "$TENSOR_PARALLEL_SIZE" -gt 1 ]] && SERVE_ARGS+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
[[ "$QUANT_FP8" == "true" ]] && SERVE_ARGS+=(--quantization fp8)

log "starting '$CONTAINER_NAME' (image=$IMAGE model=$MODEL pipeline=$PIPELINE_KEY/$MODEL_CLASS gpus=$GPUS tp=$TENSOR_PARALLEL_SIZE fp8=$QUANT_FP8 port=$PORT)"

RUN_CMD=(docker run -d
    --name "$CONTAINER_NAME"
    --gpus "$GPUS"
    -e HF_HUB_DISABLE_XET=1
    -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT="$VIDEO_SYNC_TIMEOUT"
    ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}
    -v "$HF_CACHE_DIR:/root/.cache/huggingface"
    -v "$WORKSPACE_DIR:/workspace"
    -p "${PORT}:8000"
    --ipc=host
    --restart unless-stopped
    "$IMAGE"
    "${SERVE_ARGS[@]}"
)
log "exact docker run command: ${RUN_CMD[*]}"

set +e
timeout "$CREATE_START_TIMEOUT" "${RUN_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
RUN_STATUS=${PIPESTATUS[0]}
set -e
if [[ "$RUN_STATUS" -ne 0 ]]; then
    log "ERROR: 'docker run' failed or timed out (status $RUN_STATUS)."
    dump_container_state "$CONTAINER_NAME"
    exit 1
fi

CONTAINER_STATUS=$(docker inspect "$CONTAINER_NAME" --format '{{.State.Status}}' 2>&1) || CONTAINER_STATUS="unknown"
if [[ "$CONTAINER_STATUS" != "running" ]]; then
    log "ERROR: container not running (status: $CONTAINER_STATUS)."
    dump_container_state "$CONTAINER_NAME"
    exit 1
fi

docker logs -f "$CONTAINER_NAME" >>"$LOG_FILE" 2>&1 &
TAIL_PID=$!

log "polling http://localhost:${PORT}/v1/models every ${READY_POLL_INTERVAL_S}s (timeout ${INIT_TIMEOUT}s)..."
elapsed=0
while (( elapsed < INIT_TIMEOUT )); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        kill "$TAIL_PID" 2>/dev/null || true
        log "server is READY at http://localhost:${PORT}"
        log "run:  python main.py --backend ltx-${PIPELINE_KEY}"
        log "full startup log: $LOG_FILE"
        exit 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        kill "$TAIL_PID" 2>/dev/null || true
        log "ERROR: container exited during startup."
        dump_container_state "$CONTAINER_NAME"
        exit 1
    fi
    sleep "$READY_POLL_INTERVAL_S"
    elapsed=$((elapsed + READY_POLL_INTERVAL_S))
    log "still waiting... (${elapsed}s elapsed)"
done

kill "$TAIL_PID" 2>/dev/null || true
log "ERROR: server did not become ready within ${INIT_TIMEOUT}s"
dump_container_state "$CONTAINER_NAME"
exit 1